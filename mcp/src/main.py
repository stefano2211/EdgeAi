import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime
from typing import Optional, List, Dict
import logging
from pydantic import BaseModel
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import hashlib
from minio import Minio
from minio.error import S3Error
import pdfplumber
import io
import os

# Inicialización del servicio MCP
mcp = FastMCP("Manufacturing Compliance Processor")

# Configuración global
API_URL = os.getenv("API_URL", "http://api:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sop-pdfs")
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="qdrant", port=6333)

# Inicialización del cliente MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # Cambiar a True en producción con SSL
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        date_formats = []
        try:
            if self.specific_date:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
                date_formats.append(f"Fecha específica: {self.specific_date}")
            else:
                if self.start_date:
                    datetime.strptime(self.start_date, "%Y-%m-%d")
                    date_formats.append(f"Desde: {self.start_date}")
                if self.end_date:
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                    date_formats.append(f"Hasta: {self.end_date}")
                if self.start_date and self.end_date and self.start_date > self.end_date:
                    raise ValueError("La fecha de inicio no puede ser mayor que la de fin")
        except ValueError as e:
            error_msg = f"Formato de fecha inválido. Use YYYY-MM-DD. Detalles: {', '.join(date_formats)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

def init_collections():
    try:
        vector_config = models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
        optimizer_config = models.OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000
        )
        
        qdrant_client.recreate_collection(
            collection_name="mes_logs",
            vectors_config=vector_config,
            optimizers_config=optimizer_config
        )
        
        qdrant_client.recreate_collection(
            collection_name="sop_pdfs",
            vectors_config=vector_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
                memmap_threshold=20000
            )
        )
        logger.info("Colecciones Qdrant inicializadas: mes_logs, sop_pdfs")
    except Exception as e:
        logger.error("Error inicializando Qdrant: %s", str(e))
        raise RuntimeError("No se pudieron inicializar las colecciones") from e

def init_minio_bucket():
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
            logger.info(f"Bucket {MINIO_BUCKET} creado en MinIO")
        else:
            logger.info(f"Bucket {MINIO_BUCKET} ya existe en MinIO")
    except S3Error as e:
        logger.error("Error inicializando bucket en MinIO: %s", str(e))
        raise RuntimeError("No se pudo inicializar el bucket de MinIO") from e

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """
    Lista los campos disponibles en los registros del MES (Manufacturing Execution System).
    
    Devuelve:
        - key_figures: Campos numéricos disponibles para análisis
        - key_values: Valores únicos para campos categóricos
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error"|"no_data",
                "key_figures": [str],  # Lista de campos numéricos
                "key_values": {str: [str]},  # Diccionario de campos categóricos con valores únicos
                "message": str  # Solo en caso de error o sin datos
            }
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/machines/")
            response.raise_for_status()
            records = response.json()

        if not records:
            return json.dumps({
                "status": "no_data",
                "message": "No hay registros disponibles en el MES",
                "key_figures": [],
                "key_values": {}
            })

        sample_record = records[0]
        key_figures = []
        key_values = {}
        
        for field, value in sample_record.items():
            if field == "id":
                continue
                
            if isinstance(value, (int, float)):
                key_figures.append(field)
            elif isinstance(value, str):
                unique_values = sorted({rec[field] for rec in records if field in rec})
                key_values[field] = unique_values

        return json.dumps({
            "status": "success",
            "key_figures": sorted(key_figures),
            "key_values": key_values
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Error en list_fields: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"Error al listar campos: {str(e)}",
            "key_figures": [],
            "key_values": {}
        }, ensure_ascii=False)

@mcp.tool()
async def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    time_filter: Optional[Dict[str, str]] = None
) -> str:
    """
    Recupera datos del MES aplicando filtros y almacena los resultados en Qdrant para búsquedas semánticas.
    
    Args:
        ctx: Contexto de FastMCP
        key_values: Filtros por valores específicos (e.g., {"machine": "CNC-01"})
        key_figures: Lista de métricas numéricas a incluir
        time_filter: Rango de fechas (e.g., {"start_date": "2025-01-01", "end_date": "2025-01-31"})
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error",
                "count": int,  # Número de registros devueltos
                "data": [dict],  # Registros del MES
                "message": str  # Solo en caso de error
            }
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        query_params = {}
        if time_filter:
            tf = TimeFilter(**time_filter)
            tf.validate_dates()
            
            if tf.specific_date:
                query_params["specific_date"] = tf.specific_date
            else:
                if tf.start_date:
                    query_params["start_date"] = tf.start_date
                if tf.end_date:
                    query_params["end_date"] = tf.end_date

        endpoint = f"{API_URL}/machines/"
        if "machine" in key_values:
            endpoint = f"{API_URL}/machines/{key_values['machine']}"

        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=query_params)
            response.raise_for_status()
            data = response.json()

        filtered_data = [
            record for record in data
            if all(record.get(k) == v for k, v in key_values.items() if k != "machine")
        ]

        processed_data = []
        for record in filtered_data:
            item = {
                "id": record["id"],
                "date": record["date"],
                "machine": record["machine"]
            }
            
            for field in key_figures:
                if field in record:
                    item[field] = record[field]
            
            processed_data.append(item)

        if processed_data:
            points = []
            for record in processed_data:
                record_text = json.dumps(record)
                point_id = hashlib.md5(record_text.encode()).hexdigest()
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=model.encode(record_text).tolist(),
                    payload=record
                ))
            
            qdrant_client.upsert(
                collection_name="mes_logs",
                points=points
            )

        return json.dumps({
            "status": "success",
            "count": len(processed_data),
            "data": processed_data
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Error en fetch_mes_data: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"Error al recuperar datos: {str(e)}",
            "count": 0,
            "data": []
        }, ensure_ascii=False)

@mcp.tool()
async def load_sop(ctx: Context, machine: str) -> str:
    """
    Carga y procesa el PDF de SOP (Standard Operating Procedure) para una máquina específica.
    
    Extrae reglas de cumplimiento del texto del PDF y las almacena en Qdrant para su posterior análisis.
    
    Args:
        ctx: Contexto de FastMCP
        machine: Nombre de la máquina (e.g., "CNC-01")
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"exists"|"error",
                "machine": str,
                "rules": {
                    "metric_name": {
                        "value": float,
                        "operator": ">="|"<=",
                        "unit": str,
                        "source_text": str
                    },
                    ...
                },
                "message": str  # Solo en caso de error
            }
    """
    try:
        existing = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]
            ),
            limit=1
        )
        
        if existing[0]:
            return json.dumps({
                "status": "exists",
                "machine": machine,
                "rules": existing[0][0].payload["rules"]
            }, ensure_ascii=False)

        pdf_name = f"{machine}.pdf"
        
        try:
            objects = minio_client.list_objects(MINIO_BUCKET, prefix=pdf_name)
            pdf_exists = any(obj.object_name == pdf_name for obj in objects)
            
            if not pdf_exists:
                available_pdfs = [obj.object_name for obj in minio_client.list_objects(MINIO_BUCKET)]
                return json.dumps({
                    "status": "error",
                    "message": f"PDF {pdf_name} no encontrado en MinIO. Disponibles: {', '.join(available_pdfs)}",
                    "machine": machine,
                    "rules": {}
                }, ensure_ascii=False)
            
            response = minio_client.get_object(MINIO_BUCKET, pdf_name)
            pdf_data = response.read()
            response.close()
            response.release_conn()
            
        except S3Error as e:
            logger.error("Error accediendo a MinIO para %s: %s", pdf_name, str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error al acceder al PDF en MinIO: {str(e)}",
                "machine": machine,
                "rules": {}
            }, ensure_ascii=False)

        try:
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            logger.error("Error extrayendo texto de %s: %s", pdf_name, str(e))
            return json.dumps({
                "status": "error",
                "message": f"Error al extraer texto del PDF: {str(e)}",
                "machine": machine,
                "rules": {}
            }, ensure_ascii=False)

        if not content.strip():
            return json.dumps({
                "status": "error",
                "message": f"El PDF {pdf_name} está vacío o no contiene texto extraíble",
                "machine": machine,
                "rules": {}
            }, ensure_ascii=False)

        rule_patterns = [
            (r"(?P<field>uptime|tiempo de actividad)\s*(?P<operator>>=|≥)\s*(?P<value>\d+\.\d+)\s*%", "uptime", ">=", "%"),
            (r"(?P<field>temperature|temperatura)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*°C", "temperature", "<=", "°C"),
            (r"(?P<field>vibration|vibración)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*mm/s", "vibration", "<=", "mm/s"),
            (r"(?P<field>defects|defectos)\s*(?P<operator><=|≤)\s*(?P<value>\d+)", "defects", "<=", "")
        ]
        
        rules = {}
        for pattern, field, default_op, unit in rule_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    rules[field] = {
                        "value": float(match.group("value")),
                        "operator": match.group("operator") if match.group("operator") else default_op,
                        "unit": unit,
                        "source_text": match.group(0)
                    }
                except (ValueError, KeyError) as e:
                    logger.warning("Error extrayendo regla %s: %s", field, str(e))
                    continue

        if not rules:
            return json.dumps({
                "status": "error",
                "message": f"No se encontraron reglas en {pdf_name}",
                "machine": machine,
                "rules": {}
            }, ensure_ascii=False)

        embedding = model.encode(content).tolist()
        qdrant_client.upsert(
            collection_name="sop_pdfs",
            points=[models.PointStruct(
                id=hashlib.md5(machine.encode()).hexdigest(),
                vector=embedding,
                payload={
                    "machine": machine,
                    "filename": pdf_name,
                    "rules": rules
                }
            )]
        )

        return json.dumps({
            "status": "success",
            "machine": machine,
            "rules": rules
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Error en load_sop para %s: %s", machine, str(e))
        return json.dumps({
            "status": "error",
            "message": f"Error al procesar SOP: {str(e)}",
            "machine": machine,
            "rules": {}
        }, ensure_ascii=False)

@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    time_filter: Optional[Dict[str, str]] = None
) -> str:
    """
    Analiza el cumplimiento de métricas de producción contra las reglas definidas en los SOP.
    
    Combina datos del MES con reglas de SOP para evaluar conformidad.
    
    Args:
        ctx: Contexto de FastMCP
        key_values: Filtros por valores específicos (e.g., {"machine": "CNC-01"})
        key_figures: Lista de métricas numéricas a analizar (e.g., ["uptime", "temperature"])
        time_filter: Rango de fechas (e.g., {"start_date": "2025-01-01", "end_date": "2025-01-31"})
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error",
                "period": str,  # Período analizado
                "machine_filter": str,  # Máquina(s) filtradas
                "metrics_analyzed": [str],
                "results": [
                    {
                        "id": str,
                        "date": str,
                        "machine": str,
                        "metrics": {str: float},
                        "compliance": {
                            "metric_name": {
                                "value": float,
                                "rule": str,
                                "status": "compliant"|"non_compliant"|"unknown"
                            },
                            ...
                        },
                        "compliance_percentage": float
                    },
                    ...
                ],
                "sop_coverage": str,  # Porcentaje de máquinas con SOP
                "analysis_notes": [str],
                "message": str  # Solo en caso de error
            }
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        time_filter = time_filter or {}
        
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "No se pudo obtener la estructura de campos",
                "results": []
            }, ensure_ascii=False)
        
        valid_key_figures = fields_info["key_figures"]
        valid_key_values = fields_info["key_values"]
        
        invalid_figures = [f for f in key_figures if f not in valid_key_figures]
        if invalid_figures:
            return json.dumps({
                "status": "error",
                "message": f"Key figures inválidos: {invalid_figures}. Válidos: {valid_key_figures}",
                "results": []
            }, ensure_ascii=False)
        
        invalid_values = {k: v for k, v in key_values.items() if k not in valid_key_values or v not in valid_key_values.get(k, [])}
        if invalid_values:
            return json.dumps({
                "status": "error",
                "message": f"Key values inválidos: {invalid_values}. Campos válidos: {list(valid_key_values.keys())}",
                "results": []
            }, ensure_ascii=False)
        
        fetch_result = json.loads(await fetch_mes_data(
            ctx,
            key_values=key_values,
            key_figures=key_figures,
            time_filter=time_filter
        ))
        
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"No se pudieron obtener datos: {fetch_result.get('message', '')}",
                "results": []
            }, ensure_ascii=False)
        
        unique_machines = {record["machine"] for record in fetch_result["data"]}
        
        sop_coverage = {"with_sop": 0, "without_sop": 0}
        machine_rules = {}
        
        for machine in unique_machines:
            sop_result = json.loads(await load_sop(ctx, machine))
            if sop_result["status"] in ["success", "exists"]:
                machine_rules[machine] = sop_result["rules"]
                sop_coverage["with_sop"] += 1
            else:
                machine_rules[machine] = {}
                sop_coverage["without_sop"] = 1
        
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "id": record["id"],
                "date": record["date"],
                "machine": record["machine"],
                "metrics": {},
                "compliance": {},
                "compliance_percentage": 0.0
            }
            
            total_metrics = 0
            compliant_metrics = 0
            
            for metric in key_figures:
                if metric not in record:
                    continue
                    
                metric_value = record[metric]
                analysis["metrics"][metric] = metric_value
                total_metrics += 1
                
                rules = machine_rules.get(record["machine"], {})
                if metric in rules:
                    rule = rules[metric]
                    operator = rule["operator"]
                    rule_value = rule["value"]
                    
                    is_compliant = (
                        (metric_value >= rule_value) if operator == ">=" else
                        (metric_value <= rule_value)
                    )
                    
                    analysis["compliance"][metric] = {
                        "value": metric_value,
                        "rule": f"{operator} {rule_value}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }
                    
                    if is_compliant:
                        compliant_metrics += 1
                else:
                    analysis["compliance"][metric] = {
                        "value": metric_value,
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }
            
            if total_metrics > 0:
                analysis["compliance_percentage"] = round(
                    (compliant_metrics / total_metrics) * 100, 2
                )
            
            results.append(analysis)
        
        tf = TimeFilter(**time_filter)
        tf.validate_dates()
        period = (
            tf.specific_date if tf.specific_date else
            f"{tf.start_date} a {tf.end_date}" if tf.start_date and tf.end_date else
            "sin filtro temporal"
        )
        
        return json.dumps({
            "status": "success",
            "period": period,
            "machine_filter": key_values.get("machine", "todas las máquinas"),
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sop_coverage['with_sop']}/{len(unique_machines)} máquinas con SOP",
            "analysis_notes": [
                "compliant: Cumple con la regla SOP",
                "non_compliant: No cumple con la regla SOP",
                "unknown: No hay regla definida para esta métrica"
            ]
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error("Error en analyze_compliance: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"Error en el análisis: {str(e)}",
            "results": []
        }, ensure_ascii=False)



@mcp.tool()
async def analyze_all_machines_compliance(
    ctx: Context,
    key_figures: Optional[List[str]] = None,
    time_filter: Optional[Dict[str, str]] = None
) -> str:
    """
    Analiza el cumplimiento de todas las máquinas contra las reglas SOP.

    Args:
        ctx: Contexto de FastMCP.
        key_figures: Lista de métricas numéricas a analizar (e.g., ["uptime", "temperature"]).
        time_filter: Filtro temporal (e.g., {"start_date": "2025-01-01", "end_date": "2025-01-31"}).

    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error",
                "period": str,
                "metrics_analyzed": [str],
                "results": [
                    {
                        "id": str,
                        "date": str,
                        "machine": str,
                        "metrics": {str: float},
                        "compliance": {str: {...}},
                        "compliance_percentage": float
                    },
                    ...
                ],
                "sop_coverage": str,
                "analysis_notes": [str],
                "message": str  # Solo en error
            }
    """
    try:
        key_figures = key_figures or []
        time_filter = time_filter or {}

        # Validar métricas
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "No se pudo obtener la estructura de campos",
                "results": []
            }, ensure_ascii=False)

        valid_key_figures = fields_info["key_figures"]
        invalid_figures = [f for f in key_figures if f not in valid_key_figures]
        if invalid_figures:
            return json.dumps({
                "status": "error",
                "message": f"Key figures inválidos: {invalid_figures}. Válidos: {valid_key_figures}",
                "results": []
            }, ensure_ascii=False)

        # Recuperar datos MES para todas las máquinas
        fetch_result = json.loads(await fetch_mes_data(
            ctx,
            key_values={},  # Sin filtro por máquina
            key_figures=key_figures,
            time_filter=time_filter
        ))

        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"No se pudieron obtener datos: {fetch_result.get('message', '')}",
                "results": []
            }, ensure_ascii=False)

        unique_machines = {record["machine"] for record in fetch_result["data"]}

        # Cargar reglas SOP para cada máquina
        sop_coverage = {"with_sop": 0, "without_sop": 0}
        machine_rules = {}
        for machine in unique_machines:
            sop_result = json.loads(await load_sop(ctx, machine))
            if sop_result["status"] in ["success", "exists"]:
                machine_rules[machine] = sop_result["rules"]
                sop_coverage["with_sop"] += 1
            else:
                machine_rules[machine] = {}
                sop_coverage["without_sop"] += 1

        # Analizar cumplimiento
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "id": record["id"],
                "date": record["date"],
                "machine": record["machine"],
                "metrics": {},
                "compliance": {},
                "compliance_percentage": 0.0
            }

            total_metrics = 0
            compliant_metrics = 0

            for metric in key_figures:
                if metric not in record:
                    continue

                metric_value = record[metric]
                analysis["metrics"][metric] = metric_value
                total_metrics += 1

                rules = machine_rules.get(record["machine"], {})
                if metric in rules:
                    rule = rules[metric]
                    operator = rule["operator"]
                    rule_value = rule["value"]

                    is_compliant = (
                        (metric_value >= rule_value) if operator == ">=" else
                        (metric_value <= rule_value)
                    )

                    analysis["compliance"][metric] = {
                        "value": metric_value,
                        "rule": f"{operator} {rule_value}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }

                    if is_compliant:
                        compliant_metrics += 1
                else:
                    analysis["compliance"][metric] = {
                        "value": metric_value,
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }

            if total_metrics > 0:
                analysis["compliance_percentage"] = round(
                    (compliant_metrics / total_metrics) * 100, 2
                )

            results.append(analysis)

        # Formatear período
        tf = TimeFilter(**time_filter)
        tf.validate_dates()
        period = (
            tf.specific_date if tf.specific_date else
            f"{tf.start_date} a {tf.end_date}" if tf.start_date and tf.end_date else
            "sin filtro temporal"
        )

        return json.dumps({
            "status": "success",
            "period": period,
            "machine_filter": "todas las máquinas",
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sop_coverage['with_sop']}/{len(unique_machines)} máquinas con SOP",
            "analysis_notes": [
                "compliant: Cumple con la regla SOP",
                "non_compliant: No cumple con la regla SOP",
                "unknown: No hay regla definida para esta métrica"
            ]
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Error en analyze_all_machines_compliance: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"Error en el análisis: {str(e)}",
            "results": []
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_collections()
    init_minio_bucket()
    mcp.run()