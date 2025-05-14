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

# Inicialización del servicio MCP
mcp = FastMCP("Manufacturing Compliance Processor")

# Configuración global
API_URL = "http://api:5000"  # URL de la API de manufactura
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo de embeddings
qdrant_client = QdrantClient(host="qdrant", port=6333)  # Cliente de Qdrant

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFilter(BaseModel):
    """
    Modelo para validación de filtros temporales en consultas MES.
    
    Atributos:
        start_date (str, opcional): Fecha de inicio en formato YYYY-MM-DD.
        end_date (str, opcional): Fecha de fin en formato YYYY-MM-DD.
        specific_date (str, opcional): Fecha específica en formato YYYY-MM-DD.
    
    Notas:
        - specific_date tiene prioridad sobre start_date/end_date
        - Valida el formato de fecha y coherencia temporal
    """
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        """
        Valida el formato y coherencia de las fechas proporcionadas.
        
        Raises:
            ValueError: Si las fechas tienen formato incorrecto o son incoherentes.
        """
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
    """
    Inicializa las colecciones en Qdrant para almacenamiento de datos.
    
    Crea dos colecciones:
    1. 'mes_logs': Para registros de manufactura con embeddings
    2. 'sop_pdfs': Para documentos SOP con sus reglas extraídas
    
    Configura:
    - Vectores de 384 dimensiones usando el modelo SentenceTransformer
    - Distancia coseno para similitud
    - Optimizaciones para manejo de grandes volúmenes
    
    Raises:
        RuntimeError: Si falla la creación de las colecciones
    """
    try:
        # Configuración común para ambas colecciones
        vector_config = models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
        optimizer_config = models.OptimizersConfigDiff(
            indexing_threshold=20000,
            memmap_threshold=20000
        )
        
        # Colección para registros MES
        qdrant_client.recreate_collection(
            collection_name="mes_logs",
            vectors_config=vector_config,
            optimizers_config=optimizer_config
        )
        
        # Colección para documentos SOP
        qdrant_client.recreate_collection(
            collection_name="sop_pdfs",
            vectors_config=vector_config,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,  # Indexar todo inmediatamente
                memmap_threshold=20000
            )
        )
        logger.info("Colecciones Qdrant inicializadas: mes_logs, sop_pdfs")
    except Exception as e:
        logger.error("Error inicializando Qdrant: %s", str(e))
        raise RuntimeError("No se pudieron inicializar las colecciones") from e

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """
    Obtiene la estructura de campos disponibles en los datos MES.
    
    Procesa:
    1. Consulta todos los registros MES disponibles
    2. Clasifica campos en numéricos (key_figures) y categóricos (key_values)
    3. Identifica valores únicos para campos categóricos
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error"|"no_data",
                "key_figures": ["uptime", "temperature", ...],
                "key_values": {
                    "machine": ["ModelA", "ModelB", ...],
                    "production_line": ["Line1", "Line2", ...],
                    ...
                },
                "message": str  # Solo en casos de error
            }
    
    Example:
        >>> list_fields()
        {
            "status": "success",
            "key_figures": ["uptime", "temperature", "defects"],
            "key_values": {
                "machine": ["ModelA", "ModelB"],
                "production_line": ["Line1", "Line2"]
            }
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

        # Clasificación de campos
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
    Recupera datos del MES con filtros avanzados.
    
    Args:
        key_values: Filtros categóricos ej. {"machine": "ModelA", "production_line": "Line1"}
        key_figures: Campos numéricos a incluir ej. ["uptime", "temperature"]
        time_filter: {"start_date": "2025-01-01", "end_date": "2025-01-31"} o {"specific_date": "2025-01-15"}
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error",
                "count": int,
                "data": [
                    {
                        "id": str,
                        "date": str,
                        "machine": str,
                        ...key_figures
                    },
                    ...
                ],
                "message": str  # Solo en error
            }
    
    Example:
        >>> fetch_mes_data(
                key_values={"machine": "ModelA"},
                key_figures=["uptime", "temperature"],
                time_filter={"start_date": "2025-01-01", "end_date": "2025-01-31"}
            )
        {
            "status": "success",
            "count": 10,
            "data": [
                {"id": "1", "date": "2025-01-01", "machine": "ModelA", "uptime": 95.0, "temperature": 72.0},
                ...
            ]
        }
    """
    try:
        # Validación de parámetros
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Procesamiento del filtro temporal
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

        # Construcción del endpoint
        endpoint = f"{API_URL}/machines/"
        if "machine" in key_values:
            endpoint = f"{API_URL}/machines/{key_values['machine']}"

        # Consulta a la API
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=query_params)
            response.raise_for_status()
            data = response.json()

        # Filtrado adicional por key_values
        filtered_data = [
            record for record in data
            if all(record.get(k) == v for k, v in key_values.items() if k != "machine")
        ]

        # Selección de campos
        processed_data = []
        for record in filtered_data:
            item = {
                "id": record["id"],
                "date": record["date"],
                "machine": record["machine"]
            }
            
            # Agregar key_figures solicitados
            for field in key_figures:
                if field in record:
                    item[field] = record[field]
            
            processed_data.append(item)

        # Almacenamiento en Qdrant
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
    Carga y procesa un documento SOP para una máquina específica.
    
    Args:
        machine: Nombre de la máquina (ej. "ModelA")
    
    Process:
        1. Verifica si ya existe en Qdrant
        2. Descarga el PDF correspondiente (machine.pdf)
        3. Extrae reglas usando expresiones regulares
        4. Almacena en Qdrant con embeddings
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"exists"|"error",
                "machine": str,
                "rules": {
                    "uptime": {"value": 95.0, "operator": ">=", "unit": "%"},
                    ...
                },
                "message": str  # Solo en error
            }
    
    Example:
        >>> load_sop("ModelA")
        {
            "status": "success",
            "machine": "ModelA",
            "rules": {
                "uptime": {"value": 95.0, "operator": ">=", "unit": "%", "source_text": "uptime >= 95%"},
                "temperature": {"value": 75.0, "operator": "<=", "unit": "°C", "source_text": "temperature <= 75°C"}
            }
        }
    """
    try:
        # Verificar existencia previa
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

        # Descargar contenido del PDF
        pdf_name = f"{machine}.pdf"
        async with httpx.AsyncClient() as client:
            # Verificar disponibilidad
            list_response = await client.get(f"{API_URL}/pdfs/list")
            available_pdfs = [pdf["filename"] for pdf in list_response.json()]
            
            if pdf_name not in available_pdfs:
                return json.dumps({
                    "status": "error",
                    "message": f"PDF {pdf_name} no encontrado. Disponibles: {', '.join(available_pdfs)}"
                }, ensure_ascii=False)
            
            # Obtener contenido
            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": [pdf_name], "max_length": 10000}
            )
            content = content_response.json()["pdfs"][0]["content"]

        # Extracción de reglas con patrones robustos
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

        # Almacenamiento en Qdrant
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
    Analiza el cumplimiento de métricas MES contra reglas SOP.
    
    Args:
        key_values: Filtros categóricos ej. {"machine": "ModelA"}
        key_figures: Métricas a analizar ej. ["uptime", "temperature"]
        time_filter: Rango de fechas ej. {"start_date": "2025-01-01", "end_date": "2025-01-31"}
    
    Process:
        1. Valida parámetros de entrada
        2. Obtiene datos MES con los filtros
        3. Carga reglas SOP para las máquinas involucradas
        4. Evalúa cumplimiento métrica por métrica
    
    Returns:
        str: JSON con estructura:
            {
                "status": "success"|"error",
                "period": str,
                "machine_filter": str,
                "metrics_analyzed": [str],
                "results": [
                    {
                        "id": str,
                        "date": str,
                        "machine": str,
                        "metrics": {"uptime": 95.0, ...},
                        "compliance": {
                            "uptime": {
                                "value": 95.0,
                                "rule": ">= 95%",
                                "status": "compliant"
                            },
                            ...
                        },
                        "compliance_percentage": float
                    },
                    ...
                ],
                "sop_coverage": str,
                "message": str  # Solo en error
            }
    
    Example:
        >>> analyze_compliance(
                key_values={"machine": "ModelA"},
                key_figures=["uptime"],
                time_filter={"specific_date": "2025-01-01"}
            )
        {
            "status": "success",
            "period": "2025-01-01",
            "machine_filter": "ModelA",
            "metrics_analyzed": ["uptime"],
            "results": [
                {
                    "id": "123",
                    "date": "2025-01-01",
                    "machine": "ModelA",
                    "metrics": {"uptime": 96.0},
                    "compliance": {
                        "uptime": {
                            "value": 96.0,
                            "rule": ">= 95%",
                            "status": "compliant"
                        }
                    },
                    "compliance_percentage": 100.0
                }
            ],
            "sop_coverage": "1/1 máquinas con SOP"
        }
    """
    try:
        # Validación inicial
        key_values = key_values or {}
        key_figures = key_figures or []
        time_filter = time_filter or {}
        
        # Obtener estructura de campos válidos
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "No se pudo obtener la estructura de campos",
                "results": []
            }, ensure_ascii=False)
        
        valid_key_figures = fields_info["key_figures"]
        valid_key_values = fields_info["key_values"]
        
        # Validar key_figures
        invalid_figures = [f for f in key_figures if f not in valid_key_figures]
        if invalid_figures:
            return json.dumps({
                "status": "error",
                "message": f"Key figures inválidos: {invalid_figures}. Válidos: {valid_key_figures}",
                "results": []
            }, ensure_ascii=False)
        
        # Validar key_values
        invalid_values = {k: v for k, v in key_values.items() if k not in valid_key_values or v not in valid_key_values.get(k, [])}
        if invalid_values:
            return json.dumps({
                "status": "error",
                "message": f"Key values inválidos: {invalid_values}. Campos válidos: {list(valid_key_values.keys())}",
                "results": []
            }, ensure_ascii=False)
        
        # Obtener datos MES
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
        
        # Procesar máquinas únicas en los resultados
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
        
        # Analizar cumplimiento para cada registro
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
            
            # Contadores para el porcentaje de cumplimiento
            total_metrics = 0
            compliant_metrics = 0
            
            # Evaluar cada métrica solicitada
            for metric in key_figures:
                if metric not in record:
                    continue
                    
                metric_value = record[metric]
                analysis["metrics"][metric] = metric_value
                total_metrics += 1
                
                # Verificar si existe regla para esta métrica
                rules = machine_rules.get(record["machine"], {})
                if metric in rules:
                    rule = rules[metric]
                    operator = rule["operator"]
                    rule_value = rule["value"]
                    
                    # Evaluar cumplimiento
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
            
            # Calcular porcentaje de cumplimiento
            if total_metrics > 0:
                analysis["compliance_percentage"] = round(
                    (compliant_metrics / total_metrics) * 100, 2
                )
            
            results.append(analysis)
        
        # Determinar descripción del período
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

# Punto de entrada principal
if __name__ == "__main__":
    init_collections()
    mcp.run()