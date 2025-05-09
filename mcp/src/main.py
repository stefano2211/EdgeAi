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

# Inicializar FastMCP
mcp = FastMCP("MES Compliance Processor")

# Configuración
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Inicializar Qdrant con configuración optimizada
qdrant_client = QdrantClient(
    host="qdrant",
    port=6333,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        try:
            if self.specific_date:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
            else:
                if self.start_date:
                    datetime.strptime(self.start_date, "%Y-%m-%d")
                if self.end_date:
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                if self.start_date and self.end_date and self.start_date > self.end_date:
                    raise ValueError("Start date cannot be after end date")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")

def init_collections():
    """Inicializa las colecciones en Qdrant con configuraciones optimizadas"""
    try:
        # Configuración para registros MES
        qdrant_client.recreate_collection(
            collection_name="mes_logs",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=20000
            )
        )

        # Configuración para PDFs SOP
        qdrant_client.recreate_collection(
            collection_name="sop_pdfs",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
                memmap_threshold=20000
            )
        )
        logger.info("Colecciones de Qdrant inicializadas")
    except Exception as e:
        logger.error(f"Error inicializando Qdrant: {str(e)}")
        raise

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """Lista los campos disponibles y valores únicos"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/machines/")
            response.raise_for_status()
            records = response.json()

            if not records:
                return json.dumps({
                    "status": "no_data",
                    "message": "No se encontraron registros",
                    "fields": []
                })

            # Clasificar campos
            sample = records[0]
            fields = {
                "numeric": [],
                "categorical": []
            }

            for field, value in sample.items():
                if field == "id":
                    continue
                if isinstance(value, (int, float)):
                    fields["numeric"].append(field)
                elif isinstance(value, str):
                    fields["categorical"].append(field)

            # Obtener valores únicos
            unique_values = {}
            for field in fields["categorical"]:
                values = sorted({record[field] for record in records if field in record})
                unique_values[field] = values

            return json.dumps({
                "status": "success",
                "fields": fields,
                "unique_values": unique_values
            }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en list_fields: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)

@mcp.tool()
async def fetch_mes_data(
    ctx: Context,
    machine: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> str:
    """Obtiene datos del MES incluyendo ID de registro"""
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

        params = {}
        if time_filter.specific_date:
            params["specific_date"] = time_filter.specific_date
        else:
            if time_filter.start_date:
                params["start_date"] = time_filter.start_date
            if time_filter.end_date:
                params["end_date"] = time_filter.end_date

        endpoint = f"{API_URL}/machines/{machine}" if machine else f"{API_URL}/machines/"

        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Asegurar que el ID siempre esté incluido
            processed_data = []
            for record in data:
                processed_record = {
                    "id": record["id"],  # Incluir ID siempre
                    "date": record["date"],
                    "machine": record["machine"]
                }
                
                # Agregar campos solicitados o todos los numéricos si no se especifican
                if fields:
                    for field in fields:
                        if field in record:
                            processed_record[field] = record[field]
                else:
                    # Si no se especifican campos, incluir todos los numéricos
                    fields_info = json.loads(await list_fields(ctx))
                    if fields_info["status"] == "success":
                        numeric_fields = fields_info["fields"]["numeric"]
                        for field in numeric_fields:
                            if field in record:
                                processed_record[field] = record[field]

                processed_data.append(processed_record)

            # Indexar en Qdrant
            points = []
            for record in processed_data:
                record_text = json.dumps(record)
                point_id = hashlib.md5(record_text.encode()).hexdigest()
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=model.encode(record_text).tolist(),
                        payload=record
                    )
                )

            if points:
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
        logger.error(f"Error en fetch_mes_data: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)


@mcp.tool()
async def load_sop(ctx: Context, machine: str) -> str:
    """Carga y procesa un PDF SOP para una máquina específica con mejor extracción de reglas"""
    try:
        # Verificar si ya existe
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

        # Obtener contenido del PDF
        pdf_name = f"{machine}.pdf"
        async with httpx.AsyncClient() as client:
            # Verificar existencia
            list_response = await client.get(f"{API_URL}/pdfs/list")
            if not any(pdf["filename"] == pdf_name for pdf in list_response.json()):
                return json.dumps({
                    "status": "error",
                    "message": f"PDF {pdf_name} no encontrado"
                }, ensure_ascii=False)

            # Obtener contenido
            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": [pdf_name], "max_length": 10000}
            )
            content = content_response.json()["pdfs"][0]["content"]

        # Patrones mejorados para extracción exacta de reglas
        rules = {}
        patterns = [
            (r"(uptime|tiempo de actividad)[^\d]*([>=≥]+)[^\d]*(\d+)\s*%", "uptime", ">="),
            (r"(defects|defectos|number of defects)[^\d]*([<=≤]+)[^\d]*(\d+)", "defects", "<="),
            (r"(temperature|temperatura)[^\d]*([<=≤]+)[^\d]*(\d+)\s*°?C?", "temperature", "<="),
            (r"(throughput|rendimiento|production throughput)[^\d]*([>=≥]+)[^\d]*(\d+)", "throughput", ">=")
        ]

        for pattern, field, operator in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(3))
                    rules[field] = {
                        "value": value,
                        "operator": operator,
                        "rule_text": f"{field} {operator} {value}" + 
                                    ("%" if field == "uptime" else "" + 
                                     ("°C" if field == "temperature" else ""))
                    }
                except ValueError:
                    continue

        if not rules:
            return json.dumps({
                "status": "error",
                "message": "No se encontraron reglas en el PDF"
            }, ensure_ascii=False)

        # Almacenar en Qdrant
        embedding = model.encode(content).tolist()
        qdrant_client.upsert(
            collection_name="sop_pdfs",
            points=[models.PointStruct(
                id=hashlib.md5(machine.encode()).hexdigest(),
                vector=embedding,
                payload={
                    "filename": pdf_name,
                    "machine": machine,
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
        logger.error(f"Error en load_sop: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)

@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    machine: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    focus_numeric_fields: Optional[List[str]] = None,
    categorical_filters: Optional[Dict[str, str]] = None
) -> str:
    """Analiza cumplimiento para cualquier combinación de campos con manejo automático de SOPs"""
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

        # Obtener todos los campos y máquinas disponibles
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "No se pudieron obtener los campos disponibles"
            }, ensure_ascii=False)

        # Determinar máquinas a analizar
        all_machines = fields_info["unique_values"].get("machine", [])
        target_machines = [machine] if machine else all_machines

        # Cargar reglas SOP para todas las máquinas relevantes
        machines_rules = {}
        for machine_name in target_machines:
            sop_result = json.loads(await load_sop(ctx, machine_name))
            if sop_result["status"] in ["success", "exists"]:
                machines_rules[machine_name] = sop_result.get("rules", {})
            else:
                machines_rules[machine_name] = {"status": "no_sop"}

        # Determinar campos numéricos a analizar
        if not focus_numeric_fields:
            focus_numeric_fields = fields_info["fields"]["numeric"]

        # Obtener datos para todas las máquinas objetivo
        all_results = []
        for machine_name in target_machines:
            fetch_result = json.loads(await fetch_mes_data(
                ctx,
                machine=machine_name,
                start_date=start_date,
                end_date=end_date,
                specific_date=specific_date,
                fields=list(set(focus_numeric_fields + ["id", "date", "machine"] + (list(categorical_filters.keys()) if categorical_filters else [])))
            ))

            if fetch_result["status"] != "success":
                continue

            # Aplicar filtros categóricos
            filtered_data = []
            for record in fetch_result["data"]:
                match = True
                if categorical_filters:
                    for field, value in categorical_filters.items():
                        if record.get(field) != value:
                            match = False
                            break
                if match:
                    filtered_data.append(record)

            # Preparar resultados para esta máquina
            for record in filtered_data:
                entry = {
                    "id": record["id"],
                    "date": record["date"],
                    "machine": record["machine"],
                    "metrics": {},
                    "compliance_rules": {},
                    "categorical_data": {},
                    "sop_status": "available" if machines_rules[machine_name].get("status") != "no_sop" else "not_available"
                }

                # Campos numéricos y reglas
                for field in focus_numeric_fields:
                    if field in record:
                        entry["metrics"][field] = record[field]
                        if machines_rules[machine_name].get(field):
                            entry["compliance_rules"][field] = machines_rules[machine_name][field]["rule_text"]
                        else:
                            entry["compliance_rules"][field] = "no_rule_defined"

                # Campos categóricos
                if categorical_filters:
                    for field in categorical_filters.keys():
                        if field in record:
                            entry["categorical_data"][field] = record[field]

                all_results.append(entry)

        return json.dumps({
            "status": "success",
            "period": specific_date or f"{start_date} a {end_date}",
            "machine_filter": machine or "all_machines",
            "numeric_fields_analyzed": focus_numeric_fields,
            "categorical_filters": categorical_filters or {},
            "sop_summary": {
                "machines_with_sop": sum(1 for rules in machines_rules.values() if rules.get("status") != "no_sop"),
                "machines_without_sop": sum(1 for rules in machines_rules.values() if rules.get("status") == "no_sop")
            },
            "analysis_note": "Verifique 'sop_status' en cada registro. Campos sin reglas mostrarán 'no_rule_defined'",
            "results": all_results
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en analyze_compliance: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)

if __name__ == "__main__":
    # Inicializar Qdrant al iniciar
    init_collections()
    mcp.run()