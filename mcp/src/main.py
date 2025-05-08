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

# Inicializar FastMCP
mcp = FastMCP("Dynamic MES Compliance Analysis")

# Configuración de URLs
API_URL = "http://api:5000"

# Modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cliente de Qdrant
qdrant_client = QdrantClient(host="qdrant", port=6333)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo para validar filtros de tiempo
class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        try:
            if self.specific_date:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
                self.start_date = None
                self.end_date = None
            else:
                if self.start_date:
                    datetime.strptime(self.start_date, "%Y-%m-%d")
                if self.end_date:
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                if self.start_date and self.end_date and self.start_date > self.end_date:
                    raise ValueError("La fecha de inicio no puede ser mayor que la fecha de fin")
        except ValueError as e:
            raise ValueError(f"Formato de fecha inválido. Use YYYY-MM-DD: {str(e)}")

# Función para inferir key figures y key values
async def infer_fields():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/machines/")
            response.raise_for_status()
            records = response.json()
            
            if not records:
                return {"key_figures": [], "key_values": [], "available_values": {}}
            
            # Obtener todas las claves del primer registro
            sample_record = records[0]
            all_fields = set(sample_record.keys()) - {"id"}  # Excluir 'id'
            
            # Clasificar campos
            key_figures = []
            key_values = []
            
            for field in all_fields:
                # Verificar el tipo de valor en todos los registros
                is_numeric = all(
                    isinstance(record[field], (int, float))
                    for record in records
                    if field in record
                )
                is_string = all(
                    isinstance(record[field], str)
                    for record in records
                    if field in record
                )
                
                if is_numeric:
                    key_figures.append(field)
                elif is_string:
                    key_values.append(field)
            
            # Obtener valores únicos para key values
            available_values = {}
            for key in key_values:
                values = sorted(set(record[key] for record in records if key in record))
                available_values[key] = values
            
            return {
                "key_figures": sorted(key_figures),
                "key_values": sorted(key_values),
                "available_values": available_values
            }
    except Exception as e:
        logger.error(f"Error al inferir campos: {str(e)}")
        return {"key_figures": [], "key_values": [], "available_values": {}}

# Inicializar colecciones en Qdrant
def init_qdrant():
    try:
        qdrant_client.recreate_collection(
            collection_name="mes_logs",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        qdrant_client.recreate_collection(
            collection_name="sop_pdfs",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        logger.info("Colecciones de Qdrant inicializadas: mes_logs, sop_pdfs")
    except Exception as e:
        logger.error(f"Error al inicializar Qdrant: {str(e)}")

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """
    Lista los key figures y key values inferidos de los registros MES, con valores únicos.
    """
    try:
        fields = await infer_fields()
        key_figures = fields["key_figures"]
        key_values = fields["key_values"]
        available_values = fields["available_values"]
        
        if not key_figures and not key_values:
            return (
                "Lista de Campos\n"
                "==============\n"
                "Estado: Sin datos\n"
                "Mensaje: No se encontraron registros MES para inferir campos."
            )
        
        report = [
            "Lista de Campos Disponibles",
            "==========================",
            "",
            "Key Figures (Métricas Cuantitativas)",
            "-----------------------------------",
            ", ".join(key_figures) if key_figures else "Ninguno",
            "",
            "Key Values (Datos Cualitativos)",
            "------------------------------"
        ]
        
        for key, values in available_values.items():
            report.append(f"{key}: {', '.join(map(str, values))}")
        
        return "\n".join(report)
    except Exception as e:
        logger.error(f"Error en list_fields: {str(e)}")
        return (
            "Lista de Campos\n"
            "==============\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo obtener los campos.\n"
            f"Detalles: {str(e)}"
        )

@mcp.tool()
async def fetch_mes_data(
    ctx: Context,
    key_values: Dict[str, str] = None,
    key_figures: List[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Recupera datos MES filtrados por key values y seleccionando key figures específicos.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

        # Obtener estructura de campos
        fields = await infer_fields()
        valid_key_figures = fields["key_figures"]
        valid_key_values = fields["key_values"]

        # Validar key figures
        if key_figures:
            invalid_figures = [f for f in key_figures if f not in valid_key_figures]
            if invalid_figures:
                raise ValueError(f"Key figures inválidos: {', '.join(invalid_figures)}. Opciones válidas: {', '.join(valid_key_figures)}")

        # Validar key values
        if key_values:
            invalid_keys = [k for k in key_values.keys() if k not in valid_key_values]
            if invalid_keys:
                raise ValueError(f"Key values inválidos: {', '.join(invalid_keys)}. Opciones válidas: {', '.join(valid_key_values)}")

        params = {}
        if time_filter.specific_date:
            params["specific_date"] = time_filter.specific_date
        else:
            if time_filter.start_date:
                params["start_date"] = time_filter.start_date
            if time_filter.end_date:
                params["end_date"] = time_filter.end_date

        async with httpx.AsyncClient() as client:
            endpoint = f"{API_URL}/machines/"
            if key_values and "machine" in key_values:
                endpoint = f"{API_URL}/machines/{key_values['machine']}"
            
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()

            # Filtrar por key values
            if key_values:
                for key, value in key_values.items():
                    if key != "machine":  # machine ya se filtró en el endpoint
                        logs = [log for log in logs if log.get(key) == value]

            # Seleccionar solo key figures solicitados
            if key_figures:
                filtered_logs = []
                for log in logs:
                    filtered_log = {"id": log["id"], "date": log["date"], "machine": log["machine"]}
                    for figure in key_figures:
                        if figure in log:
                            filtered_log[figure] = log[figure]
                    filtered_logs.append(filtered_log)
                logs = filtered_logs

            # Almacenar en Qdrant
            for log in logs:
                log_id = f"mes_log_{log['id']}"
                log_text = json.dumps(log)
                embedding = model.encode(log_text).tolist()
                qdrant_client.upsert(
                    collection_name="mes_logs",
                    points=[
                        models.PointStruct(
                            id=log_id,
                            vector=embedding,
                            payload=log
                        )
                    ]
                )

            if not logs:
                period = specific_date or f"{start_date} a {end_date}"
                filter_text = "\n".join([f"{k}: {v}" for k, v in (key_values or {}).items()]) or "Ninguno"
                return (
                    "Datos MES\n"
                    "========\n"
                    f"Período: {period}\n"
                    f"Filtros Aplicados:\n{filter_text}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros con los filtros especificados."
                )

            # Generar tabla en markdown
            headers = ["Date", "Machine"] + (key_figures or [])
            table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for log in logs:
                row = [log["date"], log["machine"]]
                for figure in key_figures or []:
                    value = log.get(figure, "N/A")
                    row.append(str(value))
                table.append("| " + " | ".join(row) + " |")

            report = [
                "Datos MES",
                "========",
                f"Período: {specific_date or f'{start_date} a {end_date}'}",
                f"Registros Encontrados: {len(logs)}",
                "",
                *table
            ]
            return "\n".join(report)
    except Exception as e:
        logger.error(f"Error en fetch_mes_data: {str(e)}")
        return (
            "Datos MES\n"
            "========\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo recuperar los datos.\n"
            f"Detalles: {str(e)}"
        )

@mcp.tool()
async def load_sop(ctx: Context, key_value: str, key_type: str = "machine") -> str:
    """
    Carga un PDF SOP asociado con un key value y lo almacena en Qdrant.
    """
    try:
        # Obtener key values válidos
        fields = await infer_fields()
        valid_key_values = fields["key_values"]
        
        # Validar key_type
        if key_type not in valid_key_values:
            raise ValueError(f"Key type inválido: {key_type}. Opciones válidas: {', '.join(valid_key_values)}")

        # Construir nombre del PDF (e.g., ModelA.pdf)
        pdf_name = f"{key_value}.pdf"
        
        # Verificar si el PDF ya está en Qdrant
        search_result = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=pdf_name)
                    )
                ]
            )
        )
        
        if search_result[0]:
            return (
                "Carga de SOP\n"
                "===========\n"
                f"Estado: Ya existe\n"
                f"Mensaje: El PDF '{pdf_name}' ya está almacenado en Qdrant."
            )

        # Obtener lista de PDFs disponibles
        async with httpx.AsyncClient() as client:
            pdf_list_response = await client.get(f"{API_URL}/pdfs/list")
            pdf_list = pdf_list_response.json()

            if not any(pdf["filename"] == pdf_name for pdf in pdf_list):
                return (
                    "Carga de SOP\n"
                    "===========\n"
                    "Estado: Error\n"
                    f"Mensaje: No se encontró el PDF '{pdf_name}'.\n"
                    f"PDFs disponibles: {', '.join([pdf['filename'] for pdf in pdf_list])}"
                )

            # Obtener contenido del PDF
            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": [pdf_name]}
            )
            content_response.raise_for_status()
            pdf_content = content_response.json()["pdfs"][0]["content"]

        # Obtener key figures válidos
        valid_key_figures = fields["key_figures"]

        # Extraer reglas SOP
        rules = {}
        patterns = [
            (r"uptime\s*[>=≥]+\s*(\d+\.?\d*)\s*%", "uptime", ">="),
            (r"defects\s*[<=≤]+\s*(\d+)", "defects", "<="),
            (r"vibration\s*[<=≤]+\s*(\d+\.?\d*)\s*(?:mm/s|mm\s*/\s*s)", "vibration", "<="),
            (r"temperature\s*[<=≤]+\s*(\d+\.?\d*)\s*(?:°C|C)", "temperature", "<="),
            (r"throughput\s*[>=≥]+\s*(\d+\.?\d*)\s*%", "throughput", ">="),
            (r"inventory\s*level\s*[>=≥]+\s*(\d+)", "inventory_level", ">=")
        ]
        
        for pattern, key, operator in patterns:
            if key not in valid_key_figures:
                continue  # Ignorar reglas para key figures no válidos
            match = re.search(pattern, pdf_content, re.IGNORECASE)
            if match:
                rules[key] = {"value": float(match.group(1)), "operator": operator}

        if not rules:
            return (
                "Carga de SOP\n"
                "===========\n"
                "Estado: Error\n"
                f"Mensaje: No se encontraron reglas SOP válidas en '{pdf_name}'.\n"
                f"Key figures válidos: {', '.join(valid_key_figures)}\n"
                "Recomendación: Asegure que el PDF contenga reglas como 'uptime >= 95%'."
            )

        # Almacenar en Qdrant
        sop_id = f"sop_{key_type}_{key_value}"
        sop_text = json.dumps(rules)
        embedding = model.encode(pdf_content).tolist()
        qdrant_client.upsert(
            collection_name="sop_pdfs",
            points=[
                models.PointStruct(
                    id=sop_id,
                    vector=embedding,
                    payload={
                        "filename": pdf_name,
                        key_type: key_value,
                        "rules": rules
                    }
                )
            ]
        )

        return (
            "Carga de SOP\n"
            "===========\n"
            "Estado: Éxito\n"
            f"Mensaje: El PDF '{pdf_name}' fue cargado y almacenado en Qdrant.\n"
            f"Reglas extraídas: {sop_text}"
        )
    except Exception as e:
        logger.error(f"Error en load_sop: {str(e)}")
        return (
            "Carga de SOP\n"
            "===========\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo cargar el PDF '{pdf_name}'.\n"
            f"Detalles: {str(e)}"
        )

@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    key_values: Dict[str, str] = None,
    key_figures: List[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Analiza el cumplimiento de los registros MES contra las reglas SOP.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

        # Obtener estructura de campos
        fields = await infer_fields()
        valid_key_figures = fields["key_figures"]
        valid_key_values = fields["key_values"]

        # Validar key figures
        if key_figures:
            invalid_figures = [f for f in key_figures if f not in valid_key_figures]
            if invalid_figures:
                raise ValueError(f"Key figures inválidos: {', '.join(invalid_figures)}. Opciones válidas: {', '.join(valid_key_figures)}")

        # Validar key values
        if key_values:
            invalid_keys = [k for k in key_values.keys() if k not in valid_key_values]
            if invalid_keys:
                raise ValueError(f"Key values inválidos: {', '.join(invalid_keys)}. Opciones válidas: {', '.join(valid_key_values)}")

        # Obtener registros MES
        params = {}
        if time_filter.specific_date:
            params["specific_date"] = time_filter.specific_date
        else:
            if time_filter.start_date:
                params["start_date"] = time_filter.start_date
            if time_filter.end_date:
                params["end_date"] = time_filter.end_date

        async with httpx.AsyncClient() as client:
            endpoint = f"{API_URL}/machines/"
            if key_values and "machine" in key_values:
                endpoint = f"{API_URL}/machines/{key_values['machine']}"
            
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()

            # Filtrar por key values
            if key_values:
                for key, value in key_values.items():
                    if key != "machine":
                        logs = [log for log in logs if log.get(key) == value]

            if not logs:
                period = specific_date or f"{start_date} a {end_date}"
                filter_text = "\n".join([f"{k}: {v}" for k, v in (key_values or {}).items()]) or "Ninguno"
                return (
                    "Análisis de Cumplimiento\n"
                    "=======================\n"
                    f"Período: {period}\n"
                    f"Filtros Aplicados:\n{filter_text}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros con los filtros especificados."
                )

        # Obtener reglas SOP desde Qdrant
        rules = {}
        if key_values and "machine" in key_values:
            machine = key_values["machine"]
            search_result = qdrant_client.scroll(
                collection_name="sop_pdfs",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="machine",
                            match=models.MatchValue(value=machine)
                        )
                    ]
                )
            )
            
            if search_result[0]:
                rules = search_result[0][0].payload["rules"]
            else:
                # Intentar cargar el PDF
                await load_sop(ctx, key_value=machine, key_type="machine")
                search_result = qdrant_client.scroll(
                    collection_name="sop_pdfs",
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="machine",
                                match=models.MatchValue(value=machine)
                            )
                        ]
                    )
                )
                if search_result[0]:
                    rules = search_result[0][0].payload["rules"]

        if not rules:
            return (
                "Análisis de Cumplimiento\n"
                "=======================\n"
                "Estado: Error\n"
                "Mensaje: No se encontraron reglas SOP para la máquina especificada.\n"
                "Recomendación: Cargue el PDF SOP correspondiente."
            )

        # Analizar cumplimiento
        compliance_report = []
        for log in logs:
            entry = {
                "date": log["date"],
                "machine": log["machine"],
                "compliance_status": "Compliant",
                "issues": []
            }
            for figure in key_figures or rules.keys():
                if figure not in log or figure not in rules:
                    continue
                value = log[figure]
                rule = rules[figure]
                compliant = True
                issue = None
                
                if rule["operator"] == ">=" and value < rule["value"]:
                    compliant = False
                    issue = f"{figure}: {value} < {rule['value']}"
                elif rule["operator"] == "<=" and value > rule["value"]:
                    compliant = False
                    issue = f"{figure}: {value} > {rule['value']}"
                
                if not compliant:
                    entry["compliance_status"] = "Non-Compliant"
                    entry["issues"].append(issue)
                
                entry[figure] = value
            
            compliance_report.append(entry)

        # Generar tabla en markdown
        headers = ["Date", "Machine"] + (key_figures or list(rules.keys())) + ["Compliance Status", "Issues"]
        table = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for entry in compliance_report:
            row = [entry["date"], entry["machine"]]
            for figure in key_figures or rules.keys():
                value = str(entry.get(figure, "N/A"))
                row.append(value)
            row.append(entry["compliance_status"])
            row.append(", ".join(entry["issues"]) or "None")
            table.append("| " + " | ".join(row) + " |")

        period = specific_date or f"{start_date} a {end_date}"
        filter_text = "\n".join([f"{k}: {v}" for k, v in (key_values or {}).items()]) or "Ninguno"
        report = [
            "Análisis de Cumplimiento",
            "=======================",
            f"Período: {period}",
            f"Filtros Aplicados:\n{filter_text}",
            f"Registros Analizados: {len(compliance_report)}",
            "",
            *table
        ]
        return "\n".join(report)
    except Exception as e:
        logger.error(f"Error en analyze_compliance: {str(e)}")
        return (
            "Análisis de Cumplimiento\n"
            "=======================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo realizar el análisis.\n"
            f"Detalles: {str(e)}"
        )

if __name__ == "__main__":
    init_qdrant()
    mcp.run()