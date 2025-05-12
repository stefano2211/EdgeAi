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

# Inicializar FastMCP para procesar herramientas de cumplimiento MES
mcp = FastMCP("MES Compliance Processor")

# Configuración de la URL de la API MES y el modelo de embeddings
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Inicializar cliente Qdrant para almacenamiento de vectores
qdrant_client = QdrantClient(host="qdrant", port=6333)

# Configurar logging para depuración y monitoreo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFilter(BaseModel):
    """
    Modelo para validar filtros de tiempo (fechas) usados en fetch_mes_data y analyze_compliance.
    
    Atributos:
        start_date (Optional[str]): Fecha de inicio en formato YYYY-MM-DD (opcional).
        end_date (Optional[str]): Fecha de fin en formato YYYY-MM-DD (opcional).
        specific_date (Optional[str]): Fecha específica en formato YYYY-MM-DD (opcional).
    """
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        """
        Valida que las fechas proporcionadas sean válidas y estén en formato correcto.
        
        Raises:
            ValueError: Si las fechas no están en formato YYYY-MM-DD o si start_date > end_date.
        """
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
    """
    Inicializa las colecciones de Qdrant para almacenar logs MES y reglas SOP.
    
    - Crea/recrea las colecciones 'mes_logs' y 'sop_pdfs' con configuración de vectores.
    - Usa distancia COSINE y optimizaciones para indexing y almacenamiento en memoria.
    
    Raises:
        Exception: Si falla la inicialización de las colecciones en Qdrant.
    """
    try:
        qdrant_client.recreate_collection(
            collection_name="mes_logs",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000, memmap_threshold=20000)
        )
        qdrant_client.recreate_collection(
            collection_name="sop_pdfs",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0, memmap_threshold=20000)
        )
        logger.info("Colecciones de Qdrant inicializadas")
    except Exception as e:
        logger.error(f"Error inicializando Qdrant: {str(e)}")
        raise

@mcp.tool()
async def list_available_tools(ctx: Context) -> str:
    """
    Lista todas las herramientas MCP disponibles para el análisis de cumplimiento MES.
    
    Parámetros:
        ctx (Context): Contexto de FastMCP para manejar la solicitud (requerido por mcpo).
    
    Retorna:
        str: JSON con:
            - status: "success" o "error".
            - tools: Lista de nombres de herramientas disponibles (si success).
            - message: Mensaje de error (si error).
    
    Uso por mcpo:
        - Mapea a prompts como "Lista las herramientas disponibles".
        - Útil para depuración o para verificar qué funciones están registradas.
    
    Ejemplo:
        ```json
        {"status": "success", "tools": ["list_available_tools", "list_fields", "fetch_mes_data", "load_sop", "analyze_compliance"]}
        ```
    """
    try:
        tools = [t.__name__ for t in mcp.tools]
        return json.dumps({"status": "success", "tools": tools}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en list_available_tools: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """
    Obtiene los campos disponibles (numéricos y categóricos) y sus valores únicos desde los datos MES.
    
    Parámetros:
        ctx (Context): Contexto de FastMCP para manejar la solicitud (requerido por mcpo).
    
    Retorna:
        str: JSON con:
            - status: "success", "no_data" o "error".
            - fields: Diccionario con:
                - numeric: Lista de campos numéricos (e.g., ["uptime", "temperature"]).
                - categorical: Lista de campos categóricos (e.g., ["machine", "production_line"]).
            - unique_values: Diccionario con valores únicos por campo categórico (e.g., {"machine": ["ModelA", "ModelB"], "production_line": ["Line1", "Line3"]}).
            - message: Mensaje si no hay datos o error.
    
    Uso por mcpo:
        - Mapea a prompts como "Lista las métricas disponibles" o "Muestra los valores categóricos".
        - Proporciona datos para poblar dropdowns/autocomplete en Open WebUI (e.g., máquinas, líneas de producción).
    
    Ejemplo:
        ```json
        {
            "status": "success",
            "fields": {"numeric": ["temperature", "defects"], "categorical": ["machine", "production_line"]},
            "unique_values": {"machine": ["ModelA", "ModelB"], "production_line": ["Line1", "Line3"]}
        }
        ```
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/machines/")
            response.raise_for_status()
            records = response.json()

            if not records:
                return json.dumps({"status": "no_data", "message": "No se encontraron registros", "fields": []})

            sample = records[0]
            fields = {"numeric": [], "categorical": []}
            for field, value in sample.items():
                if field == "id":
                    continue
                if isinstance(value, (int, float)):
                    fields["numeric"].append(field)
                elif isinstance(value, str):
                    fields["categorical"].append(field)

            unique_values = {}
            for field in fields["categorical"]:
                values = sorted({record[field] for record in records if field in record})
                unique_values[field] = values

            return json.dumps({"status": "success", "fields": fields, "unique_values": unique_values}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en list_fields: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

@mcp.tool()
async def fetch_mes_data(
    ctx: Context,
    machine: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> str:
    """
    Recupera datos MES filtrados por máquina, fechas y campos específicos, y los almacena en Qdrant.
    
    Parámetros:
        ctx (Context): Contexto de FastMCP para manejar la solicitud (requerido por mcpo).
        machine (Optional[str]): Nombre de la máquina a filtrar (e.g., "ModelA"). Si None, incluye todas las máquinas.
        start_date (Optional[str]): Fecha de inicio en formato YYYY-MM-DD (para rango de fechas).
        end_date (Optional[str]): Fecha de fin en formato YYYY-MM-DD (para rango de fechas).
        specific_date (Optional[str]): Fecha específica en formato YYYY-MM-DD (prioriza sobre start_date/end_date).
        fields (Optional[List[str]]): Lista de campos a recuperar (e.g., ["temperature", "defects"]). Si None, incluye todos los campos numéricos.
    
    Retorna:
        str: JSON con:
            - status: "success" o "error".
            - count: Número de registros recuperados.
            - data: Lista de registros MES con campos solicitados (e.g., [{"id": "1", "date": "2025-04-09", "machine": "ModelA", "temperature": 74.0}]).
            - message: Mensaje de error (si error).
    
    Uso por mcpo:
        - Mapea a prompts como "Obtén datos MES para ModelA el 2025-04-09" o "Recupera temperatura y defectos desde 2025-04-09 hasta 2025-04-11".
        - Usado internamente por analyze_compliance para obtener datos antes del análisis.
    
    Ejemplo:
        ```json
        {
            "status": "success",
            "count": 2,
            "data": [
                {"id": "1", "date": "2025-04-09", "machine": "ModelA", "temperature": 74.0},
                {"id": "2", "date": "2025-04-11", "machine": "ModelA", "temperature": 73.0}
            ]
        }
        ```
    """
    try:
        time_filter = TimeFilter(start_date=start_date, end_date=end_date, specific_date=specific_date)
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

            processed_data = []
            for record in data:
                processed_record = {"id": record["id"], "date": record["date"], "machine": record["machine"]}
                if fields:
                    for field in fields:
                        if field in record:
                            processed_record[field] = record[field]
                else:
                    fields_info = json.loads(await list_fields(ctx))
                    if fields_info["status"] == "success":
                        numeric_fields = fields_info["fields"]["numeric"]
                        for field in numeric_fields:
                            if field in record:
                                processed_record[field] = record[field]

                processed_data.append(processed_record)

            points = []
            for record in processed_data:
                record_text = json.dumps(record)
                point_id = hashlib.md5(record_text.encode()).hexdigest()
                points.append(models.PointStruct(
                    id=point_id,
                    vector=model.encode(record_text).tolist(),
                    payload=record
                ))

            if points:
                qdrant_client.upsert(collection_name="mes_logs", points=points)

            return json.dumps({"status": "success", "count": len(processed_data), "data": processed_data}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en fetch_mes_data: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

@mcp.tool()
async def load_sop(ctx: Context, machine: str) -> str:
    """
    Carga y parsea las reglas SOP (Standard Operating Procedure) desde un PDF asociado a una máquina, almacenándolas en Qdrant.
    
    Parámetros:
        ctx (Context): Contexto de FastMCP para manejar la solicitud (requerido por mcpo).
        machine (str): Nombre de la máquina para la cual cargar el SOP (e.g., "ModelA"). Busca el PDF `{machine}.pdf`.
    
    Retorna:
        str: JSON con:
            - status: "success", "exists" (si ya está en Qdrant), o "error".
            - machine: Nombre de la máquina.
            - rules: Diccionario de reglas parseadas (e.g., {"temperature": {"value": 75.0, "operator": "<=", "unit": "°C", "rule_text": "temperature <= 75.0°C"}}).
            - message: Mensaje de error (si error).
    
    Uso por mcpo:
        - Mapea a prompts como "Muestra las reglas SOP para ModelA".
        - Usado internamente por analyze_compliance para obtener reglas antes del análisis de cumplimiento.
    
    Ejemplo:
        ```json
        {
            "status": "success",
            "machine": "ModelA",
            "rules": {
                "temperature": {"value": 75.0, "operator": "<=", "unit": "°C", "rule_text": "temperature <= 75.0°C"},
                "defects": {"value": 2, "operator": "<=", "unit": "", "rule_text": "defects <= 2"}
            }
        }
        ```
    """
    try:
        existing = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]),
            limit=1
        )

        if existing[0]:
            return json.dumps({"status": "exists", "machine": machine, "rules": existing[0][0].payload["rules"]}, ensure_ascii=False)

        pdf_name = f"{machine}.pdf"
        async with httpx.AsyncClient() as client:
            list_response = await client.get(f"{API_URL}/pdfs/list")
            if not any(pdf["filename"] == pdf_name for pdf in list_response.json()):
                return json.dumps({"status": "error", "message": f"PDF {pdf_name} no encontrado"}, ensure_ascii=False)

            content_response = await client.get(f"{API_URL}/pdfs/content/", params={"filenames": [pdf_name], "max_length": 10000})
            content = content_response.json()["pdfs"][0]["content"]

        rules = {}
        patterns = [
            (r"(temperature|temperatura)\s*(<=|≤)\s*(\d+\.\d+)\s*°C", "temperature", "<="),
            (r"(vibration|vibración)\s*(<=|≤)\s*(\d+\.\d+)\s*mm/s", "vibration", "<="),
            (r"(defects|defectos)\s*(<=|≤)\s*(\d+)(?:\s|$|,)", "defects", "<="),
            (r"(uptime|tiempo de actividad)\s*(>=|≥)\s*(\d+\.\d+)\s*%", "uptime", ">=")
        ]

        for pattern, field, operator in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(3))
                    unit = "°C" if field == "temperature" else "mm/s" if field == "vibration" else "%" if field == "uptime" else ""
                    rules[field] = {
                        "value": value,
                        "operator": operator,
                        "unit": unit,
                        "rule_text": f"{field} {operator} {value}{unit}"
                    }
                except ValueError:
                    continue

        if not rules:
            return json.dumps({"status": "error", "message": "No se encontraron reglas en el PDF"}, ensure_ascii=False)

        embedding = model.encode(content).tolist()
        qdrant_client.upsert(
            collection_name="sop_pdfs",
            points=[models.PointStruct(
                id=hashlib.md5(machine.encode()).hexdigest(),
                vector=embedding,
                payload={"filename": pdf_name, "machine": machine, "rules": rules}
            )]
        )

        return json.dumps({"status": "success", "machine": machine, "rules": rules}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error en load_sop: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    machine: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    focus_numeric_fields: Optional[List[str]] = None,
    categorical_filters: Optional[dict] = None
) -> str:
    """
    Analiza el cumplimiento de datos MES contra reglas SOP para una o varias máquinas, con filtros de tiempo y categóricos.
    
    Parámetros:
        ctx (Context): Contexto de FastMCP para manejar la solicitud (requerido por mcpo).
        machine (Optional[str]): Nombre de la máquina a analizar (e.g., "ModelA"). Si None, analiza todas las máquinas.
        start_date (Optional[str]): Fecha de inicio en formato YYYY-MM-DD (para rango de fechas).
        end_date (Optional[str]): Fecha de fin en formato YYYY-MM-DD (para rango de fechas).
        specific_date (Optional[str]): Fecha específica en formato YYYY-MM-DD (prioriza sobre start_date/end_date).
        focus_numeric_fields (Optional[List[str]]): Lista de campos numéricos a analizar (e.g., ["temperature", "defects"]). Si None, incluye todos los campos numéricos.
        categorical_filters (Optional[dict]): Diccionario de filtros categóricos (e.g., {"production_line": "Line1", "material": "Steel"}). Si None, no aplica filtros categóricos.
    
    Retorna:
        str: JSON con:
            - status: "success", "error", "no_data", o "invalid_filter".
            - period: Período analizado (fecha específica o rango).
            - machine_filter: Máquina analizada o "all_machines".
            - numeric_fields_analyzed: Campos numéricos analizados.
            - categorical_filters_applied: Filtros categóricos aplicados.
            - sop_summary: Resumen de máquinas con/sin SOP.
            - results: Lista de registros con:
                - id: ID del registro MES.
                - date: Fecha del registro.
                - machine: Nombre de la máquina.
                - metrics: Diccionario de valores de métricas.
                - compliance_status: Estado de cumplimiento por métrica (compliant/non_compliant/unknown).
                - sop_status: "available" o "not_available".
                - filtered_fields: Filtros categóricos aplicados.
                - compliance_percentage: Porcentaje de cumplimiento (0-100).
            - analysis_note: Nota sobre el análisis.
            - message: Mensaje de error o descripción (si no es success).
    
    Uso por mcpo:
        - Mapea a prompts como "Analiza el cumplimiento para ModelA el 2025-04-09 con filtro production_line=Line1" o "Analiza temperatura desde 2025-04-09 hasta 2025-04-11".
        - Principal función para generar reportes de cumplimiento, usando fetch_mes_data y load_sop internamente.
    
    Ejemplo:
        ```json
        {
            "status": "success",
            "period": "2025-04-09 a 2025-04-11",
            "machine_filter": "ModelA",
            "numeric_fields_analyzed": ["temperature"],
            "categorical_filters_applied": {"production_line": "Line1"},
            "sop_summary": {"machines_with_sop": 1, "machines_without_sop": 0},
            "results": [
                {
                    "id": "1",
                    "date": "2025-04-09",
                    "machine": "ModelA",
                    "metrics": {"temperature": 74.0},
                    "compliance_status": {
                        "temperature": {"value": 74.0, "rule": "<= 75.0°C", "status": "compliant"}
                    },
                    "sop_status": "available",
                    "filtered_fields": {"production_line": "Line1"},
                    "compliance_percentage": 100.0
                }
            ],
            "analysis_note": "Cada métrica incluye valor actual, regla SOP, y estado de cumplimiento"
        }
        ```
    """
    try:
        # 1. Validación de fechas
        time_filter = TimeFilter(start_date=start_date, end_date=end_date, specific_date=specific_date)
        time_filter.validate_dates()

        # 2. Normalización de filtros categóricos
        norm_filters = {}
        if categorical_filters:
            for field, value in categorical_filters.items():
                if isinstance(value, list) and len(value) > 0:
                    norm_filters[field] = str(value[0])
                else:
                    norm_filters[field] = str(value)

        # 3. Validar filtros categóricos
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({"status": "error", "message": "No se pudieron obtener los campos disponibles"})

        for field, value in norm_filters.items():
            if field not in fields_info["unique_values"]:
                return json.dumps({
                    "status": "invalid_filter",
                    "message": f"El campo categórico '{field}' no es válido. Campos disponibles: {list(fields_info['unique_values'].keys())}"
                })
            if value not in fields_info["unique_values"][field]:
                return json.dumps({
                    "status": "invalid_filter",
                    "message": f"El valor '{value}' no es válido para el campo '{field}'. Valores disponibles: {fields_info['unique_values'][field]}"
                })

        # 4. Determinar máquinas
        all_machines = fields_info["unique_values"].get("machine", [])
        target_machines = [machine] if machine else all_machines

        if machine and machine not in all_machines:
            return json.dumps({
                "status": "invalid_filter",
                "message": f"La máquina '{machine}' no es válida. Máquinas disponibles: {all_machines}"
            })

        # 5. Cargar reglas SOP
        machines_rules = {}
        for machine_name in target_machines:
            sop_result = json.loads(await load_sop(ctx, machine_name))
            if sop_result["status"] in ["success", "exists"]:
                machines_rules[machine_name] = sop_result.get("rules", {})
            else:
                machines_rules[machine_name] = {"status": "no_sop"}

        # 6. Determinar campos numéricos
        numeric_fields = focus_numeric_fields if focus_numeric_fields else fields_info["fields"]["numeric"]

        # 7. Obtener y procesar datos
        all_results = []
        for machine_name in target_machines:
            fetch_result = json.loads(await fetch_mes_data(
                ctx,
                machine=machine_name,
                start_date=start_date,
                end_date=end_date,
                specific_date=specific_date,
                fields=list(set(numeric_fields + ["id", "date", "machine"] + list(norm_filters.keys())))
            ))

            if fetch_result["status"] != "success" or not fetch_result["data"]:
                filter_desc = f"máquina '{machine_name}'"
                if specific_date:
                    filter_desc += f" en {specific_date}"
                if norm_filters:
                    filter_desc += f" con filtros {norm_filters}"
                return json.dumps({
                    "status": "no_data",
                    "message": f"No se encontraron datos para {filter_desc}",
                    "results": []
                })

            # Filtrar registros
            filtered_data = []
            for record in fetch_result["data"]:
                match = True
                for field, value in norm_filters.items():
                    if str(record.get(field)) != value:
                        match = False
                        break
                if match:
                    filtered_data.append(record)

            if not filtered_data:
                filter_desc = f"máquina '{machine_name}'"
                if specific_date:
                    filter_desc += f" en {specific_date}"
                if norm_filters:
                    filter_desc += f" con filtros {norm_filters}"
                return json.dumps({
                    "status": "no_data",
                    "message": f"No se encontraron datos para {filter_desc}",
                    "results": []
                })

            # Procesar cumplimiento
            for record in filtered_data:
                entry = {
                    "id": record["id"],
                    "date": record["date"],
                    "machine": record["machine"],
                    "metrics": {},
                    "compliance_status": {},
                    "sop_status": "available" if machines_rules[machine_name].get("status") != "no_sop" else "not_available",
                    "filtered_fields": norm_filters
                }

                total_metrics = 0
                compliant_metrics = 0
                for field in numeric_fields:
                    if field in record:
                        total_metrics += 1
                        metric_value = record[field]
                        entry["metrics"][field] = metric_value
                        rule = machines_rules[machine_name].get(field)
                        if rule and "value" in rule:
                            rule_value = rule["value"]
                            operator = rule["operator"]
                            unit = rule["unit"]
                            is_compliant = (
                                metric_value <= rule_value if operator == "<=" else
                                metric_value >= rule_value
                            )
                            entry["compliance_status"][field] = {
                                "value": metric_value,
                                "rule": f"{operator} {rule_value}{unit}",
                                "status": "compliant" if is_compliant else "non_compliant"
                            }
                            if is_compliant:
                                compliant_metrics += 1
                        else:
                            entry["compliance_status"][field] = {
                                "value": metric_value,
                                "rule": "no_rule_defined",
                                "status": "unknown"
                            }

                compliance_percentage = (compliant_metrics / total_metrics * 100) if total_metrics > 0 else 0
                entry["compliance_percentage"] = round(compliance_percentage, 2)

                all_results.append(entry)

        # 8. Resumen de SOPs
        sop_summary = {
            "machines_with_sop": sum(1 for rules in machines_rules.values() if rules.get("status") != "no_sop"),
            "machines_without_sop": sum(1 for rules in machines_rules.values() if rules.get("status") == "no_sop")
        }

        return json.dumps({
            "status": "success",
            "period": specific_date or f"{start_date} a {end_date}",
            "machine_filter": machine or "all_machines",
            "numeric_fields_analyzed": numeric_fields,
            "categorical_filters_applied": norm_filters,
            "sop_summary": sop_summary,
            "results": all_results,
            "analysis_note": "Cada métrica incluye valor actual, regla SOP, y estado de cumplimiento (compliant/non_compliant/unknown)"
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error en analyze_compliance: {str(e)}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Error al analizar cumplimiento: {str(e)}",
            "parameters_received": {
                "machine": machine,
                "date_range": f"{start_date} to {end_date}",
                "numeric_fields": focus_numeric_fields,
                "categorical_filters": categorical_filters
            }
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_collections()
    mcp.run()