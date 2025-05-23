import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime
from typing import Optional, List, Dict, Union
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

# Initialize MCP service
mcp = FastMCP("Manufacturing Compliance Processor")

# Global configuration
API_URL = os.getenv("API_URL", "http://api:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sop-pdfs")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password123")

# Component initialization
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="qdrant", port=6333)
minio_client = Minio(MINIO_ENDPOINT, 
                    access_key=MINIO_ACCESS_KEY, 
                    secret_key=MINIO_SECRET_KEY, 
                    secure=False)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthClient:
    """Cliente HTTP para manejar autenticación y solicitudes autenticadas a la API."""
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.client = httpx.AsyncClient()
        self.token = None

    async def authenticate(self):
        """Autentica contra la API y obtiene un token de sesión.

        Realiza una solicitud POST al endpoint /login de la API con las credenciales
        proporcionadas. Almacena el token JWT recibido para usarlo en solicitudes futuras.

        Raises:
            ValueError: Si la autenticación falla debido a credenciales inválidas o error del servidor.
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/login",
                json={"username": self.username, "password": self.password}
            )
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            logger.info("Autenticación exitosa, token obtenido")
        except Exception as e:
            logger.error(f"Fallo en autenticación: {str(e)}")
            raise ValueError(f"No se pudo autenticar: {str(e)}")

    async def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        """Realiza una solicitud GET autenticada a la API.

        Usa el token almacenado para autenticar la solicitud. Si el token no existe,
        realiza la autenticación primero. Si recibe un error 401, reautentica y reintenta.

        Args:
            endpoint (str): Endpoint de la API (e.g., "/machines/").
            params (Optional[Dict]): Parámetros de consulta para la solicitud.

        Returns:
            httpx.Response: Respuesta de la API.

        Raises:
            Exception: Si la solicitud falla después de reintentar la autenticación.
        """
        if not self.token:
            await self.authenticate()
        
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = await self.client.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            if response.status_code == 401:
                logger.info("Token inválido, reautenticando...")
                await self.authenticate()
                headers = {"Authorization": f"Bearer {self.token}"}
                response = await self.client.get(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    params=params
                )
            return response
        except Exception as e:
            logger.error(f"Error en solicitud GET: {str(e)}")
            raise

    async def close(self):
        """Cierra el cliente HTTP."""
        await self.client.aclose()

# Cliente global
auth_client = None

def init_infrastructure():
    """Inicializa la infraestructura del MCP, incluyendo Qdrant, MinIO y el cliente autenticado.

    Configura las colecciones en Qdrant para almacenar logs, PDFs y reglas personalizadas,
    crea el bucket en MinIO si no existe, e inicializa el cliente autenticado para la API.

    Raises:
        Exception: Si falla la inicialización de algún componente.
    """
    global auth_client
    try:
        vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
        qdrant_client.recreate_collection(collection_name="mes_logs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="sop_pdfs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="custom_rules", vectors_config=vector_config)
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
        auth_client = AuthClient(API_URL, API_USERNAME, API_PASSWORD)
    except Exception as e:
        logger.error(f"Infrastructure initialization failed: {str(e)}")
        raise

class DataValidator:
    """Valida key_figures y key_values contra la estructura de datos de la API."""
    
    @staticmethod
    def validate_date(date_str: str, field: str) -> None:
        """Valida que una fecha tenga el formato YYYY-MM-DD.

        Args:
            date_str (str): Cadena de fecha a validar.
            field (str): Nombre del campo (para mensajes de error).

        Raises:
            ValueError: Si la fecha no tiene el formato correcto.
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid format for {field}. Use YYYY-MM-DD: {date_str}")

    @staticmethod
    async def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict[str, str]) -> Dict:
        """Valida que los campos solicitados sean válidos según la estructura de la API.

        También valida las fechas (start_date y end_date) si están presentes en key_values,
        asegurando que tengan el formato correcto y que start_date no sea posterior a end_date.

        Args:
            ctx (Context): Contexto de la solicitud FastMCP.
            key_figures (List[str]): Lista de campos numéricos a validar.
            key_values (Dict[str, str]): Diccionario de campos categóricos y sus valores,
                incluyendo fechas (start_date, end_date).

        Returns:
            Dict: Información de campos válidos obtenida de list_fields.

        Raises:
            ValueError: Si los campos o fechas no son válidos o si list_fields falla.
        """
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Could not validate fields against API")

        if "start_date" in key_values or "end_date" in key_values:
            if "start_date" not in key_values or "end_date" not in key_values:
                raise ValueError("Both start_date and end_date must be provided")
            DataValidator.validate_date(key_values["start_date"], "start_date")
            DataValidator.validate_date(key_values["end_date"], "end_date")
            if key_values["start_date"] > key_values["end_date"]:
                raise ValueError("start_date cannot be after end_date")

        invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
        invalid_values = {
            k: v for k, v in key_values.items()
            if k not in fields_info["key_values"] and k not in ["start_date", "end_date"]
            or (k in fields_info["key_values"] and v not in fields_info["key_values"].get(k, []))
        }
        
        if invalid_figures or invalid_values:
            errors = []
            if invalid_figures:
                errors.append(f"Invalid key_figures: {invalid_figures}")
            if invalid_values:
                errors.append(f"Invalid key_values: {invalid_values}")
            raise ValueError(" | ".join(errors))
            
        return fields_info

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """Lista los campos disponibles en los datos de la API MES.
    """
    try:
        response = await auth_client.get("/machines/")
        response.raise_for_status()
        records = response.json()

        if not records:
            return json.dumps({
                "status": "no_data",
                "message": "No records found in MES system",
                "key_figures": [],
                "key_values": {}
            })

        sample = records[0]
        key_figures = []
        key_values = {}
        
        for field, value in sample.items():
            if field == "id":
                continue
            if isinstance(value, (int, float)):
                key_figures.append(field)
            elif isinstance(value, str):
                unique_values = sorted({rec[field] for rec in records if field in rec})
                key_values[field] = unique_values

        return json.dumps({
            "status": "success",
            "key_figures": key_figures,
            "key_values": key_values
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Field listing failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "key_figures": [],
            "key_values": {}
        }, ensure_ascii=False)

@mcp.tool()
async def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None
) -> str:
    """Obtiene datos de la API MES o Qdrant, filtra dinámicamente y almacena en la base de datos vectorial.
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validate requested fields and dates
        await DataValidator.validate_fields(ctx, key_figures, key_values)
        
        # Check Qdrant for existing data
        must_conditions = []
        if "machine" in key_values:
            must_conditions.append(models.FieldCondition(key="machine", match=models.MatchValue(value=key_values["machine"])))
        if "start_date" in key_values and "end_date" in key_values:
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchText(text=f"[{key_values['start_date']} TO {key_values['end_date']}]")
            ))
        for k, v in key_values.items():
            if k not in ["machine", "start_date", "end_date"]:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )

        processed_data = [r.payload for r in qdrant_results[0]] if qdrant_results[0] else []

        # If no data in Qdrant, fetch from API
        if not processed_data:
            # Build query parameters
            params = {}
            if "start_date" in key_values and "end_date" in key_values:
                params.update({
                    "start_date": key_values["start_date"],
                    "end_date": key_values["end_date"]
                })

            # Filter out date-related keys for data filtering
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}

            # Determine API endpoint
            endpoint = f"/machines/{data_filters['machine']}" if "machine" in data_filters else "/machines/"

            # Fetch data
            response = await auth_client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Filter and process records
            processed_data = []
            for record in data:
                if all(record.get(k) == v for k, v in data_filters.items() if k != "machine"):
                    item = {
                        "id": record["id"],
                        "date": record["date"],
                        "machine": record["machine"]
                    }
                    item.update({f: record[f] for f in key_figures if f in record})
                    processed_data.append(item)

            # Store in Qdrant
            if processed_data:
                points = [
                    models.PointStruct(
                        id=hashlib.md5(json.dumps(r).encode()).hexdigest(),
                        vector=model.encode(json.dumps(r)).tolist(),
                        payload=r
                    ) for r in processed_data
                ]
                qdrant_client.upsert(collection_name="mes_logs", points=points)

        return json.dumps({
            "status": "success",
            "count": len(processed_data),
            "data": processed_data
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "data": []
        }, ensure_ascii=False)


@mcp.tool()
async def load_sop(ctx: Context, machine: str) -> str:
    """Carga y procesa un documento SOP (PDF) para una máquina específica desde MinIO."""
    try:
        # Check if SOP already exists in Qdrant
        existing = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]),
            limit=1
        )
        
        if existing[0]:
            return json.dumps({
                "status": "exists",
                "machine": machine,
                "rules": existing[0][0].payload["rules"]
            }, ensure_ascii=False)

        # Load PDF from MinIO
        pdf_name = f"{machine}.pdf"
        try:
            response = minio_client.get_object(MINIO_BUCKET, pdf_name)
            pdf_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            available_pdfs = [obj.object_name for obj in minio_client.list_objects(MINIO_BUCKET)]
            return json.dumps({
                "status": "error",
                "message": f"PDF not found. Available SOPs: {', '.join(available_pdfs)}",
                "machine": machine
            }, ensure_ascii=False)

        # Extract text content
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)

        # Extract compliance rules using flexible patterns
        rules = {}
        patterns = [
            (r"(?P<field>cycle_time|ciclo)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*segundos?", "<=", "s"),
            (r"(?P<field>error_count|errores)\s*(?P<operator><=|≤)\s*(?P<value>\d+)", "<=", ""),
            (r"(?P<field>pressure|presión)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*bar", "<=", "bar\\[0.5mm]bar", "<=", "bar"),
            (r"(?P<field>power_consumption|consumo)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*kW", "<=", "kW"),
            (r"(?P<field>output_rate|producción)\s*(?P<operator>>=|≥)\s*(?P<value>\d+\.\d+)\s*unidades/h", ">=", "units/h")
        ]

        for pattern, default_op, unit in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    field = match.group("field").lower().replace(" ", "_")
                    if field in ["ciclo", "errores", "presión", "consumo", "producción"]:
                        field = {
                            "ciclo": "cycle_time",
                            "errores": "error_count",
                            "presión": "pressure",
                            "consumo": "power_consumption",
                            "producción": "output_rate"
                        }[field]
                    rules[field] = {
                        "value": float(match.group("value")),
                        "operator": match.group("operator") if match.group("operator") else default_op,
                        "unit": unit,
                        "source_text": match.group(0)
                    }
                except Exception:
                    continue

        if not rules:
            return json.dumps({
                "status": "error",
                "message": "No compliance rules found in document",
                "machine": machine
            }, ensure_ascii=False)

        # Store in vector database
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
        logger.error(f"SOP processing failed for {machine}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "machine": machine
        }, ensure_ascii=False)



@mcp.tool()
async def add_custom_rule(
    ctx: Context,
    machines: Union[List[str], str],
    key_figures: Union[Dict[str, float], str],  # Acepta dict o string
    key_values: Optional[Dict[str, str]] = None,
    operator: str = "<=",
    unit: Optional[str] = None,
    description: str = ""
) -> str:
    """Añade una regla de cumplimiento personalizada para múltiples máquinas y métricas.

    Versión mejorada que acepta:
    - Dict: {"temperature": 70.0, "pressure": 1.2}
    - String: "temperature=70,pressure=1.2" o "temperature:70,pressure:1.2"

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        machines (Union[List[str], str]): Lista de máquinas para las cuales aplica la regla (e.g., ["ModelA", "ModelB"])
            o string JSON (Example, '["ModelA"]'). Se parseará automáticamente si es string.
        key_figures (Union[Dict[str, Dict[str, float]], str]): Diccionario de campos numéricos y sus valores umbral
            (Example, {"temperature": {"value": 80.0}}) o string JSON (Example, '{"temperature": {"value": 80.0}}').
            Se parseará automáticamente si es string.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos para filtrar
            (Example, {"material": "Steel", "batch": "B123"}). Por defecto None.
        operator (str): Operador de la regla, debe ser uno de: ">=", "<=", ">", "<", "==", "!=".
            Por defecto "<=".
        unit (Optional[str]): Unidad de medida para los key_figures (e.g., "°C"). Por defecto None.
        description (str): Descripción de la regla (Example., "Temperatura máxima por experiencia").
            Por defecto "".

    Returns:
        str: JSON con estado y detalles de la regla creada.

    Ejemplo de uso LLM:
        ```json
        {
            "machines": ["ModelA"],
            "key_figures": {"temperature": 70.0},
            "operator": ">=",
            "unit": "°C",
            "description": "Temperatura mínima requerida"
        }
        ```
        o
        ```json
        {
            "machines": ["ModelA", "ModelB"],
            "key_figures": "temperature=70,pressure=1.2",
            "operator": "<=",
            "description": "Límites superiores"
        }
        ```
    """
    try:
        # Parse machines
        if isinstance(machines, str):
            try:
                machines = json.loads(machines)
            except json.JSONDecodeError:
                raise ValueError("Formato inválido para machines. Use lista JSON")

        if not isinstance(machines, list):
            raise ValueError("machines debe ser una lista de strings")

        # Parse key_figures (acepta dict o string)
        if isinstance(key_figures, str):
            parsed_figures = {}
            for pair in key_figures.split(','):
                # Soporta = o : como separador
                if '=' in pair:
                    field, value = pair.split('=', 1)
                elif ':' in pair:
                    field, value = pair.split(':', 1)
                else:
                    raise ValueError(f"Formato inválido: {pair}. Use 'campo=valor' o 'campo:valor'")
                
                field = field.strip()
                try:
                    parsed_figures[field] = float(value.strip())
                except ValueError:
                    raise ValueError(f"Valor inválido para {field}: debe ser numérico")
            key_figures = parsed_figures
        elif isinstance(key_figures, dict):
            # Validar que los valores sean numéricos
            for field, value in key_figures.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Valor para {field} debe ser numérico")
        else:
            raise ValueError("key_figures debe ser dict o string")

        # Validación de datos
        if not machines:
            raise ValueError("Debe especificar al menos una máquina")

        if not key_figures:
            raise ValueError("Debe especificar al menos una métrica")

        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("No se pudo validar contra la API")

        # Validar máquinas
        valid_machines = fields_info["key_values"].get("machine", [])
        invalid_machines = [m for m in machines if m not in valid_machines]
        if invalid_machines:
            raise ValueError(f"Máquinas inválidas: {invalid_machines}")

        # Validar métricas
        invalid_metrics = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_metrics:
            raise ValueError(f"Métricas inválidas: {invalid_metrics}")

        # Validar operador
        valid_operators = [">=", "<=", ">", "<", "==", "!="]
        if operator not in valid_operators:
            raise ValueError(f"Operador inválido. Use uno de: {valid_operators}")

        # Validar key_values
        if key_values:
            for k, v in key_values.items():
                if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, []):
                    raise ValueError(f"Filtro inválido: {k}={v}")

        # Preparar regla final (convertir valores float)
        final_rule = {
            "machines": machines,
            "key_figures": {k: {"value": float(v)} for k, v in key_figures.items()},
            "key_values": key_values or {},
            "operator": operator,
            "unit": unit,
            "description": description
        }

        # Almacenar en Qdrant
        embedding_text = description or " ".join(
            [f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()]
        )
        embedding = model.encode(embedding_text).tolist()

        qdrant_client.upsert(
            collection_name="custom_rules",
            points=[models.PointStruct(
                id=hashlib.md5(json.dumps(final_rule).encode()).hexdigest(),
                vector=embedding,
                payload=final_rule
            )]
        )

        # Mensaje descriptivo
        metrics_desc = ", ".join(
            [f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()]
        )
        filters_desc = ", ".join([f"{k}={v}" for k, v in (key_values or {}).items()])
        
        message = f"Regla añadida para {len(machines)} máquina(s): {metrics_desc}"
        if filters_desc:
            message += f" | Filtros: {filters_desc}"

        return json.dumps({
            "status": "success",
            "message": message,
            "rule": final_rule,
            "details": {
                "machines_count": len(machines),
                "metrics_count": len(key_figures),
                "filters_count": len(key_values or {})
            }
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error al añadir regla: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "input_parameters": {
                "machines": machines,
                "key_figures": key_figures,
                "key_values": key_values,
                "operator": operator,
                "unit": unit,
                "description": description
            }
        }, ensure_ascii=False)


@mcp.tool()
async def list_custom_rules(
    ctx: Context,
    rule_id: Optional[str] = None,
    machine: Optional[str] = None,
    limit: int = 10
) -> str:
    """Lista todas las reglas personalizadas o filtra por ID/máquina.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        rule_id (Optional[str]): ID específico de la regla a buscar.
        machine (Optional[str]): Nombre de máquina para filtrar reglas.
        limit (int): Límite de resultados a devolver (default: 10).

    Returns:
        str: JSON con lista de reglas y sus metadatos.

    Ejemplo de uso LLM:
        ```json
        {"rule_id": "a1b2c3d4"}
        ```
        o
        ```json
        {"machine": "ModelA", "limit": 5}
        ```
    """
    try:
        # Construir filtro para Qdrant
        filter_conditions = []
        
        if rule_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=rule_id)
                )
            )
        
        if machine:
            filter_conditions.append(
                models.FieldCondition(
                    key="machines",
                    match=models.MatchAny(any=[machine])
                )
            )

        scroll_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        # Obtener reglas de Qdrant
        rules = qdrant_client.scroll(
            collection_name="custom_rules",
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )

        # Formatear resultados
        formatted_rules = []
        for rule in rules[0]:
            formatted_rules.append({
                "id": rule.id,
                "machines": rule.payload.get("machines", []),
                "key_figures": rule.payload.get("key_figures", {}),
                "operator": rule.payload.get("operator", ""),
                "unit": rule.payload.get("unit", ""),
                "description": rule.payload.get("description", ""),
                "created_at": rule.payload.get("created_at", ""),
                "applies_to": f"{len(rule.payload.get('machines', []))} máquina(s)",
                "metrics": list(rule.payload.get("key_figures", {}).keys())
            })

        return json.dumps({
            "status": "success",
            "count": len(formatted_rules),
            "rules": formatted_rules,
            "metadata": {
                "collection": "custom_rules",
                "limit": limit,
                "filters": {
                    "by_id": bool(rule_id),
                    "by_machine": machine if machine else None
                }
            }
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error listing rules: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "rules": []
        }, ensure_ascii=False)


@mcp.tool()
async def delete_custom_rule(
    ctx: Context,
    rule_id: str
) -> str:
    """Elimina una regla personalizada por su ID.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        rule_id (str): ID de la regla a eliminar.

    Returns:
        str: JSON con estado de la operación.

    Ejemplo de uso LLM:
        ```json
        {"rule_id": "a1b2c3d4"}
        ```
    """
    try:
        # Verificar si existe la regla
        existing = qdrant_client.retrieve(
            collection_name="custom_rules",
            ids=[rule_id],
            with_payload=True
        )

        if not existing:
            return json.dumps({
                "status": "error",
                "message": f"Regla con ID {rule_id} no encontrada"
            }, ensure_ascii=False)

        # Eliminar la regla
        qdrant_client.delete(
            collection_name="custom_rules",
            points_selector=models.PointIdsList(
                points=[rule_id]
            )
        )

        # Obtener detalles para mensaje
        rule_data = existing[0].payload
        metrics = list(rule_data.get("key_figures", {}).keys())
        machines = rule_data.get("machines", [])

        return json.dumps({
            "status": "success",
            "message": f"Regla eliminada: {', '.join(metrics)} para {len(machines)} máquina(s)",
            "deleted_rule": {
                "id": rule_id,
                "affected_machines": machines,
                "metrics": metrics,
                "operator": rule_data.get("operator", ""),
                "description": rule_data.get("description", "")
            }
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error deleting rule {rule_id}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "rule_id": rule_id
        }, ensure_ascii=False)


        
@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None
) -> str:
    """Analiza el cumplimiento de los datos MES contra reglas SOP y personalizadas.

    Esta herramienta realiza las siguientes acciones:
    1. Valida los campos solicitados (key_figures y key_values) usando DataValidator.
    2. Obtiene datos de la API MES o Qdrant usando fetch_mes_data.
    3. Carga reglas SOP y personalizadas para las máquinas relevantes.
    4. Compara cada registro contra todas las reglas, aplicando filtros de key_values.
    5. Calcula un porcentaje de cumplimiento por registro (basado solo en reglas SOP).
    6. Devuelve un informe detallado con resultados de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos y valores
            para filtrar (Example, {"equipment_id": "EquipA", "operation_mode": "Auto", "product_type":"WidgetC", "status":"Running","start_date": "2025-04-09",
            "end_date": "2025-04-11"}). Las fechas deben estar en formato YYYY-MM-DD.
        key_figures (Optional[List[str]]): Lista de campos numéricos a analizar
            (Example, ["pressure", "uptime"]).

    Returns:
        str: Cadena JSON con el estado, período, filtro de máquina, métricas analizadas,
            resultados del análisis, cobertura de reglas, y notas.
            Ejemplo:
            {
                "status": "success",
                "period": "2025-04-09 to 2025-04-11",
                "machine_filter": "ModelA",
                "metrics_analyzed": ["temperature"],
                "results": [
                    {
                        "id": 1,
                        "date": "2025-04-10",
                        "machine": "ModelA",
                        "metrics": {"temperature": 75.0},
                        "compliance": {
                            "temperature": {
                                "sop": {"value": 75.0, "rule": "<= 80.0°C", "status": "compliant"},
                                "custom": [
                                    {
                                        "value": 75.0,
                                        "rule": "<= 78.0°C",
                                        "status": "compliant",
                                        "description": "Temperatura máxima por experiencia",
                                        "filters": {"material": "Steel"}
                                    }
                                ]
                            }
                        },
                        "compliance_percentage": 100.0
                    }
                ],
                "sop_coverage": "1/1 machines with SOP",
                "custom_rules_applied": "1/1 machines with custom rules",
                "analysis_notes": [
                    "sop: Rules from Standard Operating Procedures",
                    "custom: User-defined expert rules",
                    "compliant: Meets rule requirement",
                    "non_compliant: Does not meet rule requirement",
                    "unknown: No rule defined"
                ]
            }
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validate requested fields and dates
        fields_info = await DataValidator.validate_fields(ctx, key_figures, key_values)
        
        # Fetch manufacturing data
        fetch_result = json.loads(await fetch_mes_data(ctx, key_values, key_figures))
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Data retrieval failed"),
                "results": []
            }, ensure_ascii=False)

        # Load SOP rules for relevant machines
        machines = {r["machine"] for r in fetch_result["data"]}
        machine_rules = {}
        for machine in machines:
            sop_result = json.loads(await load_sop(ctx, machine))
            machine_rules[machine] = sop_result.get("rules", {}) if sop_result["status"] in ["success", "exists"] else {}

        # Load custom rules from Qdrant
        custom_rules = []
        custom_result = qdrant_client.scroll(
            collection_name="custom_rules",
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="machines", match=models.MatchAny(any=list(machines)))
            ]),
            limit=100
        )
        custom_rules = [r.payload for r in custom_result[0]] if custom_result[0] else []

        # Analyze compliance for each record
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "id": record["id"],
                "date": record["date"],
                "machine": record["machine"],
                "metrics": {f: record[f] for f in key_figures if f in record},
                "compliance": {},
                "compliance_percentage": 0.0
            }
            
            compliant = 0
            total = 0
            
            for field in key_figures:
                if field not in record:
                    continue
                    
                total += 1
                compliance_info = {}
                
                # Check SOP rule
                sop_rules = machine_rules.get(record["machine"], {})
                if field in sop_rules:
                    rule = sop_rules[field]
                    value = record[field]
                    op = rule["operator"]
                    is_compliant = (value >= rule["value"]) if op == ">=" else (value <= rule["value"])
                    compliance_info["sop"] = {
                        "value": value,
                        "rule": f"{op} {rule['value']}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }
                    compliant += 1 if is_compliant else 0
                else:
                    compliance_info["sop"] = {
                        "value": record[field],
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }
                
                # Check custom rules
                compliance_info["custom"] = []
                for rule in custom_rules:
                    if field not in rule["key_figures"]:
                        continue
                    if record["machine"] not in rule["machines"]:
                        continue
                    if rule["key_values"]:
                        if not all(record.get(k) == v for k, v in rule["key_values"].items()):
                            continue
                    rule_value = rule["key_figures"][field]["value"]
                    op = rule["operator"]
                    value = record[field]
                    is_compliant = False
                    if op == ">=":
                        is_compliant = value >= rule_value
                    elif op == "<=":
                        is_compliant = value <= rule_value
                    elif op == ">":
                        is_compliant = value > rule_value
                    elif op == "<":
                        is_compliant = value < rule_value
                    elif op == "==":
                        is_compliant = value == rule_value
                    elif op == "!=":
                        is_compliant = value != rule_value
                    compliance_info["custom"].append({
                        "value": value,
                        "rule": f"{op} {rule_value}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant",
                        "description": rule.get("description", ""),
                        "filters": rule["key_values"]
                    })
                
                if not compliance_info["custom"]:
                    compliance_info["custom"].append({
                        "value": record[field],
                        "rule": "no_rule_defined",
                        "status": "unknown",
                        "description": "",
                        "filters": {}
                    })
                
                analysis["compliance"][field] = compliance_info
            
            if total > 0:
                analysis["compliance_percentage"] = round((compliant / total) * 100, 2)
            
            results.append(analysis)

        # Format time period description
        period = "all dates"
        if "start_date" in key_values and "end_date" in key_values:
            period = f"{key_values['start_date']} to {key_values['end_date']}"

        return json.dumps({
            "status": "success",
            "period": period,
            "machine_filter": key_values.get("machine", "all machines"),
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sum(1 for r in machine_rules.values() if r)}/{len(machines)} machines with SOP",
            "custom_rules_applied": f"{len(custom_rules)} custom rules applied",
            "analysis_notes": [
                "sop: Rules from Standard Operating Procedures",
                "custom: User-defined expert rules",
                "compliant: Meets rule requirement",
                "non_compliant: Does not meet rule requirement",
                "unknown: No rule defined"
            ]
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Compliance analysis failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "results": []
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_infrastructure()
    mcp.run()