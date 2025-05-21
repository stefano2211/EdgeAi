import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import logging
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

# In-memory caches
sop_cache = {}
expert_rules_cache = {}

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

    Configura las colecciones en Qdrant para almacenar logs, PDFs y reglas expertas, crea el bucket en MinIO
    si no existe, e inicializa el cliente autenticado para la API.

    Raises:
        Exception: Si falla la inicialización de algún componente.
    """
    global auth_client
    try:
        vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
        qdrant_client.recreate_collection(collection_name="mes_logs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="sop_pdfs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="expert_rules", vectors_config=vector_config)
        
        # MinIO configuration
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

    Esta herramienta consulta la API para obtener una muestra de registros y clasifica
    los campos en dos categorías:
    - key_figures: Campos numéricos (int o float) que pueden usarse para métricas.
    - key_values: Campos categóricos (strings) con sus valores únicos.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP, usado para autenticación y logging.

    Returns:
        str: Cadena JSON con el estado, key_figures, key_values, y mensaje de error si aplica.
            Ejemplo:
            {
                "status": "success",
                "key_figures": ["uptime", "defects", "temperature"],
                "key_values": {
                    "machine": ["ModelA", "ModelB"],
                    "material": ["Steel", "Aluminum"],
                    "date": ["2025-04-09", "2025-04-10"]
                }
            }

    Raises:
        Exception: Si falla la solicitud a la API o el procesamiento de los datos.
                  Devuelve un JSON con status="error" y el mensaje de error.

    Ejemplo de uso:
        await list_fields(ctx)
        # Retorna JSON con los campos disponibles para filtrar y analizar.
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
                logger.info(f"Detectado key_figure: {field}")
            elif isinstance(value, str):
                unique_values = sorted({rec[field] for rec in records if field in rec})
                key_values[field] = unique_values
                logger.info(f"Detectado key_value: {field} con valores {unique_values}")

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

    Esta herramienta realiza las siguientes acciones:
    1. Valida los campos solicitados (key_figures y key_values) contra la estructura de la API.
    2. Verifica si los datos ya están en Qdrant para los filtros especificados.
    3. Si no están en Qdrant, construye parámetros de consulta basados en key_values, incluyendo fechas.
    4. Consulta la API para obtener datos, filtrando por máquina si se especifica en key_values.
    5. Procesa los datos para incluir solo los campos solicitados.
    6. Almacena los datos procesados en Qdrant con embeddings generados por el modelo.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos y valores
            para filtrar (e.g., {"machine": "ModelA", "material": "Steel", "start_date": "2025-04-09",
            "end_date": "2025-04-11"}). Las fechas deben estar en formato YYYY-MM-DD.
        key_figures (Optional[List[str]]): Lista de campos numéricos a incluir
            (e.g., ["uptime", "temperature"]).

    Returns:
        str: Cadena JSON con el estado, conteo de registros, datos procesados, y mensaje de error si aplica.
            Ejemplo:
            {
                "status": "success",
                "count": 5,
                "data": [
                    {"id": 1, "date": "2025-04-10", "machine": "ModelA", "temperature": 75.0},
                    ...
                ]
            }

    Raises:
        Exception: Si falla la validación, la solicitud a la API, o el almacenamiento en Qdrant.
                  Devuelve un JSON con status="error" y el mensaje de error.

    Ejemplo de uso:
        await fetch_mes_data(
            ctx,
            key_values={"machine": "ModelA", "start_date": "2025-04-09", "end_date": "2025-04-11"},
            key_figures=["temperature"]
        )
        # Retorna datos filtrados para ModelA entre las fechas especificadas.
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validate requested fields and dates
        await DataValidator.validate_fields(ctx, key_figures, key_values)
        
        # Generate a unique hash for the query
        query_hash = hashlib.md5(json.dumps({"key_values": key_values, "key_figures": key_figures}, sort_keys=True).encode()).hexdigest()
        logger.info(f"Procesando consulta con query_hash: {query_hash}")
        
        # Check Qdrant for existing data
        filter_conditions = []
        for k, v in key_values.items():
            if k not in ["start_date", "end_date"]:
                filter_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        
        # Handle date range filtering
        date_filters = []
        if "start_date" in key_values and "end_date" in key_values:
            start_date = datetime.strptime(key_values["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(key_values["end_date"], "%Y-%m-%d")
            current_date = start_date
            while current_date <= end_date:
                date_filters.append(models.FieldCondition(
                    key="date",
                    match=models.MatchValue(value=current_date.strftime("%Y-%m-%d"))
                ))
                current_date += timedelta(days=1)
        
        if date_filters:
            filter_conditions.append(models.Filter(should=date_filters))
        
        existing = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=filter_conditions),
            limit=1000
        )[0]
        
        if existing:
            processed_data = [point.payload for point in existing if all(f in point.payload for f in key_figures)]
            if processed_data:
                logger.info(f"Datos recuperados de Qdrant para query_hash: {query_hash}")
                return json.dumps({
                    "status": "success",
                    "count": len(processed_data),
                    "data": processed_data,
                    "source": "qdrant_cache"
                }, ensure_ascii=False)

        # Build query parameters for API
        params = {}
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

        # Fetch data from API
        response = await auth_client.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

            # Filter out date-related keys for data filtering
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}

        # Store in vector database
        if processed_data:
            points = [
                models.PointStruct(
                    id=hashlib.md5(json.dumps(r).encode()).hexdigest(),
                    vector=model.encode(json.dumps(r)).tolist(),
                    payload=r
                ) for r in processed_data
            ]
            qdrant_client.upsert(collection_name="mes_logs", points=points)
            logger.info(f"Datos almacenados en Qdrant para query_hash: {query_hash}")

        return json.dumps({
            "status": "success",
            "count": len(processed_data),
            "data": processed_data,
            "source": "api"
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
    """Carga y procesa un documento SOP (PDF) para una máquina específica desde MinIO.

    Esta herramienta realiza las siguientes acciones:
    1. Verifica si las reglas SOP están en caché o Qdrant.
    2. Si no existen, carga el PDF desde MinIO (nombre: <machine>.pdf).
    3. Extrae el texto del PDF usando pdfplumber.
    4. Identifica reglas de cumplimiento usando patrones dinámicos para cualquier campo numérico.
    5. Almacena el SOP y sus reglas en Qdrant y caché con un embedding del texto.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        machine (str): Nombre de la máquina para la cual cargar el SOP (e.g., "ModelA").

    Returns:
        str: Cadena JSON con el estado, máquina, reglas extraídas, y mensaje de error si aplica.
            Ejemplo:
            {
                "status": "success",
                "machine": "ModelA",
                "rules": {
                    "uptime": {"value": 95.0, "operator": ">=", "unit": "%", "source_text": "uptime >= 95.0%"},
                    "temperature": {"value": 80.0, "operator": "<=", "unit": "°C", "source_text": "temperature <= 80.0°C"}
                }
            }

    Raises:
        Exception: Si falla la carga del PDF, la extracción de reglas, o el almacenamiento en Qdrant.
                  Devuelve un JSON con status="error" y el mensaje de error.

    Ejemplo de uso:
        await load_sop(ctx, machine="ModelA")
        # Carga el SOP para ModelA y retorna las reglas extraídas.
    """
    try:
        # Check in-memory cache
        if machine in sop_cache:
            logger.info(f"SOP recuperado de caché para máquina: {machine}")
            return json.dumps({
                "status": "cached",
                "machine": machine,
                "rules": sop_cache[machine]
            }, ensure_ascii=False)

        # Check Qdrant
        existing = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]),
            limit=1
        )[0]
        
        if existing:
            rules = existing[0].payload["rules"]
            sop_cache[machine] = rules
            logger.info(f"SOP recuperado de Qdrant para máquina: {machine}")
            return json.dumps({
                "status": "exists",
                "machine": machine,
                "rules": rules
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

        # Extract compliance rules using dynamic patterns
        rules = {}
        generic_pattern = r"(?P<field>[a-zA-Z_]+)\s*(?P<operator>>=|≤|>=|<=)\s*(?P<value>\d+\.?\d*)\s*(?P<unit>[%°Cmm/s]?)"
        patterns = [
            (generic_pattern, None, None),
            (r"(?P<field>uptime|tiempo de actividad)\s*(?P<operator>>=|≥)\s*(?P<value>\d+\.\d+)\s*%", ">=", "%"),
            (r"(?P<field>temperature|temperatura)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*°C", "<=", "°C"),
            (r"(?P<field>vibration|vibración)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*mm/s", "<=", "mm/s"),
            (r"(?P<field>defects|defectos)\s*(?P<operator><=|≤)\s*(?P<value>\d+)", "<=", "")
        ]

        for pattern, default_op, default_unit in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    field = match.group("field").lower().replace(" ", "_")
                    operator = match.group("operator") if match.group("operator") else default_op
                    value = float(match.group("value"))
                    unit = match.group("unit") if "unit" in match.groupdict() else default_unit or ""
                    rules[field] = {
                        "value": value,
                        "operator": operator,
                        "unit": unit,
                        "source_text": match.group(0)
                    }
                    logger.info(f"Regla extraída para {field}: {rules[field]}")
                except Exception as e:
                    logger.warning(f"Fallo al procesar regla: {match.group(0)}, error: {str(e)}")
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

        # Store in cache
        sop_cache[machine] = rules
        logger.info(f"SOP procesado y almacenado para máquina: {machine}")

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
async def create_expert_rule(
    ctx: Context, 
    prompt: str, 
    key_figure: Optional[str] = None, 
    machine: Optional[str] = None, 
    operator: Optional[str] = None, 
    threshold: Optional[float] = None, 
    unit: Optional[str] = None
) -> str:
    """Crea y almacena una regla experta basada en un prompt del usuario y parámetros opcionales en Qdrant.

    Esta herramienta realiza las siguientes acciones:
    1. Usa parámetros explícitos (key_figure, machine, operator, threshold, unit) si se proporcionan, con prioridad sobre los valores extraídos del prompt.
    2. Parsea el prompt para extraer la máquina, campo, operador, valor, unidad y condición (si aplica) si no se proporcionan explícitamente.
    3. Valida que el key_figure sea un campo numérico válido y que la máquina sea válida usando list_fields.
    4. Valida que el operador sea '>=', '<=', '≥', '≤' (mapeando desde términos naturales como 'no superar').
    5. Almacena la regla en Qdrant en la colección 'expert_rules' con un embedding del prompt.
    6. Almacena la regla en caché para acceso rápido.
    7. Devuelve un JSON con el estado y la regla creada.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        prompt (str): Prompt del usuario que describe la regla experta
            (e.g., "la máquina ModelA no debe superar los ≤ 50°C en la temperatura").
        key_figure (Optional[str]): Campo numérico para la regla (e.g., 'temperature').
            Si no se proporciona, se extrae del prompt.
        machine (Optional[str]): Nombre de la máquina (e.g., 'ModelA').
            Si no se proporciona, se extrae del prompt.
        operator (Optional[str]): Operador lógico para la regla (e.g., 'no superar', 'menos de', 'mayor a').
            Si no se proporciona, se extrae del prompt.
        threshold (Optional[float]): Valor límite para la regla (e.g., 50 para 50°C).
            Si no se proporciona, se extrae del prompt.
        unit (Optional[str]): Unidad de medida para el valor (e.g., 'grados Celsius', '%', 'mm/s').
            Si no se proporciona, se extrae del prompt o se asigna por defecto.

    Returns:
        str: Cadena JSON con el estado, máquina, regla creada, y mensaje de error si aplica.
            Ejemplo:
            {
                "status": "success",
                "machine": "ModelA",
                "rule": {
                    "field": "temperature",
                    "value": 50.0,
                    "operator": "<=",
                    "unit": "°C",
                    "condition": "single",
                    "prompt": "la máquina ModelA no debe superar los ≤ 50°C en la temperatura",
                    "timestamp": "2025-05-19T19:40:00.123456"
                }
            }

    Raises:
        Exception: Si falla el parseo del prompt, la validación, o el almacenamiento en Qdrant.
                  Devuelve un JSON con status="error" y el mensaje de error.

    Ejemplo de uso:
        await create_expert_rule(
            ctx,
            prompt="la máquina ModelA no debe superar los ≤ 50°C en la temperatura",
            key_figure="temperature",
            machine="ModelA",
            operator="no superar",
            threshold=50.0,
            unit="grados Celsius"
        )
        # Crea y almacena una regla experta para ModelA con temperatura <= 50°C.
    """
    try:
        # Get available fields
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": "Could not validate fields against API"
            }, ensure_ascii=False)

        # Initialize rule dictionary
        rule = {
            "condition": "single",  # Default
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }

        # Handle machine
        prompt_lower = prompt.lower()
        if machine:
            if machine not in fields_info["key_values"].get("machine", []):
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid machine: {machine}. Available machines: {fields_info['key_values'].get('machine', [])}"
                }, ensure_ascii=False)
            rule["machine"] = machine
        else:
            machine_match = re.search(r"máquina\s+([a-zA-Z0-9_-]+)", prompt_lower)
            if not machine_match:
                return json.dumps({
                    "status": "error",
                    "message": "Machine name not found in prompt. Specify 'máquina <name>' or provide 'machine' parameter"
                }, ensure_ascii=False)
            rule["machine"] = machine_match.group(1)
            if rule["machine"] not in fields_info["key_values"].get("machine", []):
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid machine: {rule['machine']}. Available machines: {fields_info['key_values'].get('machine', [])}"
                }, ensure_ascii=False)

        # Handle key_figure
        field_synonyms = {
            "temperature": ["temperatura", "temp"],
            "uptime": ["tiempo de actividad"],
            "vibration": ["vibración", "vibracion"],
            "defects": ["defectos"],
            "throughput": ["rendimiento"],
            "inventory_level": ["nivel de inventario"]
        }
        if key_figure:
            if key_figure not in fields_info["key_figures"]:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid key_figure: {key_figure}. Available key_figures: {fields_info['key_figures']}"
                }, ensure_ascii=False)
            rule["field"] = key_figure
        else:
            for field, synonyms in field_synonyms.items():
                if field in prompt_lower or any(syn in prompt_lower for syn in synonyms):
                    rule["field"] = field
                    break
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Key_figure not found in prompt. Available key_figures: {fields_info['key_figures']}. Specify 'key_figure' parameter or include field in prompt"
                }, ensure_ascii=False)

        # Operator mapping
        operator_map = {
            "no superar": "<=",
            "menos de": "<=",
            "menor a": "<=",
            "máximo": "<=",
            "mayor a": ">=",
            "mínimo": ">=",
            "mayor de": ">="
        }

        # Use provided operator or extract from prompt
        if operator:
            operator = operator.lower()
            if operator not in operator_map:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid operator: {operator}. Valid operators: {list(operator_map.keys())}"
                }, ensure_ascii=False)
            rule["operator"] = operator_map[operator]
        else:
            for op, mapped_op in operator_map.items():
                if op in prompt_lower:
                    rule["operator"] = mapped_op
                    break
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Operator not specified in prompt (e.g., 'no superar', 'mayor a')"
                }, ensure_ascii=False)

        # Use provided threshold or extract from prompt
        if threshold is not None:
            try:
                rule["value"] = float(threshold)
            except ValueError:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid threshold: {threshold}. Must be a number"
                }, ensure_ascii=False)
        else:
            value_match = re.search(r"(\d+\.?\d*)\s*(°[CF]|%|mm/s)?", prompt_lower)
            if not value_match:
                return json.dumps({
                    "status": "error",
                    "message": "Value not found in prompt (e.g., '50°C', '90%')"
                }, ensure_ascii=False)
            rule["value"] = float(value_match.group(1))

        # Use provided unit or extract from prompt
        unit_map = {
            "grados celsius": "°C",
            "grados fahrenheit": "°F",
            "porcentaje": "%",
            "mm/s": "mm/s",
            "%": "%",
            "°c": "°C",
            "°f": "°F"
        }
        if unit:
            unit = unit.lower()
            if unit in unit_map:
                rule["unit"] = unit_map[unit]
            else:
                rule["unit"] = unit  # Allow custom units if valid
        else:
            value_match = re.search(r"(\d+\.?\d*)\s*(°[CF]|%|mm/s)?", prompt_lower)
            rule["unit"] = value_match.group(2) if value_match and value_match.group(2) else ""
            if not rule["unit"] and rule["field"] == "temperature":
                rule["unit"] = "°C"  # Default for temperature

        # Extract condition
        if "ambas máquinas" in prompt_lower or "todas las máquinas" in prompt_lower:
            rule["condition"] = "both_machines"

        # Validate rule
        if rule["field"] not in fields_info["key_figures"]:
            return json.dumps({
                "status": "error",
                "message": f"Invalid field: {rule['field']}. Available fields: {fields_info['key_figures']}"
            }, ensure_ascii=False)
        if rule["operator"] not in [">=", "≤", "<=", "≥"]:
            return json.dumps({
                "status": "error",
                "message": f"Invalid operator: {rule['operator']}. Use '>=', '<='"
            }, ensure_ascii=False)

        # Validate unit consistency
        expected_units = {
            "temperature": ["°C", "°F"],
            "uptime": ["%"],
            "vibration": ["mm/s"],
            "defects": [""],
            "throughput": [""],
            "inventory_level": [""]
        }
        if rule["field"] in expected_units and rule["unit"] and rule["unit"] not in expected_units[rule["field"]]:
            return json.dumps({
                "status": "error",
                "message": f"Invalid unit for {rule['field']}: {rule['unit']}. Expected: {expected_units[rule['field']]}"
            }, ensure_ascii=False)

        # Store in Qdrant
        rule_id = hashlib.md5(f"{rule['machine']}{rule['field']}{rule['timestamp']}".encode()).hexdigest()
        embedding = model.encode(prompt).tolist()
        qdrant_client.upsert(
            collection_name="expert_rules",
            points=[models.PointStruct(
                id=rule_id,
                vector=embedding,
                payload=rule
            )]
        )

        # Store in cache
        if rule["machine"] not in expert_rules_cache:
            expert_rules_cache[rule["machine"]] = {}
        expert_rules_cache[rule["machine"]][rule["field"]] = rule
        logger.info(f"Regla experta creada para {rule['machine']}:{rule['field']}: {rule}")

        return json.dumps({
            "status": "success",
            "machine": rule["machine"],
            "rule": rule
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Expert rule creation failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        }, ensure_ascii=False)

@mcp.tool()
async def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None
) -> str:
    """Analiza el cumplimiento de los datos MES contra reglas SOP y reglas expertas.

    Esta herramienta realiza las siguientes acciones:
    1. Valida los campos solicitados (key_figures y key_values) usando DataValidator.
    2. Obtiene datos de la API MES o Qdrant usando fetch_mes_data, aplicando filtros dinámicos.
    3. Carga las reglas SOP para las máquinas relevantes usando load_sop.
    4. Consulta reglas expertas en Qdrant para las máquinas relevantes.
    5. Compara cada registro contra las reglas SOP y expertas, determinando el estado de cumplimiento.
    6. Calcula porcentajes de cumplimiento separados para SOP y reglas expertas.
    7. Devuelve un informe detallado con los resultados del análisis.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos y valores
            para filtrar (e.g., {"machine": "ModelA", "material": "Steel", "production_line": "Line1",
            "start_date": "2025-04-09", "end_date": "2025-04-11"}). Las fechas deben estar en formato YYYY-MM-DD.
        key_figures (Optional[List[str]]): Lista de campos numéricos a analizar
            (e.g., ["temperature", "uptime"]).

    Returns:
        str: Cadena JSON con el estado, período, filtro de máquina, métricas analizadas,
            resultados del análisis (SOP y expertas), cobertura de SOP, y notas.
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
                        "sop_compliance": {
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
                        "expert_compliance": {
                            "temperature": {
                                "value": 75.0,
                                "rule": "<= 50.0°C (single)",
                                "status": "non_compliant"
                            }
                        },
                        "sop_compliance_percentage": 100.0,
                        "expert_compliance_percentage": 0.0
                    }
                ],
                "sop_coverage": "1/1 machines with SOP",
                "custom_rules_applied": "1/1 machines with custom rules",
                "analysis_notes": [
                    "compliant: Meets requirement",
                    "non_compliant: Does not meet requirement",
                    "unknown: No rule defined"
                ]
            }

    Raises:
        Exception: Si falla la validación, la obtención de datos, o el análisis.
                  Devuelve un JSON con status="error" y el mensaje de error.

    Ejemplo de uso:
        await analyze_compliance(
            ctx,
            key_values={"machine": "ModelA", "start_date": "2025-04-09", "end_date": "2025-04-11"},
            key_figures=["temperature"]
        )
        # Analiza el cumplimiento de los datos de ModelA, incluyendo reglas SOP y expertas.
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
            machine_rules[machine] = sop_result.get("rules", {}) if sop_result["status"] in ["success", "exists", "cached"] else {}
            logger.info(f"Reglas SOP cargadas para máquina {machine}: {machine_rules[machine]}")

        # Load expert rules from Qdrant or cache
        expert_rules = {}
        for machine in machines:
            if machine in expert_rules_cache:
                expert_rules[machine] = expert_rules_cache[machine]
                logger.info(f"Reglas expertas recuperadas de caché para máquina {machine}")
                continue

            existing = qdrant_client.scroll(
                collection_name="expert_rules",
                scroll_filter=models.Filter(must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]),
                limit=100
            )[0]
            
            expert_rules[machine] = {}
            if existing:
                # Use the most recent rule for each field
                for point in sorted(existing, key=lambda x: x.payload["timestamp"], reverse=True):
                    field = point.payload["field"]
                    if field not in expert_rules[machine]:
                        expert_rules[machine][field] = point.payload
                expert_rules_cache[machine] = expert_rules[machine]
                logger.info(f"Reglas expertas cargadas de Qdrant para máquina {machine}: {expert_rules[machine]}")

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
                "sop_compliance": {},
                "expert_compliance": {},
                "sop_compliance_percentage": 0.0,
                "expert_compliance_percentage": 0.0
            }
            
            sop_compliant = 0
            sop_total = 0
            expert_compliant = 0
            expert_total = 0
            
            for field in key_figures:
                if field not in record:
                    continue
                
                # SOP compliance
                sop_total += 1
                rules = machine_rules.get(record["machine"], {})
                
                # Check SOP rule
                sop_rules = machine_rules.get(record["machine"], {})
                if field in sop_rules:
                    rule = sop_rules[field]
                    value = record[field]
                    op = rule["operator"]
                    
                    is_compliant = (value >= rule["value"]) if op in [">=", "≥"] else (value <= rule["value"])
                    analysis["sop_compliance"][field] = {
                        "value": value,
                        "rule": f"{op} {rule['value']}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }
                    sop_compliant += 1 if is_compliant else 0
                else:
                    analysis["sop_compliance"][field] = {
                        "value": record[field],
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }

                # Expert compliance
                expert_rules_for_machine = expert_rules.get(record["machine"], {})
                if field in expert_rules_for_machine:
                    expert_total += 1
                    rule = expert_rules_for_machine[field]
                    value = record[field]
                    op = rule["operator"]
                    
                    is_compliant = (value >= rule["value"]) if op in [">=", "≥"] else (value <= rule["value"])
                    
                    # Check combined condition (e.g., both machines)
                    if rule.get("condition") == "both_machines":
                        other_machines = [r for r in fetch_result["data"] if r["machine"] != record["machine"] and field in r]
                        if other_machines:
                            all_compliant = all(
                                (r[field] >= rule["value"] if op in [">=", "≥"] else r[field] <= rule["value"])
                                for r in other_machines
                            )
                            is_compliant = is_compliant and all_compliant
                        
                    analysis["expert_compliance"][field] = {
                        "value": value,
                        "rule": f"{op} {rule['value']}{rule.get('unit', '')} ({rule.get('condition', 'single')})",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }
                    expert_compliant += 1 if is_compliant else 0
                else:
                    analysis["expert_compliance"][field] = {
                        "value": record[field],
                        "rule": "no_expert_rule_defined",
                        "status": "unknown"
                    }
            
            if sop_total > 0:
                analysis["sop_compliance_percentage"] = round((sop_compliant / sop_total) * 100, 2)
            if expert_total > 0:
                analysis["expert_compliance_percentage"] = round((expert_compliant / expert_total) * 100, 2)
            
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
                "compliant: Meets requirement",
                "non_compliant: Does not meet requirement",
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