import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import logging
import json
import hashlib
import re
<<<<<<< HEAD
=======
import os
>>>>>>> dev
from minio import Minio
from minio.error import S3Error
import pdfplumber
import io
<<<<<<< HEAD
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
=======
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from cryptography.fernet import Fernet, InvalidToken
import inspect
>>>>>>> dev

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del MCP
mcp = FastMCP("Manufacturing Compliance Processor")
API_URL = os.getenv("API_URL", "http://api:5000")
TOKEN_API_URL = os.getenv("TOKEN_API_URL", "http://token-api:5001")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sop-pdfs")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    logger.warning("ENCRYPTION_KEY not set, generating temporary key (not suitable for production)")
    ENCRYPTION_KEY = Fernet.generate_key().decode()
fernet = Fernet(ENCRYPTION_KEY.encode())

# Inicialización de clientes
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="qdrant", port=6333)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Caché para el campo de fecha
DATE_FIELD_CACHE = {"field": None, "schema_hash": None}

# Formatos de fecha soportados
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y%m%d",
    "%Y/%m/%d", "%b %d, %Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M", "%Y/%m/%d %H:%M:%S"
]

def detect_and_normalize_date(date_str: str) -> Optional[str]:
    """
    Detecta y normaliza una cadena de fecha al formato YYYY-MM-DD.
    """
    if not isinstance(date_str, str) or not date_str.strip():
        return None
<<<<<<< HEAD
    # Filtrar valores claramente no relacionados con fechas
=======
>>>>>>> dev
    if any(keyword in date_str.lower() for keyword in ["model", "line", "batch", "aluminum", "steel", "plastic", "copper", "brass", "titanium", "chip", "crack", "dent", "scratch", "warp", "none"]):
        return None
    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    match = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2}(?::\d{2})?)?", date_str)
    if match:
        return match.group(1)
    logger.warning(f"No se pudo parsear la fecha: {date_str}")
    return None

def find_date_field(records: List[Dict], fields_info: Dict) -> Optional[str]:
    """
    Detecta el campo que contiene fechas en los registros.
<<<<<<< HEAD
    Usa el campo cacheado si está disponible y el esquema no ha cambiado.
=======
>>>>>>> dev
    """
    global DATE_FIELD_CACHE
    if not records or not fields_info:
        logger.warning("No records or fields_info provided for date detection")
        return None

<<<<<<< HEAD
    # Generar hash del esquema
    schema = json.dumps(fields_info, sort_keys=True)
    schema_hash = hashlib.md5(schema.encode()).hexdigest()

    # Usar caché si es válido
=======
    schema = json.dumps(fields_info, sort_keys=True)
    schema_hash = hashlib.md5(schema.encode()).hexdigest()

>>>>>>> dev
    if DATE_FIELD_CACHE["field"] and DATE_FIELD_CACHE["schema_hash"] == schema_hash:
        logger.info(f"Using cached date field: {DATE_FIELD_CACHE['field']}")
        return DATE_FIELD_CACHE["field"]

    key_values = fields_info.get("key_values", {})
    if not key_values:
        logger.warning("No categorical fields available in fields_info")
        return None

    best_candidate = None
    best_score = 0
<<<<<<< HEAD
    # Limitar a 10 registros
    for field in key_values.keys():
        # Excluir campos no relacionados con fechas
=======
    for field in key_values.keys():
>>>>>>> dev
        if any(keyword in field.lower() for keyword in ["machine", "production_line", "material", "batch_id", "defect_type"]):
            continue
        sample_values = [r.get(field) for r in records[:10] if field in r]
        if not sample_values:
            continue
        valid_dates = [detect_and_normalize_date(str(v)) for v in sample_values]
        valid_count = sum(1 for d in valid_dates if d)
        valid_ratio = valid_count / len(sample_values) if sample_values else 0
        score = valid_ratio
        if any(keyword in field.lower() for keyword in ["date", "time", "created", "updated", "timestamp"]):
            score += 0.2
        if score > best_score:
            best_score = score
            best_candidate = field

    if best_candidate and best_score >= 0.5:
        logger.info(f"Date field detected: {best_candidate} (score: {best_score})")
        DATE_FIELD_CACHE["field"] = best_candidate
        DATE_FIELD_CACHE["schema_hash"] = schema_hash
        return best_candidate
    else:
        logger.warning("No date field detected in the provided records")
        return None

<<<<<<< HEAD
def check_date_coverage(data: List[Dict], start_date: str, end_date: str) -> Dict:
    """
    Verifica la cobertura de fechas en los datos.
    """
    if not data:
        return {
            "has_data": False,
            "covered_dates": [],
            "message": f"No se encontraron registros en el rango de fechas del {start_date} al {end_date}."
        }
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    expected_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
    covered_dates = sorted(set(r["date"] for r in data if r.get("date") and r["date"] != "Desconocida"))
    if not covered_dates:
        return {
            "has_data": False,
            "covered_dates": [],
            "message": f"No se encontraron registros con fechas válidas en el rango del {start_date} al {end_date}."
        }
    has_start = start_date in covered_dates
    has_end = end_date in covered_dates
    missing_dates = [d for d in expected_dates if d not in covered_dates]
    if not missing_dates:
        message = "Se encontraron registros para todas las fechas en el rango solicitado."
    elif has_start and not has_end:
        message = f"Se encontraron registros para {start_date}, pero no para {end_date} ni fechas posteriores en el rango."
    elif covered_dates:
        message = f"Solo se encontraron datos para las fechas {', '.join(covered_dates)} dentro del rango solicitado."
    else:
        message = f"No se encontraron registros en el rango de fechas del {start_date} al {end_date}."
=======
def check_date_coverage(data: List[Dict], start_date: Optional[str], end_date: Optional[str], specific_dates: Optional[List[str]] = None) -> Dict:
    """
    Verifica la cobertura de fechas en los datos, considerando fechas específicas o rango de fechas.
    """
    if not data:
        message = "No se encontraron registros."
        if specific_dates:
            message = f"No se encontraron registros para las fechas específicas: {', '.join(specific_dates)}."
        elif start_date and end_date:
            message = f"No se encontraron registros en el rango de fechas del {start_date} al {end_date}."
        return {
            "has_data": False,
            "covered_dates": [],
            "message": message
        }

    covered_dates = sorted(set(r["date"] for r in data if r.get("date") and r["date"] != "Desconocida"))

    if specific_dates:
        expected_dates = [detect_and_normalize_date(d) for d in specific_dates if detect_and_normalize_date(d)]
        if not expected_dates:
            return {
                "has_data": False,
                "covered_dates": [],
                "message": "No se proporcionaron fechas válidas en specific_dates."
            }
        missing_dates = [d for d in expected_dates if d not in covered_dates]
        message = f"Datos encontrados para {len(covered_dates)} de {len(expected_dates)} fechas solicitadas."
        if missing_dates:
            message += f" Fechas faltantes: {', '.join(missing_dates)}."
    elif start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        expected_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
        missing_dates = [d for d in expected_dates if d not in covered_dates]
        if not missing_dates:
            message = "Se encontraron registros para todas las fechas en el rango solicitado."
        elif covered_dates:
            message = f"Solo se encontraron datos para las fechas {', '.join(covered_dates)} dentro del rango solicitado."
        else:
            message = f"No se encontraron registros en el rango de fechas del {start_date} al {end_date}."
    else:
        message = "Datos recuperados exitosamente sin filtros de fecha."

>>>>>>> dev
    return {
        "has_data": bool(covered_dates),
        "covered_dates": covered_dates,
        "message": message
    }

class AuthClient:
    """
<<<<<<< HEAD
    Cliente para autenticación con la API.
    """
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
=======
    Cliente para autenticación con la API principal y la API de tokens.
    """
    def __init__(self, api_url: str, token_api_url: str):
        self.api_url = api_url
        self.token_api_url = token_api_url
>>>>>>> dev
        self.client = httpx.Client()
        self.token = None

    def fetch_token(self):
        """Obtiene un token JWT de la API de tokens."""
        try:
            response = self.client.get(f"{self.token_api_url}/get-token")
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            logger.info("Token fetched from token-api")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch token from token-api: {e.response.status_code} {e.response.text}")
            raise ValueError(f"No se pudo obtener token: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error fetching token: {str(e)}")
            raise ValueError(f"Error al obtener token: {str(e)}")

    def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        """Realiza una solicitud GET a la API principal, obteniendo un token si es necesario."""
        if not self.token:
<<<<<<< HEAD
            self.authenticate()
=======
            self.fetch_token()
>>>>>>> dev
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.get(
                f"{self.api_url}{endpoint}",
                headers=headers,
                params=params
            )
            if response.status_code == 401:
                logger.info("Token inválido, obteniendo nuevo token...")
                self.token = None
                self.fetch_token()
                headers = {"Authorization": f"Bearer {self.token}"}
                response = self.client.get(
                    f"{self.api_url}{endpoint}",
                    headers=headers,
                    params=params
                )
<<<<<<< HEAD
            logger.info(f"HTTP Request: GET {self.base_url}{endpoint} \"HTTP/1.1 {response.status_code}\"")
=======
            logger.info(f"HTTP Request: GET {self.api_url}{endpoint} \"HTTP/1.1 {response.status_code}\"")
>>>>>>> dev
            return response
        except Exception as e:
            logger.error(f"Error en solicitud GET: {str(e)}")
            raise

    def post(self, endpoint: str, json_data: Optional[Dict] = None) -> httpx.Response:
<<<<<<< HEAD
        if not self.token:
            self.authenticate()
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.post(
                f"{self.base_url}{endpoint}",
=======
        """Realiza una solicitud POST a la API principal, obteniendo un token si es necesario."""
        if not self.token:
            self.fetch_token()
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.post(
                f"{self.api_url}{endpoint}",
>>>>>>> dev
                headers=headers,
                json=json_data
            )
            if response.status_code == 401:
<<<<<<< HEAD
                logger.info("Token inválido, reautenticando...")
                self.authenticate()
                headers = {"Authorization": f"Bearer {self.token}"}
                response = self.client.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    json=json_data
                )
            logger.info(f"HTTP Request: POST {self.base_url}{endpoint} \"HTTP/1.1 {response.status_code}\"")
=======
                logger.info("Token inválido, obteniendo nuevo token...")
                self.token = None
                self.fetch_token()
                headers = {"Authorization": f"Bearer {self.token}"}
                response = self.client.post(
                    f"{self.api_url}{endpoint}",
                    headers=headers,
                    json=json_data
                )
            logger.info(f"HTTP Request: POST {self.api_url}{endpoint} \"HTTP/1.1 {response.status_code}\"")
>>>>>>> dev
            return response
        except Exception as e:
            logger.error(f"Error en solicitud POST: {str(e)}")
            raise

    def close(self):
        self.client.close()

auth_client = None

def init_infrastructure():
    """
    Inicializa la infraestructura (Qdrant, MinIO, AuthClient).
    """
    global auth_client
    try:
        vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
        qdrant_client.recreate_collection(collection_name="mes_logs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="sop_pdfs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="custom_rules", vectors_config=vector_config)
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
        auth_client = AuthClient(API_URL, TOKEN_API_URL)
    except Exception as e:
        logger.error(f"Infrastructure initialization failed: {str(e)}")
        raise

class DataValidator:
    """
    Valida campos y fechas.
    """
    @staticmethod
    def validate_date(date_str: str, field: str) -> str:
        normalized_date = detect_and_normalize_date(date_str)
        if normalized_date is None:
            raise ValueError(f"Formato inválido para {field}: {date_str}. Formatos soportados: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, etc.")
        return normalized_date

    @staticmethod
<<<<<<< HEAD
    def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict) -> Dict:
        try:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                raise ValueError("No se pudo validar contra la API")
            if "start_date" in key_values or "end_date" in key_values:
                if "start_date" not in key_values or "end_date" not in key_values:
                    raise ValueError("Se deben proporcionar tanto start_date como end_date")
                start_date = DataValidator.validate_date(key_values["start_date"], "start_date")
                end_date = DataValidator.validate_date(key_values["end_date"], "end_date")
                if start_date > end_date:
                    raise ValueError("start_date no puede ser posterior a end_date")
                key_values["start_date"] = start_date
                key_values["end_date"] = end_date
=======
    def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> Dict:
        try:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.error("Failed to validate fields against API")
                raise ValueError("No se pudo validar contra la API")
            if (start_date and not end_date) or (end_date and not start_date):
                raise ValueError("Se deben proporcionar tanto start_date como end_date")
            if start_date and end_date and specific_dates:
                raise ValueError("No se pueden usar start_date/end_date y specific_dates al mismo tiempo")
            if start_date and end_date:
                start_date = DataValidator.validate_date(start_date, "start_date")
                end_date = DataValidator.validate_date(end_date, "end_date")
                if start_date > end_date:
                    raise ValueError("start_date no puede ser posterior a end_date")
            if specific_dates:
                specific_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
                if not specific_dates:
                    raise ValueError("No se proporcionaron fechas válidas en specific_dates")
>>>>>>> dev
            errors = []
            invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
            if invalid_figures:
                valid_figures = fields_info["key_figures"]
                errors.append(f"Campos numéricos inválidos: {invalid_figures}. Campos disponibles: {', '.join(valid_figures)}.")
            invalid_values = {}
            for k, v in key_values.items():
<<<<<<< HEAD
                if k in ["start_date", "end_date"]:
                    continue
=======
>>>>>>> dev
                if k not in fields_info["key_values"]:
                    invalid_values[k] = v
                    errors.append(f"Campo categórico inválido: '{k}'. Campos disponibles: {', '.join(fields_info['key_values'].keys())}.")
                elif v not in fields_info["key_values"].get(k, []):
                    invalid_values[k] = v
                    valid_vals = fields_info["key_values"].get(k, [])
                    errors.append(f"Valor inválido para '{k}': '{v}'. Valores válidos: {', '.join(valid_vals)}.")
            if errors:
                raise ValueError(" | ".join(errors))
            return fields_info
        except Exception as e:
            logger.error(f"Fallo en validación de campos: {str(e)}")
            raise

<<<<<<< HEAD
=======
def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Recupera datos MES de la API y Qdrant, encriptando payloads antes de almacenarlos.
    """
    try:
        key_values = key_values or {}
        # Si no se proporcionan key_figures, obtener todos los campos numéricos desde list_fields
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.error("No se pudieron obtener campos válidos")
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        valid_figures = fields_info["key_figures"]
        valid_values = fields_info["key_values"]
        logger.info(f"Fetching MES data for key_values={key_values}, key_figures={key_figures}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
        must_conditions = []
        for k, v in key_values.items():
            if k in valid_values:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if specific_dates:
            normalized_dates = [detect_and_normalize_date(d) for d in specific_dates if detect_and_normalize_date(d)]
            if not normalized_dates:
                return json.dumps({
                    "status": "error",
                    "message": "No se proporcionaron fechas válidas en specific_dates",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchAny(any=normalized_dates)
            ))
        elif start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                    must_conditions.append(models.FieldCondition(
                        key="date",
                        match=models.MatchAny(any=date_range)
                    ))
            except ValueError as e:
                logger.error(f"Invalid date format: {str(e)}")
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid date format: {str(e)}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
        # Recuperar datos de Qdrant y desencriptar payloads
        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = []
        for r in qdrant_results[0] if qdrant_results[0] else []:
            try:
                encrypted_payload = r.payload.get("encrypted_payload")
                if encrypted_payload:
                    decrypted_data = fernet.decrypt(encrypted_payload.encode()).decode()
                    processed_data.append(json.loads(decrypted_data))
                else:
                    logger.warning(f"No encrypted payload for point {r.id}")
            except InvalidToken:
                logger.error(f"Failed to decrypt payload for point {r.id}")
                continue
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")
        # Obtener datos frescos de la API
        params = {}
        if specific_dates:
            params.update({"specific_date": specific_dates[0]})  # API soporta specific_date
        elif start_date and end_date:
            params.update({"start_date": start_date, "end_date": end_date})
        response = auth_client.get("/machines/", params=params)
        response.raise_for_status()
        all_data = response.json()
        date_field = find_date_field(all_data, fields_info)
        logger.info(f"Detected date field: {date_field}")
        full_data = []
        for record in all_data:
            item = {}
            if date_field and date_field in record:
                normalized_date = detect_and_normalize_date(str(record[date_field]))
                item["date"] = normalized_date or "Desconocida"
            else:
                item["date"] = "Desconocida"
            for field in valid_figures + list(valid_values.keys()):
                if field in record:
                    item[field] = record[field]
            full_data.append(item)
        if full_data and specific_dates:
            normalized_dates = [detect_and_normalize_date(d) for d in specific_dates if detect_and_normalize_date(d)]
            full_data = [r for r in full_data if r.get("date") in normalized_dates]
        elif full_data and start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    for i, record in enumerate(full_data):
                        if record["date"] == "Desconocida":
                            record["date"] = (start + timedelta(days=i % delta)).strftime("%Y-%m-%d")
            except ValueError:
                pass
        # Encriptar y almacenar en Qdrant
        if full_data:
            points = []
            for r in full_data:
                payload_json = json.dumps(r)
                encrypted_payload = fernet.encrypt(payload_json.encode()).decode()
                point = models.PointStruct(
                    id=hashlib.md5(payload_json.encode()).hexdigest(),
                    vector=model.encode(payload_json).tolist(),
                    payload={"encrypted_payload": encrypted_payload}
                )
                points.append(point)
            qdrant_client.upsert(collection_name="mes_logs", points=points)
            logger.info(f"Stored {len(points)} encrypted points in Qdrant mes_logs")
        if not processed_data:
            data_filters = {k: v for k, v in key_values.items()}
            processed_data = [
                r for r in full_data
                if all(r.get(k) == v for k, v in data_filters.items())
            ]
            logger.info(f"Filtered {len(processed_data)} records in memory for {data_filters}")
        if key_figures:
            missing_figures = [k for k in key_figures if not any(k in r for r in processed_data)]
            if missing_figures:
                logger.warning(f"Missing key_figures in data: {missing_figures}")
                return json.dumps({
                    "status": "no_data",
                    "count": 0,
                    "data": [],
                    "message": f"No data found for fields: {', '.join(missing_figures)}.",
                    "covered_dates": []
                }, ensure_ascii=False)
        response_fields = ["date"] + list(key_values.keys()) + key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r}
            for r in processed_data
        ]
        coverage = check_date_coverage(response_data, start_date, end_date, specific_dates)
        return json.dumps({
            "status": "success" if response_data else "no_data",
            "count": len(response_data),
            "data": response_data,
            "message": coverage["message"],
            "covered_dates": coverage["covered_dates"]
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "data": [],
            "covered_dates": []
        }, ensure_ascii=False)

>>>>>>> dev
@mcp.tool()
def get_pdf_content(ctx: Context, filename: str) -> str:
    """
    Obtiene el contenido de un PDF desde MinIO.
    """
    try:
        try:
            response = minio_client.get_object(MINIO_BUCKET, filename)
            pdf_data = response.read()
            response.close()
            response.release_conn()
        except S3Error as e:
            available_pdfs = [obj.object_name for obj in minio_client.list_objects(MINIO_BUCKET)]
<<<<<<< HEAD
            logger.error(f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}")
=======
            logger.warning(f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}")
>>>>>>> dev
            return json.dumps({
                "status": "error",
                "message": f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}",
                "filename": filename,
                "content": ""
            }, ensure_ascii=False)
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return json.dumps({
            "status": "success",
            "filename": filename,
            "content": content
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error extracting content from {filename}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "filename": filename,
            "content": ""
        }, ensure_ascii=False)

@mcp.tool()
def list_fields(ctx: Context) -> str:
    """
    Lista los campos disponibles en la API MES.
    """
    try:
        response = auth_client.get("/machines/")
        response.raise_for_status()
        records = response.json()
        if not records:
            logger.warning("No se encontraron registros en el sistema MES")
            return json.dumps({
                "status": "no_data",
                "message": "No se encontraron registros en el sistema MES",
                "key_figures": [],
                "key_values": {}
            })
        sample = records[0]
        key_figures = []
        key_values = {}
        for field, value in sample.items():
            if isinstance(value, (int, float)):
                key_figures.append(field)
            elif isinstance(value, str):
                unique_values = sorted({rec[field] for rec in records if field in rec})
                key_values[field] = unique_values
        logger.info(f"Fields listed: key_figures={key_figures}, key_values={key_values}")
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
<<<<<<< HEAD
def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None
) -> str:
    """
    Recupera datos MES de la API y Qdrant.
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_figures = fields_info["key_figures"]
        valid_values = fields_info["key_values"]
        logger.info(f"Fetching MES data for key_values={key_values}, key_figures={key_figures}")
        start_date = key_values.get("start_date")
        end_date = key_values.get("end_date")
        must_conditions = []
        for k, v in key_values.items():
            if k not in ["start_date", "end_date"] and k in valid_values:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                    must_conditions.append(models.FieldCondition(
                        key="date",
                        match=models.MatchAny(any=date_range)
                    ))
            except ValueError as e:
                logger.error(f"Invalid date format: {str(e)}")
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid date format: {str(e)}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = [r.payload for r in qdrant_results[0]] if qdrant_results[0] else []
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")
        params = {}
        if start_date and end_date:
            params.update({"start_date": start_date, "end_date": end_date})
        response = auth_client.get("/machines/", params=params)
        response.raise_for_status()
        all_data = response.json()
        date_field = find_date_field(all_data, fields_info)
        logger.info(f"Detected date field: {date_field}")
        full_data = []
        for record in all_data:
            item = {}
            if date_field and date_field in record:
                normalized_date = detect_and_normalize_date(str(record[date_field]))
                item["date"] = normalized_date or "Desconocida"
            else:
                item["date"] = "Desconocida"
            for field in valid_figures + list(valid_values.keys()):
                if field in record:
                    item[field] = record[field]
            full_data.append(item)
        if full_data and start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    for i, record in enumerate(full_data):
                        if record["date"] == "Desconocida":
                            record["date"] = (start + timedelta(days=i % delta)).strftime("%Y-%m-%d")
            except ValueError:
                pass
        if full_data:
            points = [
                models.PointStruct(
                    id=hashlib.md5(json.dumps(r).encode()).hexdigest(),
                    vector=model.encode(json.dumps(r)).tolist(),
                    payload=r
                ) for r in full_data
            ]
            qdrant_client.upsert(collection_name="mes_logs", points=points)
            logger.info(f"Stored {len(points)} points in Qdrant mes_logs")
        if not processed_data:
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}
            processed_data = [
                r for r in full_data
                if all(r.get(k) == v for k, v in data_filters.items())
            ]
            logger.info(f"Filtered {len(processed_data)} records in memory for {data_filters}")
        if key_figures:
            missing_figures = [k for k in key_figures if not any(k in r for r in processed_data)]
            if missing_figures:
                logger.warning(f"Missing key_figures in data: {missing_figures}")
                return json.dumps({
                    "status": "no_data",
                    "count": 0,
                    "data": [],
                    "message": f"No data found for fields: {', '.join(missing_figures)}.",
                    "covered_dates": []
                }, ensure_ascii=False)
        response_fields = ["date"] + list(key_values.keys()) + key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r}
            for r in processed_data
        ]
        coverage = check_date_coverage(response_data, start_date, end_date) if start_date and end_date else {
            "has_data": bool(response_data),
            "covered_dates": [],
            "message": "No date range specified" if not response_data else "Data retrieved successfully"
        }
        return json.dumps({
            "status": "success" if response_data else "no_data",
            "count": len(response_data),
            "data": response_data,
            "message": coverage["message"],
            "covered_dates": coverage["covered_dates"]
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "data": [],
            "covered_dates": []
        }, ensure_ascii=False)

@mcp.tool()
=======
>>>>>>> dev
def add_custom_rule(
    ctx: Context,
    machines: Union[List[str], str],
    key_figures: Union[Dict[str, float], str],
    key_values: Optional[Dict[str, str]] = None,
    operator: str = "<=",
    unit: Optional[str] = None,
    description: str = ""
) -> str:
    """
    Añade una regla personalizada a Qdrant.
    """
    try:
        if isinstance(machines, str):
            try:
                machines = json.loads(machines)
            except json.JSONDecodeError:
                machines = [machines.strip()]
        if not isinstance(machines, list):
            raise ValueError("machines must be a list of strings")
        if isinstance(key_figures, str):
            try:
                parsed_figures = json.loads(key_figures)
                if not isinstance(parsed_figures, dict):
                    raise ValueError("key_figures JSON must be a dictionary")
                key_figures = parsed_figures
            except json.JSONDecodeError:
                parsed_figures = {}
                for pair in key_figures.split(','):
                    if '=' in pair:
                        field, value = pair.split('=', 1)
                    elif ':' in pair:
                        field, value = pair.split(':', 1)
                    else:
                        raise ValueError(f"Invalid format: {pair}: Use 'key=value' or 'key:value'")
                    field = field.strip()
                    try:
                        parsed_figures[field] = float(value.strip())
                    except ValueError:
                        raise ValueError(f"Invalid value for {field}: must be numeric")
                key_figures = parsed_figures
        elif isinstance(key_figures, dict):
            parsed_figures = {k: float(v) for k, v in key_figures.items()}
            key_figures = parsed_figures
        else:
            raise ValueError("key_figures must be dict or string")
        if not machines:
            raise ValueError("At least one machine must be specified")
        if not key_figures:
            raise ValueError("At least one metric must be specified")
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Could not validate against API")
        valid_machines = fields_info["key_values"].get("machine", [])
        invalid_machines = [m for m in machines if m not in valid_machines]
        if invalid_machines:
            raise ValueError(f"Invalid machines: {invalid_machines}. Valid machines: {valid_machines}")
        invalid_metrics = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid metrics: {fields_info['key_figures']}")
        valid_operators = [">=", "<=", ">", "<", "==", "!="]
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator. Use one of: {valid_operators}")
        if key_values:
            for k, v in key_values.items():
                if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, []):
                    raise ValueError(f"Invalid filter: {k}={v}. Valid values for {k}: {fields_info['key_values'].get(k, [])}")
        final_rule = {
            "machines": machines,
            "key_figures": key_figures,
            "key_values": key_values or {},
            "operator": operator,
            "unit": unit,
            "description": description
        }
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
        metrics_desc = ", ".join(
            [f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()]
        )
        filters_desc = ", ".join([f"{k}={v}" for k, v in (key_values or {}).items()])
        message = f"Rule added for {len(machines)} machine(s): {metrics_desc}"
        if filters_desc:
            message += f" | Filters: {filters_desc}"
        logger.info(message)
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
        logger.error(f"Error adding rule: {str(e)}")
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
def list_custom_rules(
    ctx: Context,
    rule_id: Optional[str] = None,
    machine: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Lista las reglas personalizadas almacenadas en Qdrant.
    """
    try:
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
        rules = qdrant_client.scroll(
            collection_name="custom_rules",
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
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
                "applies_to": f"{len(rule.payload.get('machines', []))} machines",
                "metrics": list(rule.payload.get("key_figures", {}).keys())
            })
        logger.info(f"Retrieved {len(formatted_rules)} custom rules")
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
def delete_custom_rule(
    ctx: Context,
    rule_id: str
) -> str:
    """
    Elimina una regla personalizada de Qdrant.
    """
    try:
        existing = qdrant_client.retrieve(
            collection_name="custom_rules",
            ids=[rule_id],
            with_payload=True
        )
        if not existing:
            logger.error(f"Rule with ID {rule_id} not found")
            return json.dumps({
                "status": "error",
                "message": f"Rule with ID {rule_id} not found"
            }, ensure_ascii=False)
        qdrant_client.delete(
            collection_name="custom_rules",
            points_selector=models.PointIdsList(
                points=[rule_id]
            )
        )
        rule_data = existing[0].payload
        metrics = list(rule_data.get("key_figures", {}).keys())
        machines = rule_data.get("machines", [])
        logger.info(f"Rule deleted: {', '.join(metrics)} for {len(machines)} machine(s)")
        return json.dumps({
            "status": "success",
            "message": f"Rule deleted: {', '.join(metrics)} for {len(machines)} machine(s)",
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
def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Analiza el cumplimiento de los datos MES contra reglas SOP y personalizadas.

<<<<<<< HEAD
    INSTRUCCIONES PARA EL LLM:
    1. Antes de construir la consulta, llama a la función `list_fields` para obtener los campos disponibles en el dataset MES.
    2. Usa los campos listados en `key_figures` (campos numéricos) y `key_values` (campos categóricos) para armar la consulta.
    3. Solo utiliza campos que estén presentes en la respuesta de `list_fields`.
    4. La estructura de la consulta debe ser:

        {
            "key_values": {
                "<campo_categórico_1>": "<valor>",
                "<campo_categórico_2>": "<valor>",
                "start_date": "2025-04-09",
                "end_date": "2025-04-11"
            },
            "key_figures": [
                "<campo_numérico_1>",
                "<campo_numérico_2>"
            ]
        }

    5. Ejemplo dinámico (los campos deben ser seleccionados de la respuesta de `list_fields`):

        Supón que `list_fields` devuelve:
        {
            "key_figures": ["temperature", "uptime", "vibration"],
            "key_values": {
                "machine": ["ModelA", "ModelB"],
                "production_line": ["Line1", "Line2", "Line3"]
            }
        }

        Entonces, una consulta válida sería:
        {
            "key_values": {
                "machine": "ModelA",
                "production_line": "Line3",
                "start_date": "2025-04-09",
                "end_date": "2025-04-11"
            },
            "key_figures": ["temperature", "uptime", "vibration"]
        }

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos y valores para filtrar.
        key_figures (Optional[List[str]]): Lista de campos numéricos a analizar.

    Returns:
        str: JSON con el análisis de cumplimiento.
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        valid_figures = fields_info["key_figures"]
        logger.info(f"Analyzing compliance: key_figures={key_figures}, key_values={key_values}")

        # Identificar el campo y valor para la máquina
        identifier_field = None
        identifier_value = None
        for field in valid_values:
            if field not in ["start_date", "end_date"] and field in key_values:
                identifier_field = field
                identifier_value = key_values[field]
                break
        if not identifier_field and valid_values:
            identifier_field = next(iter(valid_values))
            identifier_value = key_values.get(identifier_field)

        # Obtener datos MES
        fetch_result = json.loads(fetch_mes_data(ctx, key_values, key_figures))
=======
    Esta función recupera datos del sistema MES, los compara con procedimientos operativos estándar (SOPs)
    almacenados en MinIO y reglas personalizadas en Qdrant, y genera un informe de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud proporcionado por el MCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos (e.g., {"machine": "ModelA"}).
        key_figures (Optional[List[str]]): Campos numéricos a analizar (e.g., ["temperature", "uptime"]).
        start_date (Optional[str]): Fecha de inicio para el rango de análisis (formato YYYY-MM-DD).
        end_date (Optional[str]): Fecha de fin para el rango de análisis (formato YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas para el análisis (formato YYYY-MM-DD).

    Returns:
        str: Respuesta en formato JSON con el estado, resultados, contenido de SOPs, reglas aplicadas y notas.

    INSTRUCCIONES PARA EL LLM:
    1. **Obtener campos válidos**: Antes de construir la consulta, llama a la función `list_fields` para obtener los
       campos disponibles en el dataset MES (key_figures y key_values).
    2. **Validar campos**: Usa solo campos presentes en la respuesta de `list_fields` para `key_figures` (numéricos) y
       `key_values` (categóricos).
    3. **Estructura de la consulta**: La consulta debe seguir esta estructura:
       ```json
       {
           "key_values": {
               "<campo_categórico_1>": "<valor>",
               "<campo_categórico_2>": "<valor>"
           },
           "key_figures": [
               "<campo_numérico_1>",
               "<campo_numérico_2>"
           ],
           // Usa EITHER specific_dates OR start_date/end_date, no ambos
           "specific_dates": ["YYYY-MM-DD", ...], // Para fechas específicas
           // O
           "start_date": "YYYY-MM-DD", // Para un rango de fechas
           "end_date": "YYYY-MM-DD"
       }
       ```
    4. **Cuándo usar specific_dates vs. start_date/end_date**:
       - Usa `specific_dates` cuando la consulta menciona días concretos (e.g., "solo el 9 de abril de 2025" o
         "9 y 11 de abril de 2025"). Ejemplo: `specific_dates: ["2025-04-09", "2025-04-11"]`.
       - Usa `start_date` y `end_date` cuando la consulta menciona un rango de fechas (e.g., "del 9 al 11 de abril de
         2025"). Ejemplo: `start_date: "2025-04-09", end_date: "2025-04-11"`.
       - No combines `specific_dates` con `start_date`/`end_date` en la misma consulta.
       - Si la consulta no especifica fechas, omite ambos parámetros.
    5. **Ejemplo dinámico**:
       Supón que `list_fields` devuelve:
       ```json
       {
           "key_figures": ["temperature", "uptime", "vibration"],
           "key_values": {
               "machine": ["ModelA", "ModelB"],
               "production_line": ["Line1", "Line2", "Line3"]
           }
       }
       ```
       Consultas válidas serían:
       - Para fechas específicas:
         ```json
         {
             "key_values": {
                 "machine": "ModelA",
                 "production_line": "Line3"
             },
             "key_figures": ["temperature", "uptime"],
             "specific_dates": ["2025-04-09"]
         }
         ```
       - Para un rango de fechas:
         ```json
         {
             "key_values": {
                 "machine": "ModelA",
                 "production_line": "Line3"
             },
             "key_figures": ["temperature", "uptime"],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
        - Para un rango de fechas sin key figures:
         ```json
         {
             "key_values": {
                "machine": "ModelA", 
                "production_line": "Line3"
             },
             "key_figures": [],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
    6. **Manejo de errores**:
       - Si los campos en `key_values` o `key_figures` no están en `list_fields`, ignora la consulta y devuelve un mensaje
         de error solicitando campos válidos.
       - Si las fechas proporcionadas no tienen el formato correcto (YYYY-MM-DD), solicita al usuario que las corrija.
    """
    try:
        key_values = key_values or {}
        # Si no se proporcionan key_figures, obtener todos los campos numéricos desde list_fields
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.error("No se pudieron obtener campos válidos")
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "results": [],
                    "analysis_notes": ["No se pudieron obtener campos válidos"]
                }, ensure_ascii=False)
            key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        valid_values = fields_info["key_values"]
        valid_figures = fields_info["key_figures"]
        logger.info(f"Analyzing compliance: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")

        # Selección dinámica de identifier_field
        identifier_field = None
        identifier_value = None
        if valid_values:
            for field in valid_values.keys():
                if field in key_values:
                    identifier_field = field
                    identifier_value = key_values[field]
                    break
            if not identifier_field:
                identifier_field = next(iter(valid_values))
                identifier_value = key_values.get(identifier_field)
        logger.info(f"Selected identifier_field: {identifier_field}, identifier_value: {identifier_value}")

        fetch_result = json.loads(fetch_mes_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
>>>>>>> dev
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []

        if fetch_result["status"] == "no_data":
            logger.warning(fetch_result["message"])
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
<<<<<<< HEAD
                "period": f"{key_values.get('start_date', 'N/A')} to {key_values.get('end_date', 'N/A')}",
=======
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
>>>>>>> dev
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": key_figures,
                "results": [],
                "sop_content": {},
                "custom_rules_applied": 0,
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        if fetch_result["status"] != "success":
            logger.error(fetch_result.get("message", "Error retrieving data"))
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Error retrieving data"),
                "results": [],
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)
<<<<<<< HEAD
=======

        date_field = find_date_field(fetch_result["data"], fields_info)
        logger.info(f"Date field for compliance analysis: {date_field}")
        if not date_field:
            analysis_notes.append("No date field detected; date filters ignored.")

        identifiers = {r[identifier_field] for r in fetch_result["data"] if identifier_field in r} if identifier_field else set()
        sop_content = {}
        if identifiers and identifier_field:
            for identifier in identifiers:
                pdf_result = json.loads(get_pdf_content(ctx, f"{identifier}.pdf"))
                if pdf_result["status"] == "success":
                    sop_content[identifier] = pdf_result["content"]
                    logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
        else:
            analysis_notes.append("No identifier field or identifiers found; no SOPs loaded.")
            logger.info("No identifier field or identifiers found; no SOPs loaded.")
>>>>>>> dev

        # Identificar el campo de fecha
        date_field = find_date_field(fetch_result["data"], fields_info)
        logger.info(f"Date field for compliance analysis: {date_field}")
        if not date_field:
            analysis_notes.append("No date field detected; date filters ignored.")

        # Obtener contenido de PDFs para cada máquina
        identifiers = {r[identifier_field] for r in fetch_result["data"] if identifier_field in r} if identifier_field else set()
        sop_content = {}
        for identifier in identifiers:
            if identifier_field == "machine":
                pdf_result = json.loads(get_pdf_content(ctx, f"{identifier}.pdf"))
                if pdf_result["status"] == "success":
                    sop_content[identifier] = pdf_result["content"]
                    logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {identifier}: {pdf_result['message']}")
            else:
                sop_content[identifier] = ""
                analysis_notes.append(f"No SOP loaded for {identifier_field}={identifier} (not a machine).")
                logger.info(f"No SOP loaded for {identifier_field}={identifier} (not a machine).")

        # Obtener reglas personalizadas
        custom_rules = []
<<<<<<< HEAD
        if identifiers and identifier_field == "machine":
            custom_result = qdrant_client.scroll(
                collection_name="custom_rules",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="machines", match=models.MatchAny(any=list(identifiers))),
=======
        if identifiers and identifier_field:
            custom_result = qdrant_client.scroll(
                collection_name="custom_rules",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key=identifier_field, match=models.MatchAny(any=list(identifiers))),
>>>>>>> dev
                ]),
                limit=100
            )
            custom_rules = [r.payload for r in custom_result[0]] if custom_result and custom_result[0] else []
<<<<<<< HEAD
            logger.info(f"Custom rules: {len(custom_rules)}")
=======
            logger.info(f"Custom rules found: {len(custom_rules)}")
>>>>>>> dev

        # Preparar resultados para el LLM
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida")
            }
            for k in key_values:
<<<<<<< HEAD
                if k not in ["start_date", "end_date"] and k in record:
=======
                if k in record:
>>>>>>> dev
                    analysis[k] = record[k]
            analysis.update({
                "metrics": {k: record[k] for k in key_figures if k in record}
            })
            results.append(analysis)

        period = "all dates"
        if specific_dates:
            period = f"Specific dates: {', '.join(specific_dates)}"
        elif start_date and end_date:
            period = f"{start_date} to {end_date}"

        return json.dumps({
            "status": "success",
            "period": period,
            "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_content": sop_content,
            "custom_rules": custom_rules,
            "analysis_notes": analysis_notes
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Compliance analysis failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "results": [],
            "analysis_notes": [str(e)]
<<<<<<< HEAD
=======
        }, ensure_ascii=False)

@mcp.tool()
def get_mes_dataset(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Recupera datos del sistema MES aplicando filtros por campos categóricos, métricas numéricas y fechas.

    INSTRUCCIONES PARA EL LLM:
    - Antes de construir la consulta, llama a `list_fields` para obtener los campos válidos (`key_figures` y `key_values`).
    - Usa solo campos presentes en la respuesta de `list_fields`.
    - Usa `specific_dates` (lista de fechas YYYY-MM-DD) para días concretos, o `start_date` y `end_date` (YYYY-MM-DD) para rangos. No combines ambos.
    - Si los campos o fechas no son válidos, devuelve un mensaje de error solicitando corrección.

    Ejemplos de uso:
    1. Fechas específicas:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": ["defects"],
           "specific_dates": ["2025-04-09", "2025-04-11"]
       }
    2. Rango de fechas:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": ["defects"],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }
    3. Sin filtros:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": [],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11",
       }

    Args:
        ctx (Context): Contexto FastMCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos.
        key_figures (Optional[List[str]]): Métricas numéricas.
        start_date (Optional[str]): Fecha inicio (YYYY-MM-DD).
        end_date (Optional[str]): Fecha fin (YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas (YYYY-MM-DD).

    Returns:
        str: JSON con los datos filtrados.
    """
    try:
        key_values = key_values or {}
        # Si no se proporcionan key_figures, obtener todos los campos numéricos desde list_fields
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.error("No se pudieron obtener campos válidos")
                return json.dumps([], ensure_ascii=False)
            key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        valid_figures = fields_info["key_figures"]
        valid_values = fields_info["key_values"]
        logger.info(f"Fetching MES dataset for key_values={key_values}, key_figures={key_figures}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")

        # Construir condiciones de filtrado para Qdrant
        must_conditions = []
        for k, v in key_values.items():
            if k in valid_values:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if specific_dates:
            normalized_dates = [detect_and_normalize_date(d) for d in specific_dates if detect_and_normalize_date(d)]
            if not normalized_dates:
                logger.error("No valid dates provided in specific_dates")
                return json.dumps([], ensure_ascii=False)
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchAny(any=normalized_dates)
            ))
        elif start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                    must_conditions.append(models.FieldCondition(
                        key="date",
                        match=models.MatchAny(any=date_range)
                    ))
            except ValueError as e:
                logger.error(f"Invalid date format: {str(e)}")
                return json.dumps([], ensure_ascii=False)

        # Recuperar datos de Qdrant
        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = []
        for r in qdrant_results[0] if qdrant_results[0] else []:
            try:
                encrypted_payload = r.payload.get("encrypted_payload")
                if encrypted_payload:
                    decrypted_data = fernet.decrypt(encrypted_payload.encode()).decode()
                    processed_data.append(json.loads(decrypted_data))
                else:
                    logger.warning(f"No encrypted payload for point {r.id}")
            except InvalidToken:
                logger.error(f"Failed to decrypt payload for point {r.id}")
                continue
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")

        # Obtener datos frescos de la API si no hay datos en Qdrant o si no hay filtros
        if not processed_data or not (key_values or key_figures or start_date or end_date or specific_dates):
            params = {}
            if specific_dates:
                params.update({"specific_date": specific_dates[0]})  # API soporta specific_date
            elif start_date and end_date:
                params.update({"start_date": start_date, "end_date": end_date})
            response = auth_client.get("/machines/", params=params)
            response.raise_for_status()
            all_data = response.json()
            date_field = find_date_field(all_data, fields_info)
            logger.info(f"Detected date field: {date_field}")
            full_data = []
            for record in all_data:
                item = {}
                if date_field and date_field in record:
                    normalized_date = detect_and_normalize_date(str(record[date_field]))
                    item["date"] = normalized_date or "Desconocida"
                else:
                    item["date"] = "Desconocida"
                for field in valid_figures + list(valid_values.keys()):
                    if field in record:
                        item[field] = record[field]
                full_data.append(item)
            if full_data and specific_dates:
                normalized_dates = [detect_and_normalize_date(d) for d in specific_dates if detect_and_normalize_date(d)]
                full_data = [r for r in full_data if r.get("date") in normalized_dates]
            elif full_data and start_date and end_date:
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    delta = (end - start).days + 1
                    if delta > 0:
                        for i, record in enumerate(full_data):
                            if record["date"] == "Desconocida":
                                record["date"] = (start + timedelta(days=i % delta)).strftime("%Y-%m-%d")
                except ValueError:
                    pass
            # Encriptar y almacenar en Qdrant
            if full_data:
                points = []
                for r in full_data:
                    payload_json = json.dumps(r)
                    encrypted_payload = fernet.encrypt(payload_json.encode()).decode()
                    point = models.PointStruct(
                        id=hashlib.md5(payload_json.encode()).hexdigest(),
                        vector=model.encode(payload_json).tolist(),
                        payload={"encrypted_payload": encrypted_payload}
                    )
                    points.append(point)
                qdrant_client.upsert(collection_name="mes_logs", points=points)
                logger.info(f"Stored {len(points)} encrypted points in Qdrant mes_logs")
            processed_data = full_data

        # Filtrar datos en memoria si es necesario
        if processed_data:
            data_filters = {k: v for k, v in key_values.items()}
            processed_data = [
                r for r in processed_data
                if all(r.get(k) == v for k, v in data_filters.items())
            ]
            logger.info(f"Filtered {len(processed_data)} records in memory for {data_filters}")

        # Seleccionar campos solicitados
        response_fields = ["date"] + list(key_values.keys()) + key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r}
            for r in processed_data
        ]

        # Devolver solo los datos en JSON
        return json.dumps(response_data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dataset retrieval failed: {str(e)}")
        return json.dumps([], ensure_ascii=False)

@mcp.tool()
def list_available_tools(ctx: Context) -> str:
    """
    Lista todas las herramientas disponibles definidas con el decorador @mcp.tool().
    """
    try:
        tools = []
        # Intento 1: Acceder al registro interno de herramientas de FastMCP
        try:
            if hasattr(mcp, 'tools'):
                tool_registry = mcp.tools
                logger.info("Accediendo al registro interno de herramientas de FastMCP")
            elif hasattr(mcp, 'get_tools'):
                tool_registry = mcp.get_tools()
                logger.info("Accediendo a get_tools() de FastMCP")
            else:
                tool_registry = None
                logger.warning("No se encontró registro interno de herramientas en FastMCP")

            if tool_registry:
                for tool_name, tool_func in tool_registry.items():
                    if callable(tool_func):
                        signature = inspect.signature(tool_func)
                        parameters = [
                            {
                                "name": param_name,
                                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                                "default": param.default if param.default != inspect.Parameter.empty else None
                            }
                            for param_name, param in signature.parameters.items()
                            if param_name != 'ctx'
                        ]
                        docstring = inspect.getdoc(tool_func) or "Sin descripción disponible."
                        tools.append({
                            "name": tool_name,
                            "description": docstring,
                            "parameters": parameters
                        })
                        logger.debug(f"Tool registrada desde FastMCP: {tool_name}")
        except Exception as e:
            logger.warning(f"Fallo al acceder al registro interno de FastMCP: {str(e)}")

        # Intento 2: Inspección del módulo como respaldo
        if not tools:
            logger.info("Realizando inspección del módulo como respaldo")
            module = inspect.getmodule(inspect.currentframe())
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    try:
                        source = inspect.getsource(obj)
                        if '@mcp.tool' in source:
                            signature = inspect.signature(obj)
                            parameters = [
                                {
                                    "name": param_name,
                                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                                    "default": param.default if param.default != inspect.Parameter.empty else None
                                }
                                for param_name, param in signature.parameters.items()
                                if param_name != 'ctx'
                            ]
                            docstring = inspect.getdoc(obj) or "Sin descripción disponible."
                            tools.append({
                                "name": name,
                                "description": docstring,
                                "parameters": parameters
                            })
                            logger.debug(f"Tool detectada por inspección: {name}")
                    except Exception as e:
                        logger.debug(f"No se pudo inspeccionar la función {name}: {str(e)}")

        if not tools:
            logger.warning("No se encontraron herramientas disponibles")

        logger.info(f"Retrieved {len(tools)} available tools")
        return json.dumps({
            "status": "success" if tools else "no_data",
            "count": len(tools),
            "tools": tools,
            "message": "Lista de herramientas recuperada exitosamente." if tools else "No se encontraron herramientas disponibles."
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to list available tools: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "tools": []
>>>>>>> dev
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_infrastructure()
    mcp.run()