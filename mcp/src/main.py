import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
import logging
import json
import hashlib
import re
from minio import Minio
from minio.error import S3Error
import pdfplumber
import io
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración del MCP
mcp = FastMCP("Manufacturing Compliance Processor")
API_URL = os.getenv("API_URL", "http://api:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sop-pdfs")
API_USERNAME = "admin"
API_PASSWORD = "password123"

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
    # Evitar procesar valores claramente no relacionados con fechas
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
    logger.debug(f"No se pudo parsear la fecha: {date_str}")
    return None

def find_date_field(records: List[Dict], fields_info: Dict) -> Optional[str]:
    """
    Detecta el campo que contiene fechas en los registros.
    Retorna el campo cacheado si está disponible y el esquema no ha cambiado.
    """
    global DATE_FIELD_CACHE
    if not records or not fields_info:
        logger.warning("No records or fields_info provided for date detection")
        return None

    # Generar un hash del esquema para detectar cambios
    schema = json.dumps(fields_info, sort_keys=True)
    schema_hash = hashlib.md5(schema.encode()).hexdigest()

    # Usar el campo cacheado si el esquema no ha cambiado
    if DATE_FIELD_CACHE["field"] and DATE_FIELD_CACHE["schema_hash"] == schema_hash:
        logger.info(f"Using cached date field: {DATE_FIELD_CACHE['field']}")
        return DATE_FIELD_CACHE["field"]

    key_values = fields_info.get("key_values", {})
    if not key_values:
        logger.warning("No categorical fields available in fields_info")
        return None

    best_candidate = None
    best_score = 0
    # Limitar a los primeros 10 registros para mejorar el rendimiento
    for field in key_values.keys():
        # Excluir campos claramente no relacionados con fechas
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
        # Actualizar el caché
        DATE_FIELD_CACHE["field"] = best_candidate
        DATE_FIELD_CACHE["schema_hash"] = schema_hash
        return best_candidate
    else:
        logger.warning("No date field detected in the provided records")
        return None

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
    return {
        "has_data": bool(covered_dates),
        "covered_dates": covered_dates,
        "message": message
    }

class AuthClient:
    """
    Cliente para autenticación con la API.
    """
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.client = httpx.Client()
        self.token = None

    def authenticate(self):
        try:
            response = self.client.post(
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

    def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        if not self.token:
            self.authenticate()
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params
            )
            if response.status_code == 401:
                logger.info("Token inválido, reautenticando...")
                self.authenticate()
                headers = {"Authorization": f"Bearer {self.token}"}
                response = self.client.get(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    params=params
                )
            return response
        except Exception as e:
            logger.error(f"Error en solicitud GET: {str(e)}")
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
        auth_client = AuthClient(API_URL, API_USERNAME, API_PASSWORD)
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
            errors = []
            invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
            if invalid_figures:
                valid_figures = fields_info["key_figures"]
                errors.append(f"Campos numéricos inválidos: {invalid_figures}. Campos disponibles: {', '.join(valid_figures)}.")
            invalid_values = {}
            for k, v in key_values.items():
                if k in ["start_date", "end_date"]:
                    continue
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
    key_figures: Optional[List[str]] = None
) -> str:
    """
    Analiza el cumplimiento de los datos MES contra reglas SOP y personalizadas.

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
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []

        if fetch_result["status"] == "no_data":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{key_values.get('start_date', 'N/A')} to {key_values.get('end_date', 'N/A')}",
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": key_figures,
                "results": [],
                "sop_content": {},
                "custom_rules_applied": 0,
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Error retrieving data"),
                "results": [],
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

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
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier}: {pdf_result['message']}")
                logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
            else:
                sop_content[identifier] = ""
                analysis_notes.append(f"No SOP loaded for {identifier_field}={identifier} (not a machine).")

        # Obtener reglas personalizadas
        custom_rules = []
        if identifiers and identifier_field == "machine":
            custom_result = qdrant_client.scroll(
                collection_name="custom_rules",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="machines", match=models.MatchAny(any=list(identifiers))),
                ]),
                limit=100
            )
            custom_rules = [r.payload for r in custom_result[0]] if custom_result and custom_result[0] else []
            logger.info(f"Custom rules: {len(custom_rules)}")

        # Preparar resultados para el LLM
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida")
            }
            for k in key_values:
                if k not in ["start_date", "end_date"] and k in record:
                    analysis[k] = record[k]
            analysis.update({
                "metrics": {k: record[k] for k in key_figures if k in record}
            })
            results.append(analysis)

        period = "all dates"
        if "start_date" in key_values and "end_date" in key_values:
            period = f"{key_values['start_date']} to {key_values['end_date']}"

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
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_infrastructure()
    mcp.run()