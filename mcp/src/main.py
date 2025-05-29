import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
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
import spacy
from spacy.language import Language
from spacy.tokens import Span

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

mcp = FastMCP("Manufacturing Compliance Processor")

API_URL = os.getenv("API_URL", "http://api:5000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sop-pdfs")
API_USERNAME = "admin"
API_PASSWORD = "password123"

model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(host="qdrant", port=6333)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y%m%d",
    "%Y/%m/%d", "%b %d, %Y", "%d %b %Y"
]

def detect_and_normalize_date(date_str: str) -> Optional[str]:
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning(f"No se pudo parsear la fecha: {date_str}")
    return None

def find_date_field(records: List[Dict], fields_info: Dict) -> Optional[str]:
    date_field_candidates = ["date", "timestamp", "created_at", "record_date", "time", "datetime"]
    key_values = fields_info.get("key_values", {})
    for field in key_values:
        if field.lower() in [c.lower() for c in date_field_candidates]:
            sample_values = [r.get(field) for r in records[:10] if field in r]
            valid_dates = [detect_and_normalize_date(str(v)) for v in sample_values]
            if any(valid_dates):
                logger.info(f"Campo de fecha detectado por nombre: {field}")
                return field
    for field in key_values:
        sample_values = [r.get(field) for r in records[:10] if field in r]
        valid_dates = [detect_and_normalize_date(str(v)) for v in sample_values]
        if len([d for d in valid_dates if d]) > len(sample_values) * 0.5:
            logger.info(f"Campo de fecha detectado por valores: {field}")
            return field
    logger.warning("No se detectó campo de fecha")
    return None

def check_date_coverage(data: List[Dict], start_date: str, end_date: str) -> Dict:
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
    @staticmethod
    def validate_date(date_str: str, field: str) -> str:
        normalized_date = detect_and_normalize_date(date_str)
        if normalized_date is None:
            raise ValueError(f"Formato inválido para {field}: {date_str}. Formatos soportados: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY, etc.")
        return normalized_date

    @staticmethod
    def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict[str, str]) -> Dict:
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

def clean_ocr_text(content: str) -> str:
    content = re.sub(r"\$_[0-9a-z]\$", "", content)
    content = content.replace("$_1$", "").replace("$_c$", "").replace("$_v$", "")
    content = re.sub(r"\s+", " ", content)
    content = content.replace("•", "").replace("◦", "")
    content = re.sub(r"([a-zA-Z])\s+([a-zA-Z])", r"\1\2", content)
    content = content.replace("ime≤", "time ≤").replace("ount≤", "count ≤")
    content = content.replace("onsumption", "consumption").replace("ate≥", "rate ≥")
    content = content.replace(r"$<=$", "<=").replace(r"$>=$", ">=").replace(r"^{\circ}", "°").replace(r"\%", "%")
    content = re.sub(r"<[ ]*=", "<=", content)
    content = re.sub(r">+[ ]*=", ">=", content)
    content = re.sub(r"(\w+)(<=|>=|=|<|>)", r"\1 \2", content)
    return content.strip()

def detect_operator_between(param_end: int, value_start: int, sentence: str) -> str:
    text_before = sentence[max(0, param_end - 20):value_start].lower()
    operator_map = {
        ">=": [r">=", r"≥", r"greater\s+than\s+or\s+equal", r"at\s+least"],
        "<=": [r"<=", r"≤", r"less\s+than\s+or\s+equal", r"at\s+most", r"not\s+exceeding"],
        ">": [r">", r"greater\s+than", r"exceeds"],
        "<": [r"<", r"less\s+than", r"below", r"must\s+be\s+below"],
        "==": [r"=", r"==", r"equals", r"equal\s+to"],
        "!=": [r"!=", r"not\s+equal"]
    }
    for op_key, patterns in operator_map.items():
        for pattern in patterns:
            if re.search(pattern, text_before):
                return op_key
    logger.warning(f"No operator detected in '{text_before}', defaulting to >=")
    return ">="

def detect_unit_near(value_end: int, param: str, sentence: str) -> str:
    text_after = sentence[value_end:value_end + 20].lower()
    unit_patterns = [
        (r"°[cf]", ["temperature"]),
        (r"mm/s|in/s|mm/sec", ["vibration"]),
        (r"%", ["uptime", "humidity"]),
        (r"bar|psi|hpa", ["pressure"]),
        (r"/min|/h", ["throughput", "production_rate", "output_rate"]),
        (r"db|decibel", ["noise_level"]),
        (r"rpm", ["speed"]),
        (r"mm|cm|m|in|ft", ["length", "width", "height"]),
    ]
    for pattern, params in unit_patterns:
        if not params or param in params:
            match = re.search(pattern, text_after)
            if match:
                return match.group(0)
    generic_unit = re.search(r"[a-zA-Z0-9]+(?:/[a-zA-Z0-9]+)?(?:\^[0-9]+)?", text_after)
    if generic_unit:
        logger.info(f"Detected generic unit for {param}: {generic_unit.group(0)}")
        return generic_unit.group(0)
    return "" if param in ["error_count", "defects", "inventory_level"] else ""

def generate_field_synonyms(valid_fields: List[str]) -> Dict[str, List[str]]:
    synonyms = {}
    for field in valid_fields:
        base_synonyms = [field, field.replace("_", " ")]
        if field.endswith("s"):
            base_synonyms.append(field[:-1])
        elif field in ["uptime", "downtime", "runtime"]:
            base_synonyms.extend([f"{field[:-4]}{suffix}" for suffix in ["", " percentage", " ratio"]])
        elif "temperature" in field:
            base_synonyms.extend(["temp", "thermo", f"{field} level"])
        elif "vibration" in field:
            base_synonyms.extend(["vib", "oscillation", f"{field} amplitude"])
        elif "defects" in field or "error_count" in field:
            base_synonyms.extend(["defect", "rejections", "faults", "errors", "error"])
        elif "pressure" in field:
            base_synonyms.extend(["press", "force"])
        elif "humidity" in field:
            base_synonyms.extend(["moisture", "hum"])
        elif "throughput" in field or "rate" in field:
            base_synonyms.extend(["output", "production rate", "flow"])

        synonyms[field] = list(set(base_synonyms))
    return synonyms

def load_sop(machine: str) -> Dict:
    try:
        existing = qdrant_client.scroll(
            collection_name="sop_pdfs",
            scroll_filter=models.Filter(must=[models.FieldCondition(key="machine", match=models.MatchValue(value=machine))]),
            limit=1
        )
        if existing[0]:
            logger.info(f"Rules found in Qdrant for {machine}")
            return existing[0][0].payload.get("rules", {})

        try:
            response = minio_client.get_object(MINIO_BUCKET, f"{machine}.pdf")
            pdf_data = response.read()
            response.close()
            response.release_conn()
        except S3Error:
            logger.warning(f"PDF not found for {machine}")
            return {}

        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            content = ""
            for page in pdf.pages:
                content += (page.extract_text() or "")
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        content += " ".join(str(cell) for cell in row if cell) + "\n"

        content = clean_ocr_text(content)
        logger.info(f"Extracted text for {machine}: {content[:100]}...")

        fields_info = json.loads(list_fields(Context()))
        valid_fields = fields_info.get("key_figures", []) if fields_info.get("status") == "success" else []
        logger.info(f"Valid fields from list_fields: {valid_fields}")

        field_synonyms = generate_field_synonyms(valid_fields)
        logger.info(f"Generated field_synonyms: {field_synonyms}")

        @Language.component("custom_ner")
        def custom_ner(doc):
            new_ents = []
            for token in doc:
                for field, synonyms in field_synonyms.items():
                    if token.text.lower() in [s.lower() for s in synonyms]:
                        new_ents.append(Span(doc, token.i, token.i + 1, label="PARAM"))
                        break
                if re.match(r"\d+\.?\d*(?:[a-zA-Z/°%]+)?", token.text):
                    new_ents.append(Span(doc, token.i, token.i + 1, label="VALUE"))
            doc.ents = new_ents
            return doc

        if "custom_ner" in nlp.pipe_names:
            nlp.remove_pipe("custom_ner")
        nlp.add_pipe("custom_ner", last=True)

        rules = {}
        doc = nlp(content)
        sentences = list(doc.sents)

        for sent in sentences:
            entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in sent.ents]
            param_ents = [(text, start, end) for text, label, start, end in entities if label == "PARAM"]
            value_ents = [(text, start, end) for text, label, start, end in entities if label == "VALUE"]

            logger.info(f"Entities in sentence '{sent.text}': PARAM={param_ents}, VALUE={value_ents}")

            for param_text, param_start, param_end in param_ents:
                param = None
                for field, synonyms in field_synonyms.items():
                    if param_text.lower() in [s.lower() for s in synonyms]:
                        param = field
                        break
                if not param:
                    logger.warning(f"Parameter {param_text} not mapped to valid field")
                    continue

                closest_value = None
                min_distance = float('inf')
                selected_operator = None
                selected_unit = ""

                for value_text, value_start, value_end in value_ents:
                    match = re.match(r"(\d+\.?\d*)([a-zA-Z/°%]+)?", value_text)
                    if not match:
                        logger.warning(f"Invalid value format: {value_text}")
                        continue
                    number = match.group(1)
                    unit_in_text = match.group(2) or ""
                    try:
                        value = float(number)
                    except ValueError:
                        logger.warning(f"Invalid numeric value: {number}")
                        continue

                    distance = value_start - param_end
                    if distance < 0 or distance > 100:
                        continue

                    try:
                        operator = detect_operator_between(param_end, value_start, sent.text)
                        unit = unit_in_text or detect_unit_near(value_end, param, sent.text)
                    except Exception as e:
                        logger.error(f"Error detecting operator/unit for {param_text} and {value_text}: {str(e)}")
                        continue

                    if distance < min_distance:
                        min_distance = distance
                        closest_value = value
                        selected_operator = operator
                        selected_unit = unit

                if closest_value is not None:
                    rules[param] = {
                        "value": closest_value,
                        "operator": selected_operator,
                        "unit": selected_unit,
                        "source_text": sent.text
                    }
                    logger.info(f"Rule extracted: {param} {selected_operator} {closest_value}{selected_unit}")

        if rules:
            embedding = model.encode(content).tolist()
            qdrant_client.upsert(
                collection_name="sop_pdfs",
                points=[models.PointStruct(
                    id=hashlib.md5(machine.encode()).hexdigest(),
                    vector=embedding,
                    payload={
                        "machine": machine,
                        "filename": f"{machine}.pdf",
                        "rules": rules
                    }
                )]
            )
            logger.info(f"Rules stored in Qdrant for {machine}: {rules}")

        return rules

    except Exception as e:
        logger.error(f"SOP processing failed for {machine}: {str(e)}")
        return {}

@mcp.tool()
def get_pdf_content(ctx: Context, filename: str) -> str:
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
    """Lista los campos disponibles en el dataset MES.
    Returns:
        str: JSON con los campos disponibles.
            - `key_figures`: Lista de campos numéricos (e.g., ["temperature", "uptime"]).
            - `key_values`: Diccionario de campos categóricos y sus valores posibles
              (e.g., {"machine": ["ModelA", "ModelB"], "production_line": ["Line1", "Line2"]}).
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
        Obtiene datos MES con filtros dinámicos.
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        logger.info(f"Fetching MES data for key_values={key_values}, key_figures={key_figures}")

        must_conditions = []
        if "machine" in key_values:
            must_conditions.append(models.FieldCondition(key="machine", match=models.MatchValue(value=key_values["machine"])))
        if "production_line" in key_values:
            must_conditions.append(models.FieldCondition(key="production_line", match=models.MatchValue(value=key_values["production_line"])))
        if "start_date" in key_values and "end_date" in key_values:
            start_date = datetime.strptime(key_values["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(key_values["end_date"], "%Y-%m-%d")
            delta = (end_date - start_date).days + 1
            if delta > 0:
                date_range = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                must_conditions.append(models.FieldCondition(
                    key="date",
                    match=models.MatchAny(any=date_range)
                ))
        for k, v in key_values.items():
            if k not in ["machine", "production_line", "start_date", "end_date"] and k in valid_values:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = [r.payload for r in qdrant_results[0]] if qdrant_results[0] else []
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")

        if not processed_data:
            params = {}
            if "start_date" in key_values and "end_date" in key_values:
                params.update({
                    "start_date": key_values["start_date"],
                    "end_date": key_values["end_date"]
                })
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}

            endpoint = f"/machines/{key_values['machine']}" if "machine" in key_values else "/machines/"
            response = auth_client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            date_field = find_date_field(data, fields_info)
            logger.info(f"Campo de fecha detectado: {date_field}")

            processed_data = []
            for record in data:
                if all(record.get(k) == v for k, v in data_filters.items()):
                    item = {}
                    if date_field and date_field in record:
                        normalized_date = detect_and_normalize_date(str(record[date_field]))
                        item["date"] = normalized_date or ""
                    else:
                        item["date"] = ""
                    for k in key_figures:
                        if k in record:
                            item[k] = record[k]
                    for k in valid_values:
                        if k in record:
                            item[k] = record[k]
                    processed_data.append(item)

            if processed_data and "start_date" in key_values and "end_date" in key_values:
                start_date = datetime.strptime(key_values["start_date"], "%Y-%m-%d")
                end_date = datetime.strptime(key_values["end_date"], "%Y-%m-%d")
                delta = (end_date - start_date).days + 1
                if delta > 0:
                    for i, record in enumerate(processed_data):
                        if not record["date"]:
                            record["date"] = (start_date + timedelta(days=i % delta)).strftime("%Y-%m-%d")
                else:
                    for record in processed_data:
                        if not record["date"]:
                            record["date"] = "Desconocida"
            else:
                for record in processed_data:
                    if not record["date"]:
                        record["date"] = "Desconocida"

            if processed_data:
                points = [
                    models.PointStruct(
                        id=hashlib.md5(json.dumps(r).encode()).hexdigest(),
                        vector=model.encode(json.dumps(r)).tolist(),
                        payload=r
                    ) for r in processed_data
                ]
                qdrant_client.upsert(collection_name="mes_logs", points=points)
                logger.info(f"Stored {len(points)} points in Qdrant mes_logs")

        coverage = {"has_data": bool(processed_data), "covered_dates": [], "message": ""}
        if processed_data and "start_date" in key_values and "end_date" in key_values:
            coverage = check_date_coverage(processed_data, key_values["start_date"], key_values["end_date"])

        if not processed_data:
            return json.dumps({
                "status": "no_data",
                "count": 0,
                "data": [],
                "message": coverage["message"] or "No se encontraron registros para los filtros solicitados.",
                "covered_dates": []
            }, ensure_ascii=False)

        response_data = [
            {k: r[k] for k in (["date"] + list(valid_values) + key_figures) if k in r}
            for r in processed_data
        ]

        return json.dumps({
            "status": "success",
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
    """Analiza cumplimiento de datos MES contra reglas SOP/personalizadas.

    Args:
        ctx (Context): Contexto FastMCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos (de `list_fields` `key_values`).
            - Solo incluir campos solicitados por el usuario.
            - Incluir `start_date` y `end_date` (YYYY-MM-DD) si se filtran fechas.
            - Ejemplo (ilustrativo, usar `list_fields`): `{"machine": "ModelB", "start_date": "2025-04-09"}`
            - Nota: No generar campos vacíos (e.g., `{"machine": ""}`).
        key_figures (Optional[List[str]]): Campos numéricos (de `list_fields` `key_figures`).
            - Usar los solicitados o todos los relevantes.
            - Ejemplo (ilustrativo, usar `list_fields`): `["uptime", "vibration"]`

    Instrucciones: Consultar `list_fields` para campos válidos. No usar ejemplos si no se piden.

    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        logger.info(f"Validating fields: key_figures={key_figures}, key_values={key_values}")
        
        identifier_field = None
        identifier_value = None
        if key_values:
            for field in key_values:
                if field in valid_values and field not in ["start_date", "end_date"]:
                    identifier_field = field
                    identifier_value = key_values[field]
                    break
        if not identifier_field and valid_values:
            identifier_field = next(iter(valid_values))
            identifier_value = key_values.get(identifier_field)

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
                "sop_coverage": "0/0 machines with SOP",
                "custom_rules_applied": 0,
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Error al obtener datos"),
                "results": [],
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        filter_fields = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}
        if filter_fields:
            fetch_result["data"] = [
                r for r in fetch_result["data"]
                if all(r.get(k) == v for k, v in filter_fields.items())
            ]
            if not fetch_result["data"]:
                analysis_notes.append(f"No se encontraron registros para los filtros: {', '.join(f'{k}={v}' for k, v in filter_fields.items())}.")
                return json.dumps({
                    "status": "no_data",
                    "message": "No se encontraron registros para los filtros especificados.",
                    "period": f"{key_values.get('start_date', 'N/A')} to {key_values.get('end_date', 'N/A')}",
                    "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                    "metrics_analyzed": key_figures,
                    "results": [],
                    "sop_coverage": "0/0 machines with SOP",
                    "custom_rules_applied": 0,
                    "analysis_notes": analysis_notes
                }, ensure_ascii=False)
            logger.info(f"Filtered data by {filter_fields}: {len(fetch_result['data'])} records")

        machines = {r[identifier_field] for r in fetch_result["data"] if identifier_field in r} if identifier_field else set()
        machine_rules = {}
        for machine in machines:
            machine_rules[machine] = load_sop(machine)
            logger.info(f"SOP rules for {machine}: {machine_rules[machine]}")

        custom_rules = []
        if machines:
            custom_result = qdrant_client.scroll(
                collection_name="custom_rules",
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="machines", match=models.MatchAny(any=list(machines))),
                ]),
                limit=100
            )
            custom_rules = [r.payload for r in custom_result[0]] if custom_result[0] else []
            logger.info(f"Custom rules: {len(custom_rules)}")

        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "")
            }
            for k in key_values:
                if k not in ["start_date", "end_date"] and k in record:
                    analysis[k] = record[k]
            
            analysis.update({
                "metrics": {k: record[k] for k in key_figures if k in record},
                "compliance": {},
                "compliance_percentage": 0.0
            })
            
            compliant = 0
            total = 0
            
            for field in key_figures:
                if field not in record:
                    logger.warning(f"Field {field} not in record: {record}")
                    analysis_notes.append(f"El campo '{field}' no está presente en el registro para la fecha {record.get('date', 'Desconocida')}.")
                    continue
                    
                total += 1
                compliance_info = {}
                
                machine = record.get(identifier_field) if identifier_field else None
                sop_rules = machine_rules.get(machine, {}) if machine else {}
                if field in sop_rules:
                    rule = sop_rules[field]
                    value = record[field]
                    op = rule["operator"]
                    is_compliant = False
                    if op == ">=":
                        is_compliant = value >= rule['value']
                    elif op == "<=":
                        is_compliant = value <= rule['value']
                    elif op == ">":
                        is_compliant = value > rule['value']
                    elif op == "<":
                        is_compliant = value < rule['value']
                    elif op == "==":
                        is_compliant = value == rule['value']
                    elif op == "!=":
                        is_compliant = value != rule['value']
                    compliance_info["sop"] = {
                        "value": value,
                        "rule": f"{op} {rule['value']}{rule.get('unit', '')}",
                        "status": "is_compliant" if is_compliant else "non_compliant"
                    }
                    compliant += 1 if is_compliant else 0
                    logger.info(f"SOP compliance for {field}: {compliance_info['sop']}")
                else:
                    compliance_info["sop"] = {
                        "value": record[field],
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }
                
                compliance_info["custom"] = []
                for rule in custom_rules:
                    if field not in rule["key_figures"]:
                        continue
                    if identifier_field and record.get(identifier_field) not in rule["machines"]:
                        continue
                    if rule["key_values"]:
                        if not all(record.get(k) == v for k, v in rule["key_values"].items()):
                            continue
                    rule_value = rule["key_figures"][field]
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
                        "status": "is_compliant" if is_compliant else "non_compliant",
                        "description": rule.get("description", ""),
                        "filters": rule["key_values"]
                    })
                    logger.info(f"Custom compliance for {field}: {compliance_info['custom'][-1]}")
                
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

        period = "all dates"
        if "start_date" in key_values and "end_date" in key_values:
            period = f"{key_values['start_date']} to {key_values['end_date']}"

        logger.info(f"Compliance analysis completed: {len(results)} results")
        return json.dumps({
            "status": "success",
            "period": period,
            "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sum(1 for r in machine_rules.values() if r)}/{len(machines)} machines with SOP",
            "custom_rules_applied": len(custom_rules),
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