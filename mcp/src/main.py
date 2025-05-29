import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime
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
    def validate_date(date_str: str, field: str) -> None:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid format for {field}. Use YYYY-MM-DD: {date_str}")

    @staticmethod
    def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict[str, str]) -> Dict:
        try:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                raise ValueError("Could not validate against API")

            if "start_date" in key_values or "end_date" in key_values:
                if "start_date" not in key_values or "end_date" not in key_values:
                    raise ValueError("Both start_date and end_date must be provided")
                DataValidator.validate_date(key_values["start_date"], "start_date")
                DataValidator.validate_date(key_values["end_date"], "end_date")
                if key_values["start_date"] > key_values["end_date"]:
                    raise ValueError("start_date cannot be later than end_date")

            invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
            invalid_values = {
                k: v for k, v in key_values.items()
                if k not in fields_info["key_values"] and k not in ["start_date", "end_date"]
                or (k in fields_info["key_values"] and v not in fields_info["key_values"].get(k, []))
            }
            
            if invalid_figures or invalid_values:
                errors = []
                if invalid_figures:
                    errors.append(f"Invalid numeric fields: {invalid_figures}")
                if invalid_values:
                    errors.append(f"Invalid categorical fields: {invalid_values}")
                raise ValueError(" | ".join(errors))
            
            return fields_info
        except Exception as e:
            logger.error(f"Field validation failed: {str(e)}")
            raise

def clean_ocr_text(content: str) -> str:
    """Limpia artefactos OCR comunes y normaliza texto."""
    content = re.sub(r"\$_[0-9a-z]\$", "", content)  # Elimina $_1$, $_c$, etc.
    content = content.replace("$_1$", "").replace("$_c$", "").replace("$_v$", "")
    content = re.sub(r"\s+", " ", content)  # Normaliza espacios
    content = content.replace("•", "").replace("◦", "")  # Elimina viñetas
    content = re.sub(r"([a-zA-Z])\s+([a-zA-Z])", r"\1\2", content)  # Une palabras rotas (e.g., 'ime t' → 'imet')
    content = content.replace("ime≤", "time ≤").replace("ount≤", "count ≤")  # Corrige cycle_time, error_count
    content = content.replace("onsumption", "consumption").replace("ate≥", "rate ≥")  # Corrige power_consumption, output_rate
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
    text_after = sentence[value_end:value_end + 20].lower()  # Ampliar ventana
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
                # Detectar valores numéricos con o sin unidades
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
                    # Separar número y unidad
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
    try:
        response = auth_client.get("/machines/")
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
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        
        identifier_field = None
        identifier_value = None
        if key_values:
            for field in key_values:
                if field in valid_values:
                    identifier_field = field
                    identifier_value = key_values[field]
                    break
        if not identifier_field and valid_values:
            identifier_field = next(iter(valid_values))
            identifier_value = key_values.get(identifier_field)

        must_conditions = []
        if identifier_field and identifier_value:
            must_conditions.append(models.FieldCondition(key=identifier_field, match=models.MatchValue(value=identifier_value)))
        if "start_date" in key_values and "end_date" in key_values:
            must_conditions.append(models.FieldCondition(
                key="date",
                range=models.Range(
                    gte=datetime.strptime(key_values["start_date"], "%Y-%m-%d").timestamp(),
                    lte=datetime.strptime(key_values["end_date"], "%Y-%m-%d").timestamp()
                )
            ))
        for k, v in key_values.items():
            if k != identifier_field and k not in ["start_date", "end_date"]:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )

        processed_data = [r.payload for r in qdrant_results[0]] if qdrant_results[0] else []

        if not processed_data:
            params = {}
            if "start_date" in key_values and "end_date" in key_values:
                params.update({
                    "start_date": key_values["start_date"],
                    "end_date": key_values["end_date"]
                })
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}

            endpoint = f"/machines/{identifier_value}" if identifier_field and identifier_value else "/machines/"
            response = auth_client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            processed_data = []
            for record in data:
                if all(record.get(k) == v for k, v in data_filters.items()):
                    item = {k: record[k] for k in record if k in key_figures or k in key_values}
                    if identifier_field and identifier_field in record:
                        item[identifier_field] = record[identifier_field]
                    processed_data.append(item)

            if processed_data:
                points = [
                    models.PointStruct(
                        id=hashlib.md5(json.dumps(r).encode()).hexdigest(),
                        vector=model.encode(json.dumps(r)).tolist(),
                        payload=r
                    ) for r in processed_data
                ]
                qdrant_client.upsert(collection_name="mes_logs", points=points)

        logger.info(f"Fetched {len(processed_data)} records for {key_values}")
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
    """Analiza el cumplimiento de los datos MES contra reglas SOP y personalizadas.

    Esta herramienta realiza las siguientes acciones:
    1. Valida los campos solicitados (key_figures y key_values) usando DataValidator.
    2. Obtiene datos de la API MES o Qdrant usando fetch_mes_data.
    3. Carga reglas SOP y personalizadas para las máquinas relevantes.
    4. Filtra los datos por todos los key_values proporcionados.
    5. Compara cada registro contra todas las reglas, aplicando filtros de key_values.
    6. Calcula un porcentaje de cumplimiento por registro (basado solo en reglas SOP).
    7. Devuelve un informe detallado con resultados de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud FastMCP.
        key_values (Optional[Dict[str, str]]): Diccionario de campos categóricos y valores
            para filtrar (Ejemplo, {"machine": "ModelA", "production_line": "Line3", "start_date": "2025-04-09",
            "end_date": "2025-04-11"}). Las fechas deben estar en formato YYYY-MM-DD.
        key_figures (Optional[List[str]]): Lista de campos numéricos a analizar
            (Ejemplo, ["temperature", "uptime"]).
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
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Data retrieval failed"),
                "results": []
            }, ensure_ascii=False)

        # Filtrar dinámicamente por todos los key_values (excepto fechas)
        filter_fields = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}
        if filter_fields:
            fetch_result["data"] = [
                r for r in fetch_result["data"]
                if all(r.get(k) == v for k, v in filter_fields.items())
            ]
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
                "metrics": {k: record[k] for k in key_figures if k in record},
                "compliance": {},
                "compliance_percentage": 0.0,
                "date": record.get("date", ""),
            }
            # Incluir todos los key_values en los resultados
            for k in valid_values:
                if k in record:
                    analysis[k] = record[k]
            if identifier_field and identifier_field in record:
                analysis[identifier_field] = record[identifier_field]
            
            compliant = 0
            total = 0
            
            for field in key_figures:
                if field not in record:
                    logger.warning(f"Field {field} not in record: {record}")
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
            "analysis_notes": []
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