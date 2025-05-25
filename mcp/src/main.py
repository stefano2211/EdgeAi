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
        """Autentica contra la API y obtiene un token de sesión."""
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
        """Realiza una solicitud GET autenticada a la API."""
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
    """Inicializa la infraestructura del MCP, incluyendo Qdrant, MinIO y el cliente autenticado."""
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

# Recursos para datos de la API
@mcp.resource("api://machines/{machine}")
async def get_machine_data(machine: str) -> str:
    """Obtiene datos de una máquina específica desde la API."""
    try:
        response = await auth_client.get(f"/machines/{machine}")
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

# Recursos para PDFs en MinIO
@mcp.resource("minio://sop-pdfs/{machine}.pdf")
async def get_sop_pdf(machine: str) -> bytes:
    """Obtiene el PDF de un SOP desde MinIO."""
    try:
        response = minio_client.get_object(MINIO_BUCKET, f"{machine}.pdf")
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        raise ValueError(f"PDF no encontrado para {machine}: {str(e)}")

class DataValidator:
    """Valida key_figures y key_values contra la estructura de datos de la API."""
    
    @staticmethod
    def validate_date(date_str: str, field: str) -> None:
        """Valida que una fecha tenga el formato YYYY-MM-DD."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Formato inválido para {field}. Use YYYY-MM-DD: {date_str}")

    @staticmethod
    async def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict[str, str]) -> Dict:
        """Valida que los campos solicitados sean válidos según la estructura de la API."""
        try:
            fields_info = json.loads(await list_fields(ctx))
            if fields_info["status"] != "success":
                raise ValueError("No se pudo validar contra la API")

            if "start_date" in key_values or "end_date" in key_values:
                if "start_date" not in key_values or "end_date" not in key_values:
                    raise ValueError("Deben proporcionarse tanto start_date como end_date")
                DataValidator.validate_date(key_values["start_date"], "start_date")
                DataValidator.validate_date(key_values["end_date"], "end_date")
                if key_values["start_date"] > key_values["end_date"]:
                    raise ValueError("start_date no puede ser posterior a end_date")

            invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
            invalid_values = {
                k: v for k, v in key_values.items()
                if k not in fields_info["key_values"] and k not in ["start_date", "end_date"]
                or (k in fields_info["key_values"] and v not in fields_info["key_values"].get(k, []))
            }
            
            if invalid_figures or invalid_values:
                errors = []
                if invalid_figures:
                    errors.append(f"Campos numéricos inválidos: {invalid_figures}")
                if invalid_values:
                    errors.append(f"Campos categóricos inválidos: {invalid_values}")
                raise ValueError(" | ".join(errors))
            
            return fields_info
        except Exception as e:
            logger.error(f"Validación de campos falló: {str(e)}")
            raise

@mcp.tool()
async def list_fields(ctx: Context) -> str:
    """Lista los campos disponibles en los datos de la API MES."""
    try:
        response = await auth_client.get("/machines/")
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
        logger.error(f"Listado de campos falló: {str(e)}")
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
    """Obtiene datos de la API MES o Qdrant, filtra dinámicamente y almacena en Qdrant."""
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validar campos solicitados
        fields_info = await DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        
        # Determinar el campo identificador
        identifier_field = None
        identifier_value = None
        for field in valid_values:
            if field == "machine":
                identifier_field = field
                identifier_value = key_values.get(field)
                break
        if not identifier_field:
            for field in valid_values:
                if field not in ["start_date", "end_date", "date"]:
                    identifier_field = field
                    identifier_value = key_values.get(field)
                    break
        
        # Verificar en Qdrant
        must_conditions = []
        if identifier_field and identifier_value:
            must_conditions.append(models.FieldCondition(key=identifier_field, match=models.MatchValue(value=identifier_value)))
        if "start_date" in key_values and "end_date" in key_values:
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchText(text=f"[{key_values['start_date']} TO {key_values['end_date']}]")
            ))
        for k, v in key_values.items():
            if k not in [identifier_field, "start_date", "end_date"]:
                must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        qdrant_results = qdrant_client.scroll(
            collection_name="mes_logs",
            scroll_filter=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )

        processed_data = [r.payload for r in qdrant_results[0]] if qdrant_results[0] else []

        # Si no hay datos en Qdrant, intentar usar recurso
        if not processed_data:
            params = {}
            if "start_date" in key_values and "end_date" in key_values:
                params.update({
                    "start_date": key_values["start_date"],
                    "end_date": key_values["end_date"]
                })
            data_filters = {k: v for k, v in key_values.items() if k not in ["start_date", "end_date"]}

            try:
                # Intentar usar recurso
                resource_uri = f"api://machines/{identifier_value}" if identifier_field and identifier_value else "api://machines/all"
                resource = await ctx.get_resource(resource_uri)
                data = json.loads(resource.read().decode())
            except AttributeError as e:
                # Fallback a solicitud directa si get_resource no está disponible
                logger.warning(f"ctx.get_resource no disponible, usando solicitud directa: {str(e)}")
                endpoint = f"/machines/{identifier_value}" if identifier_field and identifier_value else "/machines/"
                response = await auth_client.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()

            # Filtrar y procesar registros
            processed_data = []
            for record in data:
                if all(record.get(k) == v for k, v in data_filters.items() if k != identifier_field):
                    item = {
                        "id": record["id"],
                        "date": record["date"],
                        identifier_field: record[identifier_field] if identifier_field else None
                    }
                    item.update({f: record[f] for f in key_figures if f in record})
                    processed_data.append(item)

            # Almacenar en Qdrant
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
        # Verificar si ya existe en Qdrant
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

        # Intentar cargar PDF como recurso
        try:
            resource_uri = f"minio://sop-pdfs/{machine}.pdf"
            resource = await ctx.get_resource(resource_uri)
            if resource.content_type != "application/pdf":
                raise ValueError("El recurso debe ser un PDF")
            pdf_data = resource.read()
        except (AttributeError, ValueError) as e:
            # Fallback a carga directa desde MinIO
            logger.warning(f"ctx.get_resource no disponible o inválido, usando MinIO directo: {str(e)}")
            try:
                response = minio_client.get_object(MINIO_BUCKET, f"{machine}.pdf")
                pdf_data = response.read()
                response.close()
                response.release_conn()
            except S3Error as e:
                available_pdfs = [obj.object_name for obj in minio_client.list_objects(MINIO_BUCKET)]
                return json.dumps({
                    "status": "error",
                    "message": f"PDF no encontrado para {machine}. SOPs disponibles: {', '.join(available_pdfs)}",
                    "machine": machine
                }, ensure_ascii=False)

        # Extraer texto del PDF
        with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
            content = "\n".join(page.extract_text() or "" for page in pdf.pages)

        # Extraer reglas de cumplimiento
        rules = {}
        patterns = [
            (r"(?P<field>uptime|tiempo de actividad)\s*(?P<operator>>=|≥)\s*(?P<value>\d+\.\d+)\s*%", ">=", "%"),
            (r"(?P<field>temperature|temperatura)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*°C", "<=", "°C"),
            (r"(?P<field>vibration|vibración)\s*(?P<operator><=|≤)\s*(?P<value>\d+\.\d+)\s*mm/s", "<=", "mm/s"),
            (r"(?P<field>defects|defectos)\s*(?P<operator><=|≤)\s*(?P<value>\d+)", "<=", "")
        ]

        for pattern, default_op, unit in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    field = match.group("field").lower().replace(" ", "_")
                    rules[field] = {
                        "value": float(match.group("value")),
                        "operator": match.group("operator") or default_op,
                        "unit": unit,
                        "source_text": match.group(0)
                    }
                except Exception:
                    continue

        if not rules:
            return json.dumps({
                "status": "error",
                "message": "No se encontraron reglas de cumplimiento en el documento",
                "machine": machine
            }, ensure_ascii=False)

        # Almacenar en Qdrant
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

        return json.dumps({
            "status": "success",
            "machine": machine,
            "rules": rules
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Procesamiento de SOP falló para {machine}: {str(e)}")
        available_pdfs = [obj.object_name for obj in minio_client.list_objects(MINIO_BUCKET)]
        return json.dumps({
            "status": "error",
            "message": f"Error procesando SOP: {str(e)}. SOPs disponibles: {', '.join(available_pdfs)}",
            "machine": machine
        }, ensure_ascii=False)

@mcp.tool()
async def add_custom_rule(
    ctx: Context,
    machines: Union[List[str], str],
    key_figures: Union[Dict[str, float], str],
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
        machines (Union[List[str], str]): Lista de máquinas o string JSON.
            Ejemplo válido: ["ModelA"] o '["ModelA", "ModelB"]'
        key_figures (Union[Dict[str, float], str]): Métricas y valores umbral.
            Ejemplos válidos:
            - {"temperature": 70.0, "pressure": 1.2}
            - "temperature=70,pressure=1.2"
            - "temperature:70,pressure:1.2"
        key_values (Optional[Dict[str, str]]): Filtros categóricos.
            Ejemplo: {"material": "Steel", "batch": "A123"}
        operator (str): Operador de comparación (>=, <=, >, <, ==, !=).
        unit (Optional[str]): Unidad de medida común para todas las métricas.
        description (str): Descripción de la regla.

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
    """
    try:
        # Parsear máquinas
        if isinstance(machines, str):
            try:
                machines = json.loads(machines)
            except json.JSONDecodeError:
                machines = [machines.strip()]

        if not isinstance(machines, list):
            raise ValueError("machines debe ser una lista de strings")

        # Parsear key_figures
        if isinstance(key_figures, str):
            try:
                parsed_figures = json.loads(key_figures)
                if not isinstance(parsed_figures, dict):
                    raise ValueError("key_figures JSON debe ser un diccionario")
                key_figures = parsed_figures
            except json.JSONDecodeError:
                parsed_figures = {}
                for pair in key_figures.split(','):
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
            parsed_figures = {k: float(v) for k, v in key_figures.items()}
            key_figures = parsed_figures
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
            raise ValueError(f"Máquinas inválidas: {invalid_machines}. Máquinas válidas: {valid_machines}")

        # Validar métricas
        invalid_metrics = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_metrics:
            raise ValueError(f"Métricas inválidas: {invalid_metrics}. Métricas válidas: {fields_info['key_figures']}")

        # Validar operador
        valid_operators = [">=", "<=", ">", "<", "==", "!="]
        if operator not in valid_operators:
            raise ValueError(f"Operador inválido. Use uno de: {valid_operators}")

        # Validar key_values
        if key_values:
            for k, v in key_values.items():
                if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, []):
                    raise ValueError(f"Filtro inválido: {k}={v}. Valores válidos para {k}: {fields_info['key_values'].get(k, [])}")

        # Preparar regla final
        final_rule = {
            "machines": machines,
            "key_figures": key_figures,
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
        logger.error(f"Error al listar reglas: {str(e)}")
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
                "message": f"Regla con ID {rule_id} no encontrada"
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
        logger.error(f"Error al eliminar regla {rule_id}: {str(e)}")
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
            para filtrar (Example, {"machine": "ModelA", "material": "Steel", "production_line":"Line1", "start_date": "2025-04-09",
            "end_date": "2025-04-11"}). Las fechas deben estar en formato YYYY-MM-DD.
        key_figures (Optional[List[str]]): Lista de campos numéricos a analizar
            (Example, ["temperature", "uptime"]).
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validar campos solicitados
        fields_info = await DataValidator.validate_fields(ctx, key_figures, key_values)
        valid_values = fields_info["key_values"]
        
        # Determinar el campo identificador
        identifier_field = None
        identifier_value = None
        for field in valid_values:
            if field == "machine":
                identifier_field = field
                identifier_value = key_values.get(field)
                break
        if not identifier_field:
            for field in valid_values:
                if field not in ["start_date", "end_date", "date"]:
                    identifier_field = field
                    identifier_value = key_values.get(field)
                    break
        
        # Obtener datos de manufactura
        fetch_result = json.loads(await fetch_mes_data(ctx, key_values, key_figures))
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Data retrieval failed"),
                "results": []
            }, ensure_ascii=False)

        # Cargar reglas SOP para las máquinas relevantes
        machines = {r[identifier_field] for r in fetch_result["data"] if identifier_field}
        machine_rules = {}
        for machine in machines:
            sop_result = json.loads(await load_sop(ctx, machine))
            machine_rules[machine] = sop_result.get("rules", {}) if sop_result["status"] in ["success", "exists"] else {}

        # Cargar reglas personalizadas desde Qdrant
        custom_rules = []
        custom_result = qdrant_client.scroll(
            collection_name="custom_rules",
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="machines", match=models.MatchAny(any=list(machines)))
            ]),
            limit=100
        )
        custom_rules = [r.payload for r in custom_result[0]] if custom_result[0] else []

        # Analizar cumplimiento para cada registro
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "id": record["id"],
                "date": record["date"],
                "machine": record[identifier_field] if identifier_field else None,
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
                
                # Verificar regla SOP
                sop_rules = machine_rules.get(record[identifier_field] if identifier_field else "", {})
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
                
                # Verificar reglas personalizadas
                compliance_info["custom"] = []
                for rule in custom_rules:
                    if field not in rule["key_figures"]:
                        continue
                    if identifier_field and record[identifier_field] not in rule["machines"]:
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

        # Formatear descripción del período
        period = "all dates"
        if "start_date" in key_values and "end_date" in key_values:
            period = f"{key_values['start_date']} to {key_values['end_date']}"

        return json.dumps({
            "status": "success",
            "period": period,
            "machine_filter": identifier_value if identifier_value else "all machines",
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sum(1 for r in machine_rules.values() if r)}/{len(machines)} machines with SOP",
            "custom_rules_applied": f"{len(custom_rules)} custom rules applied",
            "analysis_notes": [
                "sop: Reglas de Procedimientos Operativos Estándar",
                "custom: Reglas definidas por el usuario",
                "compliant: Cumple con el requisito",
                "non_compliant: No cumple con el requisito",
                "unknown: No hay regla definida"
            ]
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Análisis de cumplimiento falló: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "results": []
        }, ensure_ascii=False)

if __name__ == "__main__":
    init_infrastructure()
    mcp.run()