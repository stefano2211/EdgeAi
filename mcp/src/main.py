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

class TimeFilter(BaseModel):
    """Filter for date ranges in MES data queries."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        """Validates date formats and logical consistency."""
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

def init_infrastructure():
    """Initializes required infrastructure (Qdrant collections and MinIO bucket)."""
    try:
        # Qdrant configuration
        vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
        qdrant_client.recreate_collection(collection_name="mes_logs", vectors_config=vector_config)
        qdrant_client.recreate_collection(collection_name="sop_pdfs", vectors_config=vector_config)
        
        # MinIO configuration
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
            
    except Exception as e:
        logger.error(f"Infrastructure initialization failed: {str(e)}")
        raise

class DataValidator:
    """Validates key_figures and key_values against API data structure."""
    
    @staticmethod
    async def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict[str, str]) -> Dict:
        """
        Validates that requested fields exist in the API data structure.
        
        Args:
            ctx: MCP context
            key_figures: List of numerical fields to validate
            key_values: Dictionary of categorical filters to validate
            
        Returns:
            Dictionary with field information from API
            
        Raises:
            ValueError: If any field or value is invalid
        """
        fields_info = json.loads(await list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Could not validate fields against API")
            
        invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
        invalid_values = {
            k: v for k, v in key_values.items() 
            if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, [])
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
    """
    Lists all available fields and their unique values from the MES data.
    
    Returns:
        JSON string with structure:
        {
            "status": "success"|"error"|"no_data",
            "key_figures": [str],  # List of numerical fields
            "key_values": {         # Dictionary of categorical fields
                "field1": [values], 
                "field2": [values],
                ...
            },
            "message": str  # Optional error message
        }
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/machines/")
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
    key_figures: Optional[List[str]] = None,
    time_filter: Optional[Dict[str, str]] = None
) -> str:
    """
    Retrieves manufacturing data from MES system with flexible filtering.
    
    Args:
        ctx: MCP context
        key_values: Dictionary of categorical filters (e.g., {"machine": "CNC-1"})
        key_figures: List of numerical metrics to retrieve (e.g., ["uptime", "temperature"])
        time_filter: Date range filter (e.g., {"start_date": "2025-01-01", "end_date": "2025-01-31"})
    
    Returns:
        JSON string with structure:
        {
            "status": "success"|"error",
            "count": int,  # Number of records returned
            "data": [      # List of records
                {
                    "id": str,
                    "date": str,
                    "machine": str,
                    ...  # Other requested fields
                },
                ...
            ],
            "message": str  # Optional error message
        }
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validate requested fields
        await DataValidator.validate_fields(ctx, key_figures, key_values)
        
        # Build query parameters
        params = {}
        if time_filter:
            tf = TimeFilter(**time_filter)
            tf.validate_dates()
            params.update({
                k: v for k, v in {
                    "specific_date": tf.specific_date,
                    "start_date": tf.start_date,
                    "end_date": tf.end_date
                }.items() if v is not None
            })

        # Determine API endpoint
        endpoint = f"{API_URL}/machines/{key_values['machine']}" if "machine" in key_values else f"{API_URL}/machines/"

        # Fetch data
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

        # Filter and process records
        processed_data = []
        for record in data:
            if all(record.get(k) == v for k, v in key_values.items() if k != "machine"):
                item = {
                    "id": record["id"],
                    "date": record["date"],
                    "machine": record["machine"]
                }
                item.update({f: record[f] for f in key_figures if f in record})
                processed_data.append(item)

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
    """
    Loads and processes Standard Operating Procedures (SOP) for a specific machine.
    
    Args:
        ctx: MCP context
        machine: Machine identifier (e.g., "CNC-1")
    
    Returns:
        JSON string with structure:
        {
            "status": "success"|"exists"|"error",
            "machine": str,
            "rules": {       # Extracted compliance rules
                "metric1": {
                    "value": float,
                    "operator": ">="|"<=",
                    "unit": str,
                    "source_text": str
                },
                ...
            },
            "message": str  # Optional error message
        }
    """
    try:
        # Check if SOP already exists
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
async def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    time_filter: Optional[Dict[str, str]] = None
) -> str:
    """
    Analyzes manufacturing data compliance against SOP standards.
    
    Args:
        ctx: MCP context
        key_values: Dictionary of categorical filters (e.g., {"machine": "CNC-1"})
        key_figures: List of numerical metrics to analyze (e.g., ["uptime", "temperature"])
        time_filter: Date range filter (e.g., {"start_date": "2025-01-01", "end_date": "2025-01-31"})
    
    Returns:
        JSON string with structure:
        {
            "status": "success"|"error",
            "period": str,  # Analyzed time period
            "machine_filter": str,  # Machine filter applied
            "metrics_analyzed": [str],  # List of analyzed metrics
            "results": [     # Compliance analysis results
                {
                    "id": str,
                    "date": str,
                    "machine": str,
                    "metrics": {str: float},  # Metric values
                    "compliance": {   # Compliance status per metric
                        "metric1": {
                            "value": float,
                            "rule": str,
                            "status": "compliant"|"non_compliant"|"unknown"
                        },
                        ...
                    },
                    "compliance_percentage": float  # Overall compliance percentage
                },
                ...
            ],
            "sop_coverage": str,  # SOP coverage statistics
            "analysis_notes": [str],  # Interpretation guidelines
            "message": str  # Optional error message
        }
    """
    try:
        key_values = key_values or {}
        key_figures = key_figures or []
        
        # Validate requested fields
        fields_info = await DataValidator.validate_fields(ctx, key_figures, key_values)
        
        # Fetch manufacturing data
        fetch_result = json.loads(await fetch_mes_data(ctx, key_values, key_figures, time_filter))
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
                rules = machine_rules.get(record["machine"], {})
                
                if field in rules:
                    rule = rules[field]
                    value = record[field]
                    op = rule["operator"]
                    
                    is_compliant = (value >= rule["value"]) if op == ">=" else (value <= rule["value"])
                    analysis["compliance"][field] = {
                        "value": value,
                        "rule": f"{op} {rule['value']}{rule.get('unit', '')}",
                        "status": "compliant" if is_compliant else "non_compliant"
                    }
                    compliant += 1 if is_compliant else 0
                else:
                    analysis["compliance"][field] = {
                        "value": record[field],
                        "rule": "no_rule_defined",
                        "status": "unknown"
                    }
            
            if total > 0:
                analysis["compliance_percentage"] = round((compliant / total) * 100, 2)
            
            results.append(analysis)

        # Format time period description
        period = "all dates"
        if time_filter:
            tf = TimeFilter(**time_filter)
            tf.validate_dates()
            period = tf.specific_date or f"{tf.start_date} to {tf.end_date}"

        return json.dumps({
            "status": "success",
            "period": period,
            "machine_filter": key_values.get("machine", "all machines"),
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_coverage": f"{sum(1 for r in machine_rules.values() if r)}/{len(machines)} machines with SOP",
            "analysis_notes": [
                "compliant: Meets SOP requirement",
                "non_compliant: Does not meet SOP requirement",
                "unknown: No SOP rule defined"
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