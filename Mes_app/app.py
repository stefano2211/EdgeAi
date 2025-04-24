from fastapi import FastAPI, HTTPException, UploadFile, File,Form,Query
import sqlite3
from pydantic import BaseModel
from datetime import datetime
import PyPDF2
import sqlite3
import json
from typing import Optional, List
import httpx

app = FastAPI()

class ProductionMetrics(BaseModel):
    quantity: int
    product_type: str

class Event(BaseModel):
    event_type: str  # Ejemplo: "machine_failure", "production_alert"
    description: str
    timestamp: datetime
    equipment: str

# Modelos de datos
class SensorData(BaseModel):
    temperature: float
    pressure: float
    vibration: float

class ComplianceRules(BaseModel):
    temperature_limit: float
    pressure_limit: float
    operator_certification_required: bool
    process_notes: Optional[str] = None

class ContextualInfo(BaseModel):
    compliance_rules: ComplianceRules

class MachineRecord(BaseModel):
    transaction_id: str
    work_order_id: str
    timestamp: datetime
    equipment: str
    operator: str
    sensor_data: SensorData
    contextual_info: ContextualInfo
    production_metrics: ProductionMetrics  # Nuevo campo

# Inicialización de la base de datos
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL UNIQUE,
            work_order_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            equipment TEXT NOT NULL,
            operator TEXT NOT NULL,
            sensor_data TEXT NOT NULL,
            contextual_info TEXT NOT NULL,
            production_metrics TEXT NOT NULL 
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            description TEXT NOT NULL,
            upload_timestamp DATETIME NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()

# Endpoints para máquinas
@app.post("/machines/")
async def create_machine_record(record: MachineRecord):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO machines (
                transaction_id, work_order_id, timestamp, equipment,
                operator, sensor_data, contextual_info, production_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.transaction_id,
            record.work_order_id,
            record.timestamp.isoformat(),
            record.equipment,
            record.operator,
            record.sensor_data.json(),
            record.contextual_info.json(),
            record.production_metrics.json()
        ))
        conn.commit()
        return {"message": "Registro creado exitosamente"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Transaction ID ya existe")
    finally:
        conn.close()


@app.post("/events/")
async def create_event(event: Event):
    # Conectar al MCP (asumimos que corre en http://mcp:8000)
    MCP_URL = "http://mcp:8000/process_event"  # Ajusta según tu configuración
    
    async with httpx.AsyncClient() as client:
        try:
            # Enviar el evento al MCP
            response = await client.post(
                MCP_URL,
                json={
                    "event_type": event.event_type,
                    "description": event.description,
                    "timestamp": event.timestamp.isoformat(),
                    "equipment": event.equipment
                }
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Error al enviar evento al MCP")
            return {"message": "Evento enviado al MCP exitosamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/machines/")
async def get_all_machines(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    base_query = """
        SELECT transaction_id, work_order_id, timestamp, equipment,
               operator, sensor_data, contextual_info, production_metrics
        FROM machines
    """
    
    conditions = []
    params = []
    
    if specific_date:
        conditions.append("DATE(timestamp) = ?")
        params.append(specific_date)
    else:
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
    
    query = base_query
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    
    machines = []
    for row in cursor.fetchall():
        machines.append({
            "transaction_id": row[0],
            "work_order_id": row[1],
            "timestamp": row[2],
            "equipment": row[3],
            "operator": row[4],
            "sensor_data": json.loads(row[5]),
            "contextual_info": json.loads(row[6]),
            "production_metrics": json.loads(row[7])
        })
    conn.close()
    return machines

@app.get("/machines/{equipment}")
async def get_machine_records(
    equipment: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    query = """
        SELECT transaction_id, work_order_id, timestamp, equipment, operator,
               sensor_data, contextual_info, production_metrics
        FROM machines 
        WHERE equipment = ?
    """
    
    params = [equipment]
    conditions = []
    
    if specific_date:
        conditions.append("DATE(timestamp) = ?")
        params.append(specific_date)
    else:
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    
    records = []
    for row in cursor.fetchall():
        records.append({
            "transaction_id": row[0],
            "work_order_id": row[1],
            "timestamp": row[2],
            "equipment": row[3],  # Asegurando que el campo equipment esté incluido
            "operator": row[4],
            "sensor_data": json.loads(row[5]),
            "contextual_info": json.loads(row[6]),
            "production_metrics": json.loads(row[7])
        })
    conn.close()
    
    if not records:
        raise HTTPException(status_code=404, detail="Equipo no encontrado")
    
    return records


# Endpoint para obtener todos los PDFs (sin búsqueda inteligente)
@app.post("/pdfs/")
async def upload_pdf(file: UploadFile = File(...), description: str = Form(None)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    # Leer y extraer el contenido del PDF
    pdf_reader = PyPDF2.PdfReader(file.file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() or ""
    
    # Guardar en la base de datos con descripción
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO pdfs (filename, content, description, upload_timestamp)
        VALUES (?, ?, ?, ?)
    """, (file.filename, content, description, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return {"message": f"PDF '{file.filename}' subido y procesado exitosamente"}

@app.get("/pdfs/list")
async def list_pdfs():
    """Devuelve lista de PDFs con sus descripciones"""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, description FROM pdfs ORDER BY upload_timestamp DESC")
    pdfs = [{"filename": row[0], "description": row[1]} for row in cursor.fetchall()]
    conn.close()
    return pdfs



@app.get("/pdfs/content/")
async def get_pdf_contents(
    filenames: List[str] = Query(..., description="Nombres de los archivos PDF a consultar"),
    max_length: Optional[int] = Query(None, description="Longitud máxima del contenido")
):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    # Normalizar nombres de archivos
    normalized_filenames = [f.strip() for f in filenames]
    
    placeholders = ','.join(['?'] * len(normalized_filenames))
    
    cursor.execute(f"""
        SELECT filename, content, description 
        FROM pdfs 
        WHERE LOWER(TRIM(filename)) IN ({placeholders})
    """, [f.lower() for f in normalized_filenames])
    
    results = []
    for row in cursor.fetchall():
        filename, content, description = row
        results.append({
            "filename": filename,
            "description": description or "Sin descripción",
            "content": content[:max_length] if max_length else content,
            "content_length": len(content),
            "truncated": max_length is not None and len(content) > max_length
        })
    
    conn.close()
    
    if not results:
        found_files = cursor.execute("SELECT filename FROM pdfs").fetchall()
        raise HTTPException(
            status_code=404,
            detail={
                "message": "PDFs no encontrados",
                "requested": normalized_filenames,
                "available": [f[0] for f in found_files]
            }
        )
    
    return {
        "count": len(results),
        "requested_files": normalized_filenames,
        "pdfs": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)