from fastapi import FastAPI, HTTPException, UploadFile, File
import sqlite3
from pydantic import BaseModel
from datetime import datetime
import PyPDF2
import sqlite3
from datetime import datetime
import json
from typing import Optional
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
            record.production_metrics.json()  # Nuevo campo
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
async def get_all_machines():
    """Obtiene TODOS los registros de máquinas sin límite"""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transaction_id, work_order_id, timestamp, equipment,
               operator, sensor_data, contextual_info, production_metrics
        FROM machines ORDER BY timestamp DESC
    """)  # Eliminamos el LIMIT
    
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
async def get_machine_records(equipment: str):
    """Obtiene TODOS los registros de un equipo específico sin límite"""
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transaction_id, work_order_id, timestamp, operator,
               sensor_data, contextual_info, production_metrics
        FROM machines 
        WHERE equipment = ? 
        ORDER BY timestamp DESC
    """, (equipment,))  # Eliminamos el LIMIT
    
    records = []
    for row in cursor.fetchall():
        records.append({
            "transaction_id": row[0],
            "work_order_id": row[1],
            "timestamp": row[2],
            "operator": row[3],
            "sensor_data": json.loads(row[4]),
            "contextual_info": json.loads(row[5]),
            "production_metrics": json.loads(row[6])
        })
    conn.close()
    
    if not records:
        raise HTTPException(status_code=404, detail="Equipo no encontrado")
    return records

# Endpoint para subir un PDF
@app.post("/pdfs/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    # Leer y extraer el contenido del PDF
    pdf_reader = PyPDF2.PdfReader(file.file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() or ""
    
    # Guardar en la base de datos
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO pdfs (filename, content, upload_timestamp)
        VALUES (?, ?, ?)
    """, (file.filename, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return {"message": f"PDF '{file.filename}' subido y procesado exitosamente"}

# Endpoint para obtener todos los PDFs (sin búsqueda inteligente)
@app.get("/pdfs/")
async def get_all_pdfs():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content, upload_timestamp FROM pdfs ORDER BY upload_timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"filename": row[0], "content": row[1], "upload_timestamp": row[2]} for row in rows]

# Endpoint para obtener un PDF por nombre de archivo
@app.get("/pdfs/{filename}")
async def get_pdf_content(filename: str):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content, upload_timestamp FROM pdfs WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    return {"filename": row[0], "content": row[1], "upload_timestamp": row[2]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)