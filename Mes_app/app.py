from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
import sqlite3
from pydantic import BaseModel
from datetime import datetime
import PyPDF2
import json
from typing import Optional, List
import httpx

app = FastAPI()

class Event(BaseModel):
    event_type: str  # Ejemplo: "machine_failure", "production_alert"
    description: str
    timestamp: datetime
    equipment: str

class MachineRecord(BaseModel):
    date: str  # Ejemplo: "2025-04-10"
    machine: str  # Ejemplo: "ModelA"
    production_line: str  # Ejemplo: "Line1"
    material: str  # Ejemplo: "Steel"
    uptime: float  # Ejemplo: 95.0
    defects: int  # Ejemplo: 2
    vibration: float  # Ejemplo: 0.5
    temperature: float  # Ejemplo: 75.2
    defect_type: str  # Ejemplo: "surface_scratch"
    throughput: float  # Ejemplo: 100.0
    inventory_level: int  # Ejemplo: 500

# Inicialización de la base de datos con registros fijos
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    # Crear tabla machines
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            machine TEXT NOT NULL,
            production_line TEXT NOT NULL,
            material TEXT NOT NULL,
            uptime REAL NOT NULL,
            defects INTEGER NOT NULL,
            vibration REAL NOT NULL,
            temperature REAL NOT NULL,
            defect_type TEXT NOT NULL,
            throughput REAL NOT NULL,
            inventory_level INTEGER NOT NULL
        )
    """)
    
    # Verificar si la tabla machines está vacía
    cursor.execute("SELECT COUNT(*) FROM machines")
    count = cursor.fetchone()[0]
    
    # Insertar 9 registros fijos si la tabla está vacía
    if count == 0:
        fixed_records = [
            ("2025-04-10", "ModelA", "Line1", "Steel", 87.0, 3, 0.8, 73.0, "surface_crack", 100.0, 500),
            ("2025-04-11", "ModelA", "Line2", "Aluminum", 95.0, 0, 0.5, 80.0, "none", 110.0, 400),
            ("2025-04-12", "ModelB", "Line3", "Copper", 90.0, 1, 0.9, 73.0, "surface_scratch", 105.0, 600),
            ("2025-04-13", "ModelB", "Line1", "Plastic", 95.0, 0, 0.5, 78.0, "surface_dent", 108.0, 450),
            ("2025-04-14", "ModelC", "Line2", "Steel", 88.0, 4, 0.2, 80.0, "surface_crack", 95.0, 550),
            ("2025-04-15", "ModelC", "Line3", "Aluminum", 97.0, 0, 0.9, 73.0, "none", 115.0, 380),
            ("2025-04-16", "ModelA", "Line1", "Copper", 85.0, 1, 0.2, 72.0, "surface_scratch", 98.0, 520),
            ("2025-04-17", "ModelB", "Line2", "Plastic", 91.0, 2, 0.4, 76.0, "surface_dent", 112.0, 470),
            ("2025-04-18", "ModelC", "Line3", "Steel", 89.0, 1, 0.6, 72.0, "surface_crack", 102.0, 510)
        ]
        
        cursor.executemany("""
            INSERT INTO machines (
                date, machine, production_line, material, uptime, defects,
                vibration, temperature, defect_type, throughput, inventory_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, fixed_records)
    
    # Crear tabla pdfs (sin cambios)
    cursor.execute("""
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

# Endpoint para eventos (sin cambios)
@app.post("/events/")
async def create_event(event: Event):
    MCP_URL = "http://mcp:8000/process_event"
    
    async with httpx.AsyncClient() as client:
        try:
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

# Endpoint para obtener todos los registros
@app.get("/machines/")
async def get_all_machines(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    base_query = """
        SELECT id, date, machine, production_line, material, uptime, defects,
               vibration, temperature, defect_type, throughput, inventory_level
        FROM machines
    """
    
    conditions = []
    params = []
    
    if specific_date:
        conditions.append("date = ?")
        params.append(specific_date)
    else:
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
    
    query = base_query
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date DESC"
    
    cursor.execute(query, params)
    
    machines = []
    for row in cursor.fetchall():
        machines.append({
            "id": row[0],
            "date": row[1],
            "machine": row[2],
            "production_line": row[3],
            "material": row[4],
            "uptime": row[5],
            "defects": row[6],
            "vibration": row[7],
            "temperature": row[8],
            "defect_type": row[9],
            "throughput": row[10],
            "inventory_level": row[11]
        })
    conn.close()
    return machines

# Endpoint para obtener registros por máquina
@app.get("/machines/{machine}")
async def get_machine_records(
    machine: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    
    query = """
        SELECT id, date, machine, production_line, material, uptime, defects,
               vibration, temperature, defect_type, throughput, inventory_level
        FROM machines 
        WHERE machine = ?
    """
    
    params = [machine]
    conditions = []
    
    if specific_date:
        conditions.append("date = ?")
        params.append(specific_date)
    else:
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += " ORDER BY date DESC"
    
    cursor.execute(query, params)
    
    records = []
    for row in cursor.fetchall():
        records.append({
            "id": row[0],
            "date": row[1],
            "machine": row[2],
            "production_line": row[3],
            "material": row[4],
            "uptime": row[5],
            "defects": row[6],
            "vibration": row[7],
            "temperature": row[8],
            "defect_type": row[9],
            "throughput": row[10],
            "inventory_level": row[11]
        })
    conn.close()
    
    if not records:
        raise HTTPException(status_code=404, detail="Máquina no encontrada")
    
    return records

# Endpoints para PDFs (sin cambios)
@app.post("/pdfs/")
async def upload_pdf(file: UploadFile = File(...), description: str = Form(None)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    pdf_reader = PyPDF2.PdfReader(file.file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() or ""
    
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