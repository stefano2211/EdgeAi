from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
import sqlite3
from pydantic import BaseModel
from datetime import datetime
import PyPDF2
import json
from typing import Optional, List
import httpx

app = FastAPI()


class MachineRecord(BaseModel):
    date: str  # Ejemplo: "2025-04-10"
    machine: str  # Ejemplo: "ModelA"
    production_line: str  # Ejemplo: "Line1"
    material: str  # Ejemplo: "Steel"
    batch_id: str  # Ejemplo: "BATCH101"
    uptime: float  # Ejemplo: 95.0
    defects: int  # Ejemplo: 2
    vibration: float  # Ejemplo: 0.5
    temperature: float  # Ejemplo: 75.2
    defect_type: str  # Ejemplo: "surface_scratch"
    throughput: float  # Ejemplo: 100.0
    inventory_level: int  # Ejemplo: 500

# Inicialización de la base de datos con registros fijos
def init_db():

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Crear tabla machines con batch_id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            machine TEXT NOT NULL,
            production_line TEXT NOT NULL,
            material TEXT NOT NULL,
            batch_id TEXT NOT NULL,
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
    
    # Insertar nuevos registros fijos si la tabla está vacía
    if count == 0:
        fixed_records = [
            ("2025-04-10", "ModelA", "Line1", "Steel", "BATCH101", 95.0, 2, 0.7, 75.0, "scratch", 90.0, 400),
            ("2025-04-10", "ModelB", "Line2", "Aluminum", "BATCH102", 97.5, 1, 0.6, 70.0, "dent", 88.0, 350),
            ("2025-04-11", "ModelC", "Line1", "Copper", "BATCH137", 87.0, 3, 0.8, 81.0, "crack", 85.0, 350),
            ("2025-04-10", "ModelD", "Line3", "Plastic", "BATCH104", 97.5, 4, 0.9, 78.0, "warp", 87.0, 300),
            ("2025-04-10", "ModelE", "Line2", "Brass", "BATCH105", 90.0, 2, 0.65, 72.0, "chip", 89.0, 320),
            ("2025-04-10", "ModelF", "Line1", "Titanium", "BATCH106", 95.0, 2, 0.75, 76.0, "scratch", 91.0, 380),
            ("2025-04-09", "ModelA", "Line1", "Steel", "BATCH107", 97.5, 1, 0.6, 74.0, "dent", 92.0, 410),
            ("2025-04-09", "ModelB", "Line2", "Aluminum", "BATCH108", 92.0, 3, 0.8, 79.0, "crack", 86.0, 340),
            ("2025-04-09", "ModelC", "Line1", "Copper", "BATCH109", 88.5, 4, 0.85, 82.0, "warp", 84.0, 360),
            ("2025-04-09", "ModelD", "Line3", "Plastic", "BATCH110", 90.0, 2, 0.7, 77.0, "chip", 88.0, 310),
            ("2025-04-09", "ModelE", "Line2", "Brass", "BATCH111", 99.0, 0, 0.5, 70.0, "none", 93.0, 330),
            ("2025-04-09", "ModelF", "Line1", "Titanium", "BATCH112", 85.5, 5, 0.9, 80.0, "scratch", 83.0, 370),
            ("2025-04-11", "ModelA", "Line1", "Steel", "BATCH113", 93.0, 1, 0.65, 73.0, "dent", 90.0, 390),
            ("2025-04-11", "ModelB", "Line2", "Aluminum", "BATCH114", 96.5, 2, 0.7, 71.0, "crack", 89.0, 360),
            ("2025-04-11", "ModelD", "Line3", "Plastic", "BATCH115", 89.0, 3, 0.8, 79.0, "warp", 86.0, 320),
        ]
        
        cursor.executemany("""
            INSERT INTO machines (
                date, machine, production_line, material, batch_id, uptime, defects,
                vibration, temperature, defect_type, throughput, inventory_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, fixed_records)
    
    # Crear tabla pdfs
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
        SELECT id, date, machine, production_line, material, batch_id, uptime, defects,
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
            "batch_id": row[5],
            "uptime": row[6],
            "defects": row[7],
            "vibration": row[8],
            "temperature": row[9],
            "defect_type": row[10],
            "throughput": row[11],
            "inventory_level": row[12]
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
        SELECT id, date, machine, production_line, material, batch_id, uptime, defects,
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
            "batch_id": row[5],
            "uptime": row[6],
            "defects": row[7],
            "vibration": row[8],
            "temperature": row[9],
            "defect_type": row[10],
            "throughput": row[11],
            "inventory_level": row[12]
        })
    conn.close()
    
    if not records:
        raise HTTPException(status_code=404, detail="Máquina no encontrada")
    
    return records

# Endpoints para PDFs
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