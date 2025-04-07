from fastapi import FastAPI, HTTPException, UploadFile, File
import sqlite3
from pydantic import BaseModel
from datetime import datetime
import PyPDF2

app = FastAPI()

# Modelo para los datos de entrada de máquinas
class MachineRecord(BaseModel):
    maquina_nombre: str
    status: str
    temperatura: float
    timestamp: datetime

# Inicializar la base de datos con tablas para máquinas y PDFs
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            maquina_nombre TEXT NOT NULL,
            status TEXT NOT NULL,
            temperatura REAL NOT NULL,
            timestamp DATETIME NOT NULL
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

# Endpoint para insertar un registro de máquina
@app.post("/machines/")
async def create_machine_record(record: MachineRecord):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO machines (maquina_nombre, status, temperatura, timestamp)
        VALUES (?, ?, ?, ?)
    """, (record.maquina_nombre, record.status, record.temperatura, record.timestamp.isoformat()))
    conn.commit()
    conn.close()
    return {"message": "Registro creado exitosamente"}

# Endpoint para obtener todos los registros de máquinas
@app.get("/machines/")
async def get_all_machines():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT maquina_nombre, status, temperatura, timestamp FROM machines ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"maquina_nombre": row[0], "status": row[1], "temperatura": row[2], "timestamp": row[3]} for row in rows]

# Endpoint para obtener registros por máquina
@app.get("/machines/{maquina_nombre}")
async def get_machine_records(maquina_nombre: str):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT maquina_nombre, status, temperatura, timestamp 
        FROM machines 
        WHERE maquina_nombre = ? 
        ORDER BY timestamp DESC
    """, (maquina_nombre,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        raise HTTPException(status_code=404, detail="Máquina no encontrada")
    return [{"maquina_nombre": row[0], "status": row[1], "temperatura": row[2], "timestamp": row[3]} for row in rows]

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