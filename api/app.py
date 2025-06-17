from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import jwt
import uuid
from typing import Optional, List
import os
import logging
import bcrypt

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuración
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # Cambia en producción
ALGORITHM = "HS256"

# Modelos
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

# Esquema de autenticación
security = HTTPBearer()

# Inicialización de la base de datos
def init_db():
    """Inicializa la base de datos SQLite con tablas para máquinas, sesiones y usuarios."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Tabla para máquinas con nuevas columnas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                machine TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                operation_mode TEXT NOT NULL,
                energy_consumption REAL NOT NULL,
                production_rate REAL NOT NULL,
                error_code TEXT NOT NULL,
                maintenance_status TEXT NOT NULL,
                cycle_time REAL NOT NULL,
                quality_score INTEGER NOT NULL
            )
        """)
        
        # Tabla para tokens de sesión (sin cambios)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expiry DATETIME
            )
        """)
        
        # Tabla para usuarios (sin cambios)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
        
        # Insertar registros fijos para máquinas con datos nuevos
        cursor.execute("SELECT COUNT(*) FROM machines")
        count = cursor.fetchone()[0]
        if count == 0:
            fixed_records = [
                ("2025-05-01", "MachineX1", "OP001", "Automático", 120.5, 150.0, "Ninguno", "Completado", 45.2, 92),
                ("2025-05-01", "MachineX2", "OP002", "Manual", 130.0, 140.0, "E001", "Pendiente", 50.1, 88),
                ("2025-05-02", "MachineY1", "OP003", "Automático", 115.7, 160.0, "Ninguno", "No requerido", 42.8, 95),
                ("2025-05-02", "MachineY2", "OP001", "Mantenimiento", 80.0, 0.0, "E002", "Completado", 0.0, 0),
                ("2025-05-03", "MachineX1", "OP004", "Automático", 125.3, 155.0, "Ninguno", "Completado", 46.5, 90),
                ("2025-05-03", "MachineZ1", "OP005", "Manual", 135.2, 145.0, "E003", "Pendiente", 48.9, 85),
                ("2025-05-04", "MachineX2", "OP002", "Automático", 122.1, 152.0, "Ninguno", "No requerido", 44.7, 93),
                ("2025-05-04", "MachineY1", "OP003", "Automático", 118.9, 158.0, "Ninguno", "Completado", 43.2, 94),
                ("2025-05-05", "MachineZ2", "OP006", "Manual", 140.0, 142.0, "E001", "Pendiente", 49.5, 87),
                ("2025-05-05", "MachineX1", "OP001", "Automático", 123.4, 149.0, "Ninguno", "Completado", 45.8, 91),
                ("2025-05-06", "MachineY2", "OP004", "Automático", 119.2, 157.0, "Ninguno", "No requerido", 43.9, 96),
                ("2025-05-06", "MachineZ1", "OP005", "Mantenimiento", 85.0, 0.0, "E002", "Completado", 0.0, 0),
                ("2025-05-07", "MachineX2", "OP002", "Automático", 121.8, 151.0, "Ninguno", "Completado", 44.5, 92),
                ("2025-05-07", "MachineY1", "OP003", "Manual", 132.5, 143.0, "E004", "Pendiente", 47.8, 89),
                ("2025-05-08", "MachineZ2", "OP006", "Automático", 124.6, 156.0, "Ninguno", "No requerido", 45.1, 94),
            ]
            cursor.executemany("""
                INSERT INTO machines (
                    date, machine, operator_id, operation_mode, energy_consumption,
                    production_rate, error_code, maintenance_status, cycle_time, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, fixed_records)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    """Ejecuta la inicialización de la base de datos al iniciar la aplicación."""
    init_db()

# Generar token JWT
def create_jwt_token(username: str) -> str:
    """Genera un token JWT para un usuario dado sin tiempo de expiración."""
    try:
        to_encode = {"sub": username, "jti": str(uuid.uuid4())}
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Almacenar token en la base de datos con expiry NULL
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (token, username, expiry) VALUES (?, ?, ?)",
                       (token, username, None))
        conn.commit()
        return token
    except Exception as e:
        logger.error(f"Failed to create JWT token: {str(e)}")
        raise
    finally:
        conn.close()

# Validar token
async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Valida un token JWT proporcionado en el encabezado de autorización."""
    try:
        token = credentials.credentials
        # Validar solo con JWT, sin verificar expiración
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token: No username")
        return username
    except jwt.PyJWTError as e:
        logger.error(f"JWT validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or malformed token")
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Endpoint de registro
@app.post("/register")
async def register(request: RegisterRequest):
    """Registra un nuevo usuario en la base de datos."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Verificar si el usuario ya existe
        cursor.execute("SELECT username FROM users WHERE username = ?", (request.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Hashear la contraseña
        hashed_password = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Insertar nuevo usuario
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                       (request.username, hashed_password))
        conn.commit()
        
        # Generar token para el nuevo usuario
        token = create_jwt_token(request.username)
        
        return {"access_token": token, "token_type": "bearer", "message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint de login
@app.post("/login")
async def login(request: LoginRequest):
    """Autentica a un usuario y genera un token JWT."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (request.username,))
        user = cursor.fetchone()
        
        if not user or not bcrypt.checkpw(request.password.encode('utf-8'), user[0].encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = create_jwt_token(request.username)
        return {"access_token": token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener todos los registros (protegido)
@app.get("/machines/")
async def get_all_machines(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene todos los registros de máquinas, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        base_query = """
            SELECT id, date, machine, operator_id, operation_mode, energy_consumption,
                   production_rate, error_code, maintenance_status, cycle_time, quality_score
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
                "operator_id": row[3],
                "operation_mode": row[4],
                "energy_consumption": row[5],
                "production_rate": row[6],
                "error_code": row[7],
                "maintenance_status": row[8],
                "cycle_time": row[9],
                "quality_score": row[10]
            })
        return machines
    except Exception as e:
        logger.error(f"Failed to fetch machines: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener registros por máquina (protegido)
@app.get("/machines/{machine}")
async def get_machine_records(
    machine: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene registros para una máquina específica, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        query = """
            SELECT id, date, machine, operator_id, operation_mode, energy_consumption,
                   production_rate, error_code, maintenance_status, cycle_time, quality_score
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
                "operator_id": row[3],
                "operation_mode": row[4],
                "energy_consumption": row[5],
                "production_rate": row[6],
                "error_code": row[7],
                "maintenance_status": row[8],
                "cycle_time": row[9],
                "quality_score": row[10]
            })
        
        if not records:
            raise HTTPException(status_code=404, detail="Máquina no encontrada")
        
        return records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch machine records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)