from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import jwt
import uuid
from typing import Optional, List
import os
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuración
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60

# Modelo para login
class LoginRequest(BaseModel):
    username: str
    password: str

# Esquema de autenticación
security = HTTPBearer()

# Inicialización de la base de datos
def init_db():
    """Inicializa la base de datos SQLite con tablas para equipos, sesiones y usuarios."""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        # Tabla para equipos (usando date en formato YYYY-MM-DD)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                machine TEXT NOT NULL,
                operation_mode TEXT NOT NULL,
                product_type TEXT NOT NULL,
                cycle_time REAL NOT NULL,
                error_count INTEGER NOT NULL,
                pressure REAL NOT NULL,
                power_consumption REAL NOT NULL,
                status TEXT NOT NULL,
                output_rate REAL NOT NULL
            )
        """)
        
        # Tabla para tokens de sesión
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                expiry DATETIME NOT NULL
            )
        """)
        
        # Tabla para usuarios
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", 
                       ("admin", "password123"))
        
        # Insertar registros fijos para equipos
        cursor.execute("SELECT COUNT(*) FROM machines")
        count = cursor.fetchone()[0]
        if count == 0:
            fixed_records = [
                ("2025-05-20", "EquipA", "Auto", "WidgetA", 12.5, 1, 5.2, 15.0, "Running", 120.0),
                ("2025-05-20", "EquipB", "Manual", "WidgetB", 14.0, 2, 4.8, 16.5, "Running", 110.0),
                ("2025-05-20", "EquipC", "Auto", "WidgetC", 11.8, 0, 5.0, 14.5, "Running", 125.0),
                ("2025-05-21", "EquipA", "Auto", "WidgetA", 13.0, 3, 5.5, 15.8, "Stopped", 100.0),
                ("2025-05-21", "EquipB", "Auto", "WidgetB", 12.7, 1, 4.9, 15.2, "Running", 115.0),
                ("2025-05-21", "EquipD", "Manual", "WidgetD", 15.0, 4, 5.3, 17.0, "Stopped", 95.0),
                ("2025-05-22", "EquipA", "Auto", "WidgetA", 12.3, 0, 5.1, 14.8, "Running", 122.0),
                ("2025-05-22", "EquipC", "Auto", "WidgetC", 11.5, 2, 5.0, 14.0, "Running", 130.0),
                ("2025-05-22", "EquipB", "Manual", "WidgetB", 14.5, 1, 4.7, 16.0, "Running", 108.0),
                ("2025-05-22", "EquipD", "Auto", "WidgetD", 13.8, 2, 5.4, 16.2, "Running", 112.0),
            ]
            cursor.executemany("""
                INSERT INTO machines (
                    date, machine, operation_mode, product_type, cycle_time,
                    error_count, pressure, power_consumption, status, output_rate
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
    """Genera un token JWT para un usuario dado.

    Args:
        username (str): Nombre de usuario para incluir en el token.

    Returns:
        str: Token JWT generado.

    Raises:
        Exception: Si falla la generación del token o el almacenamiento en la base de datos.
    """
    try:
        expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        to_encode = {"sub": username, "exp": expire, "jti": str(uuid.uuid4())}
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Almacenar token en la base de datos
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (token, username, expiry) VALUES (?, ?, ?)",
                       (token, username, expire.isoformat()))
        conn.commit()
        return token
    except Exception as e:
        logger.error(f"Failed to create JWT token: {str(e)}")
        raise
    finally:
        conn.close()

# Validar token
async def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Valida un token JWT proporcionado en el encabezado de autorización.

    Args:
        credentials (HTTPAuthorizationCredentials): Credenciales extraídas del encabezado.

    Returns:
        str: Nombre de usuario asociado al token.

    Raises:
        HTTPException: Si el token es inválido, expirado o no existe.
    """
    try:
        token = credentials.credentials
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT username, expiry FROM sessions WHERE token = ?", (token,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        username, expiry = session
        if datetime.fromisoformat(expiry) < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token expired")
        
        # Verificar JWT
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint de login
@app.post("/login")
async def login(request: LoginRequest):
    """Autentica a un usuario y genera un token JWT.

    Args:
        request (LoginRequest): Objeto con username y password.

    Returns:
        dict: Diccionario con el token de acceso y el tipo de token.

    Raises:
        HTTPException: Si las credenciales son inválidas (401).
    """
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (request.username,))
        user = cursor.fetchone()
        
        if not user or user[0] != request.password:
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
    """Obtiene todos los registros de equipos, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        base_query = """
            SELECT id, date, machine, operation_mode, product_type, cycle_time,
                   error_count, pressure, power_consumption, status, output_rate
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
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "id": row[0],
                "date": row[1],
                "machine": row[2],
                "operation_mode": row[3],
                "product_type": row[4],
                "cycle_time": row[5],
                "error_count": row[6],
                "pressure": row[7],
                "power_consumption": row[8],
                "status": row[9],
                "output_rate": row[10]
            })
        return records
    except Exception as e:
        logger.error(f"Failed to fetch equipment records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

@app.get("/machines/{machine}")
async def get_machine_records(
    machine: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None,
    username: str = Depends(validate_token)
):
    """Obtiene registros para un equipo específico, opcionalmente filtrados por fechas."""
    try:
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        query = """
            SELECT id, date, machine, operation_mode, product_type, cycle_time,
                   error_count, pressure, power_consumption, status, output_rate
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
                "operation_mode": row[3],
                "product_type": row[4],
                "cycle_time": row[5],
                "error_count": row[6],
                "pressure": row[7],
                "power_consumption": row[8],
                "status": row[9],
                "output_rate": row[10]
            })
        
        if not records:
            raise HTTPException(status_code=404, detail="Equipo no encontrado")
        
        return records
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch equipment records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)