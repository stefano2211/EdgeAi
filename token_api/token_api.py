from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import jwt
import os
import logging
import uuid

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuración
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # Debe coincidir con la API principal
ALGORITHM = "HS256"
TOKEN_DB = "token_database.db"
TOKEN_EXPIRE_MINUTES = 60  # Tokens expiran en 60 minutos
DEFAULT_USER = "mcp_user"

# Modelo
class TokenRequest(BaseModel):
    token: str

# Inicialización de la base de datos
def init_db():
    """Inicializa la base de datos SQLite para almacenar tokens."""
    try:
        conn = sqlite3.connect(TOKEN_DB)
        cursor = conn.cursor()
        
        # Tabla para tokens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                stored_at DATETIME NOT NULL
            )
        """)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Token database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    """Ejecuta la inicialización de la base de datos al iniciar la aplicación."""
    init_db()

# Endpoint para almacenar token
@app.post("/store-token")
async def store_token(request: TokenRequest):
    """Almacena un token JWT proporcionado por el usuario."""
    try:
        # Validar el token
        try:
            payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                raise HTTPException(status_code=401, detail="Invalid token: No username")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        conn = sqlite3.connect(TOKEN_DB)
        cursor = conn.cursor()
        
        # Verificar si el token ya está almacenado
        cursor.execute("SELECT token FROM tokens WHERE token = ?", (request.token,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Token already stored")
        
        # Almacenar el token
        cursor.execute("INSERT INTO tokens (token, username, stored_at) VALUES (?, ?, ?)",
                       (request.token, username, datetime.utcnow().isoformat()))
        conn.commit()
        
        return {"message": "Token stored successfully", "username": username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para validar token
@app.post("/validate-token")
async def validate_token(request: TokenRequest):
    """Valida un token JWT y devuelve el username asociado."""
    try:
        token = request.token
        conn = sqlite3.connect(TOKEN_DB)
        cursor = conn.cursor()
        
        # Buscar el token en la base de datos
        cursor.execute("SELECT username FROM tokens WHERE token = ?", (token,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=401, detail="Token not found")
        
        username = result[0]
        
        # Validar el token con JWT
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("sub") != username:
                raise HTTPException(status_code=401, detail="Token username mismatch")
            if datetime.fromtimestamp(payload.get("exp")) < datetime.utcnow():
                raise HTTPException(status_code=401, detail="Token expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return {"username": username, "message": "Token is valid"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        conn.close()

# Endpoint para obtener un token
@app.get("/get-token")
async def get_token():
    """Genera un token JWT para el usuario predeterminado."""
    try:
        expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": DEFAULT_USER,
            "exp": expire,
            "jti": str(uuid.uuid4())
        }
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        # Almacenar token en la base de datos
        conn = sqlite3.connect(TOKEN_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tokens (token, username, stored_at) VALUES (?, ?, ?)",
                       (token, DEFAULT_USER, datetime.utcnow().isoformat()))
        conn.commit()
        
        logger.info("Token generated and stored successfully")
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Failed to generate token: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate token")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)