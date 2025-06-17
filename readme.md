# EdgeAi Manufacturing Compliance Platform

Este proyecto implementa una plataforma de cumplimiento para manufactura, integrando procesamiento de datos MES, autenticación JWT, almacenamiento de documentos y reglas personalizadas usando microservicios en Python y contenedores Docker.

## Arquitectura

- **api/**: API principal FastAPI para autenticación de usuarios, gestión de sesiones y consulta de datos MES.
- **token_api/**: Servicio FastAPI para almacenamiento y validación de tokens JWT.
- **mcp/**: Manufacturing Compliance Processor. Orquesta la consulta de datos, reglas y SOPs, usando Qdrant, MinIO y Sentence Transformers.
- **qdrant**: Base vectorial para almacenamiento y consulta de embeddings.
- **minio**: Almacenamiento de objetos (PDFs de SOPs).
- **ollama** y **open-webui**: Opcional, para integración con modelos LLM y UI.

## Servicios

- **API MES** (`api/`):  
  - Registro y login de usuarios.
  - Endpoints protegidos para consulta de datos de máquinas.
- **Token API** (`token_api/`):  
  - Almacena y valida tokens JWT.
- **MCP** (`mcp/`):  
  - Procesa reglas de cumplimiento, consulta datos, integra Qdrant y MinIO.
- **Qdrant**:  
  - Almacena embeddings y reglas personalizadas.
- **MinIO**:  
  - Almacena documentos PDF de procedimientos operativos estándar (SOPs).

## Requisitos

- Docker y Docker Compose
- (Opcional) Python 3.11+ para desarrollo local

## Levantar el entorno

```sh
docker-compose up --build
```

Esto levantará todos los servicios en la red `mes-network`.

## Variables de entorno principales

- `JWT_SECRET_KEY`: Clave secreta para JWT (debe ser igual en `api` y `token_api`)
- `MINIO_*`: Configuración de MinIO
- `API_URL`, `TOKEN_API_URL`: URLs internas para comunicación entre servicios

## Flujo de autenticación

1. El usuario se registra o inicia sesión en la API principal (`/register`, `/login`).
2. El token JWT generado se almacena en el servicio `token_api` (`/store-token`).
3. MCP obtiene el token desde `token_api` para autenticar solicitudes a la API MES.

## Consultas MES y reglas

- MCP permite consultar datos MES, analizar cumplimiento contra SOPs y reglas personalizadas, y gestionar reglas usando Qdrant.
- Los SOPs se almacenan como PDFs en MinIO y se pueden consultar por máquina.

## Ejemplo de uso

1. Registrar usuario en la API principal.
2. Almacenar el token en `token_api`.
3. Usar MCP para consultar datos, analizar cumplimiento y gestionar reglas.

## Estructura del repositorio

```
api/
  app.py
  requirements.txt
  Dockerfile.api
token_api/
  token_api.py
  requirements.txt
  Dockerfile.token
mcp/
  src/main.py
  requirements.txt
  Dockerfile.mcp
docker-compose.yaml
```

