import httpx
from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch

# Configuración del servidor MCP
mcp = FastMCP("Mess Data")
API_URL = "http://api:5000"

# Cargar el modelo de embeddings al inicio (se ejecuta una vez)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo ligero y eficiente

# Herramienta para obtener el estado de todas las máquinas
@mcp.tool()
async def get_machines_status(ctx: Context) -> str:
    """Devuelve el estado actual de todas las máquinas"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = response.json()
        return "\n".join([f"{m['maquina_nombre']}: {m['status']} (Temp: {m['temperatura']}°C, Last update: {m['timestamp']})" for m in machines])

# Herramienta para obtener registros de una máquina específica
@mcp.tool()
async def get_machine_records(maquina_nombre: str, ctx: Context) -> List[Dict]:
    """Obtiene registros de una máquina específica"""
    if not maquina_nombre:
        return [{"error": "Se requiere el nombre de la máquina"}]
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{maquina_nombre}")
        if response.status_code == 404:
            return []
        return response.json()

# Herramienta para analizar la temperatura de una máquina
@mcp.tool()
async def analyze_machine_temperature(maquina_nombre: str, ctx: Context) -> str:
    """Analiza la temperatura y estado de una máquina específica"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{maquina_nombre}")
        if response.status_code == 404:
            return f"No se encontraron registros para la máquina '{maquina_nombre}'"
        records = response.json()
        if not records:
            return f"No hay datos para la máquina '{maquina_nombre}'"
        
        temperaturas = [record["temperatura"] for record in records]
        temp_promedio = sum(temperaturas) / len(temperaturas)
        ultimo_estado = records[0]["status"]
        ultimo_timestamp = records[0]["timestamp"]
        
        return f"""
        Análisis de la máquina '{maquina_nombre}':
        - Temperatura promedio: {temp_promedio:.2f}°C
        - Estado actual: {ultimo_estado}
        - Última actualización: {ultimo_timestamp}
        """

# Herramienta para análisis general de máquinas
@mcp.tool()
async def general_machines_analysis(ctx: Context) -> str:
    """Realiza un análisis general del estado de todas las máquinas"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = response.json()
        if not machines:
            return "No hay datos disponibles para análisis"
        
        online = sum(1 for m in machines if m["status"] == "Online")
        offline = sum(1 for m in machines if m["status"] == "Offline")
        temperaturas = [m["temperatura"] for m in machines]
        temp_promedio = sum(temperaturas) / len(temperaturas) if temperaturas else 0
        
        return f"""
        Análisis general de máquinas:
        - Máquinas Online: {online}
        - Máquinas Offline: {offline}
        - Temperatura promedio: {temp_promedio:.2f}°C
        - Total de máquinas: {len(machines)}
        """

# Herramienta para analizar y predecir todas las máquinas
@mcp.tool()
async def analyze_and_predict_all(request: str, ctx: Context) -> str:
    """Realiza un análisis y predicción para todas las máquinas según la solicitud del usuario"""
    if not request:
        return "Por favor, especifica qué análisis o predicción deseas (ejemplo: 'predice la temperatura las próximas horas')."
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = response.json()
        if not machines:
            return "No hay datos disponibles para análisis o predicción."
        
        maquinas = {}
        for record in machines:
            nombre = record["maquina_nombre"]
            if nombre not in maquinas:
                maquinas[nombre] = []
            maquinas[nombre].append(record)
        
        predicciones = []
        for nombre, records in maquinas.items():
            temperaturas = [r["temperatura"] for r in records]
            timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
            temp_promedio = sum(temperaturas) / len(temperaturas)
            ultimo_estado = records[0]["status"]
            ultimo_timestamp = timestamps[0]
            
            if len(records) > 1:
                time_diffs = [(timestamps[i] - timestamps[i+1]).total_seconds() / 3600 for i in range(len(timestamps)-1)]
                temp_diffs = [temperaturas[i] - temperaturas[i+1] for i in range(len(temperaturas)-1)]
                tendencia_por_hora = sum(temp_diffs[i] / time_diffs[i] for i in range(len(temp_diffs))) / len(temp_diffs)
                prediccion = temperaturas[0] + tendencia_por_hora * 1
            else:
                prediccion = temperaturas[0]
            
            predicciones.append(f"""
            Máquina '{nombre}':
            - Temperatura promedio: {temp_promedio:.2f}°C
            - Última temperatura: {temperaturas[0]}°C
            - Último estado: {ultimo_estado}
            - Última actualización: {ultimo_timestamp}
            - Predicción simple para la próxima hora: {prediccion:.2f}°C
            """)
        
        data_summary = f"""
        Predicciones para todas las máquinas:
        {''.join(predicciones)}
        Total de máquinas: {len(maquinas)}
        """

        instruction = f"""
        Basado en los siguientes datos:
        {data_summary}
        
        Por favor, realiza el siguiente análisis o predicción según la solicitud del usuario: '{request}'.
        Si los datos proporcionados no son suficientes para una predicción precisa, sugiere cómo podrían mejorarse (por ejemplo, más datos históricos o un modelo específico).
        Proporciona una respuesta detallada y explica tu razonamiento.
        """
        return instruction

# Herramienta con búsqueda semántica en el MCP
@mcp.tool()
async def get_pdf_data(request: str, ctx: Context) -> str:
    """Realiza una búsqueda semántica entre los PDFs para resolver la solicitud del usuario"""
    if not request:
        return "Por favor, especifica qué información deseas extraer o analizar de los PDFs."
    
    async with httpx.AsyncClient() as client:
        # Obtener todos los PDFs desde la API
        response = await client.get(f"{API_URL}/pdfs/")
        if response.status_code != 200:
            return f"Error al obtener PDFs: {response.text}"
        
        pdfs = response.json()
        if not pdfs:
            return "No hay PDFs almacenados en la base de datos."
        
        # Generar embedding para la solicitud del usuario
        request_embedding = model.encode(request, convert_to_tensor=True)
        
        # Calcular embeddings y similitudes para cada PDF
        pdfs_with_scores = []
        for pdf in pdfs:
            content = pdf["content"]
            if len(content) > 1000:  # Limitar para optimizar
                content = content[:1000]
            pdf_embedding = model.encode(content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(request_embedding, pdf_embedding).item()
            if similarity > 0.3:  # Umbral de relevancia
                pdfs_with_scores.append((pdf["filename"], content, similarity))
        
        if not pdfs_with_scores:
            return f"No se encontraron PDFs relevantes para la solicitud: '{request}'."
        
        # Ordenar por similitud (mayor primero)
        pdfs_with_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Limitar a los 3 más relevantes
        selected_pdfs = pdfs_with_scores[:min(3, len(pdfs_with_scores))]
        
        data_summary = ""
        for filename, content, similarity in selected_pdfs:
            data_summary += f"""
            Contenido del PDF '{filename}':
            {content}
            (Similitud: {similarity:.2f})
            """
        
        instruction = f"""
        Basado en el siguiente contenido extraído de los PDFs más relevantes:
        {data_summary}
        
        Por favor, realiza el siguiente análisis o responde a la solicitud del usuario: '{request}'.
        Extrae la información relevante del contenido y proporciona una respuesta detallada.
        Si el contenido no contiene los datos necesarios para responder, indícalo y sugiere cómo proceder.
        """
        return instruction

if __name__ == "__main__":
    mcp.run()