import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import statistics
from sentence_transformers import SentenceTransformer, util
import torch
import json
import smtplib
from email.mime.text import MIMEText
import os
import logging

mcp = FastMCP("Industrial Analytics MCP")
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de correo y API Key desde variables de entorno
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SUPERVISOR_EMAIL = os.getenv("SUPERVISOR_EMAIL")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
API_URL = "http://api:5000"  # Asegúrate de que esto esté definido

@mcp.tool()
async def process_event(ctx: Context, event_type: str, description: str, timestamp: str, equipment: str) -> str:
    """Procesa un evento industrial, lo contextualiza y notifica al supervisor."""
    logger.info(f"Procesando evento: {event_type} - {equipment}")
    
    # Aumentar timeout a 180 segundos
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        logger.info(f"Obteniendo datos de {equipment} desde la API")
        response = await client.get(f"{API_URL}/machines/{equipment}", params={"limit": 5})
        if response.status_code != 200:
            logger.error(f"Error al obtener datos: {response.text}")
            return f"Error al obtener datos de {equipment}: {response.text}"
        
        machine_data = response.json()
        latest_record = machine_data[0] if machine_data else {}
        logger.info("Datos de la máquina obtenidos exitosamente")
        
        context = {
            "event": {
                "event_type": event_type,
                "description": description,
                "timestamp": timestamp,
                "equipment": equipment
            },
            "latest_machine_data": {
                "sensor_data": latest_record.get("sensor_data", {}),
                "production_metrics": latest_record.get("production_metrics", {}),
                "timestamp": latest_record.get("timestamp", ""),
                "operator": latest_record.get("operator", "")
            }
        }
        
        OPENWEBUI_URL = "http://open-webui:8080/api/chat/completions"
        logger.info(f"Enviando solicitud a OpenWebUI: {OPENWEBUI_URL}")
        prompt = f"""
        Eres un asistente experto en mantenimiento industrial. Procesa el siguiente evento industrial y genera un informe claro y conciso para el supervisor:

        **Evento**: {event_type} - {description}
        **Equipo**: {equipment}
        **Timestamp**: {timestamp}
        **Datos recientes del equipo**: {json.dumps(context['latest_machine_data'], indent=2)}

        **Instrucciones**:
        1. Analiza el evento y los datos proporcionados.
        2. Identifica posibles causas del evento.
        3. Evalúa los impactos potenciales en la producción.
        4. Sugiere acciones correctivas específicas.
        5. Redacta un informe en formato narrativo (sin listas de opciones múltiples).

        **Ejemplo de formato esperado**:
        Informe para el supervisor:
        El evento "[event_type]" ocurrió en [equipment] el [timestamp]. Según los datos recientes, [análisis de los datos]. Las posibles causas incluyen [causas]. Esto podría impactar [impactos]. Se recomienda [acciones correctivas].
        """
        
        headers = {
            "Authorization": f"Bearer {OPENWEBUI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3.2:3b",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        logger.info(f"Enviando payload a OpenWebUI: {json.dumps(payload, indent=2)}")
        try:
            response = await client.post(OPENWEBUI_URL, json=payload, headers=headers)
            logger.info(f"Respuesta de OpenWebUI: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Error en OpenWebUI: {response.text}")
                return f"Error al procesar en OpenWebUI (código {response.status_code}): {response.text}"
            
            response_json = response.json()
            logger.info(f"Respuesta completa de OpenWebUI: {json.dumps(response_json, indent=2)}")
            
            llm_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"Respuesta del LLM: {llm_response}")
            
            if not llm_response:
                logger.warning("El contenido del LLM está vacío")
                llm_response = "No se pudo generar un informe debido a un error en el modelo."
        except httpx.TimeoutException as e:
            logger.error(f"Timeout al contactar OpenWebUI después de 180 segundos: {str(e)}")
            return f"Error: Timeout al contactar OpenWebUI después de 180 segundos: {str(e)}"
        except httpx.RequestError as e:
            logger.error(f"Error de red al contactar OpenWebUI: {str(e)}")
            return f"Error de red al contactar OpenWebUI: {str(e)}"
        
        logger.info("Enviando correo al supervisor")
        msg = MIMEText(llm_response, "plain", "utf-8")
        msg["Subject"] = f"Evento Industrial: {event_type} en {equipment}"
        msg["From"] = SMTP_USER
        msg["To"] = SUPERVISOR_EMAIL
        
        logger.info(f"Contenido del correo: {llm_response}")
        
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
            logger.info("Correo enviado exitosamente")
            return f"Evento procesado y enviado al supervisor: {llm_response[:100]}..."
        except Exception as e:
            logger.error(f"Error al enviar correo: {str(e)}")
            return f"Evento procesado pero error al enviar correo: {str(e)}"
# =============================================
# HERRAMIENTAS DE MONITOREO EN TIEMPO REAL MESS
# =============================================


@mcp.tool()
async def equipment_status(ctx: Context, equipment: Optional[str] = None) -> str:
    endpoint = f"{API_URL}/machines/{equipment}" if equipment else f"{API_URL}/machines/"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint)
        if response.status_code == 404:
            return f"Equipo {equipment} no encontrado" if equipment else "No hay equipos registrados"
        
        machines = response.json() if isinstance(response.json(), list) else [response.json()]
        
        report = ["🏭 Estado del Equipamiento:"]
        for machine in machines:
            status = (
                f"\n🔧 {machine['equipment']} ({machine['production_metrics']['product_type']})"
                f"\n- Operador: {machine['operator']}"
                f"\n- Producción: {machine['production_metrics']['quantity']} unidades"
                f"\n- Sensores: {machine['sensor_data']['temperature']}°C (Límite: {machine['contextual_info']['compliance_rules']['temperature_limit']}°C), "
                f"{machine['sensor_data']['pressure']} psi (Límite: {machine['contextual_info']['compliance_rules']['pressure_limit']} psi), "
                f"{machine['sensor_data']['vibration']} mm/s"
                f"\n- Última actualización: {machine['timestamp']}"
                f"\n- Notas de cumplimiento: {machine['contextual_info']['compliance_rules']['process_notes']}"
                f"\n\n💡 CONTEXTO PARA EL LLM: Los valores de sensores deben compararse con los límites especificados. "
                f"Generar alertas si temperatura > {machine['contextual_info']['compliance_rules']['temperature_limit']}°C "
                f"o presión > {machine['contextual_info']['compliance_rules']['pressure_limit']} psi. "
                f"Considerar notas operativas: {machine['contextual_info']['compliance_rules']['process_notes']}"
            )
            report.append(status)
        
        return "\n".join(report)

@mcp.tool()
async def production_dashboard(ctx: Context) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = response.json()
        
        if not machines:
            return "No hay datos de producción disponibles"
        
        total_production = sum(m["production_metrics"]["quantity"] for m in machines)
        unique_products = {m["production_metrics"]["product_type"] for m in machines}
        active_equipment = {m["equipment"] for m in machines}
        avg_temp = statistics.mean(m["sensor_data"]["temperature"] for m in machines)
        
        rules = machines[0]["contextual_info"]["compliance_rules"]
        return f"""
        📊 Dashboard de Producción:
        - Total producido: {total_production} unidades
        - Tipos de producto: {len(unique_products)} ({', '.join(unique_products)})
        - Equipos activos: {len(active_equipment)}
        - Temperatura promedio: {avg_temp:.1f}°C (Límite típico: {rules['temperature_limit']}°C)
        - Normas de cumplimiento relevantes:
          - Límite de presión: {rules['pressure_limit']} psi
          - Límite de temperatura: {rules['temperature_limit']}°C
          - Certificación de operador: {'Requerida' if rules['operator_certification_required'] else 'No requerida'}
          - Notas de proceso: {rules['process_notes']}
        
        💡 CONTEXTO PARA EL LLM: Analizar relación entre promedios y límites operativos. 
        Alertar si temperatura promedio está dentro del 10% del límite ({rules['temperature_limit']}°C) 
        o presión promedio dentro del 10% del límite ({rules['pressure_limit']} psi). 
        Considerar contexto de certificación y notas operativas.
        """

# =============================================
# HERRAMIENTAS DE ANÁLISIS DE PRODUCCIÓN
# =============================================



@mcp.tool()
async def product_analysis(ctx: Context, product_type: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = [m for m in response.json() if m["production_metrics"]["product_type"].lower() == product_type.lower()]
        
        if not machines:
            return f"No hay datos para el producto {product_type}"
        
        total = sum(m["production_metrics"]["quantity"] for m in machines)
        equipment_count = len({m["equipment"] for m in machines})
        avg_per_batch = total / len(machines)
        temps = [m["sensor_data"]["temperature"] for m in machines]
        avg_temp = statistics.mean(temps)
        rules = machines[0]["contextual_info"]["compliance_rules"]
        
        return f"""
        🍞 Análisis de {product_type}:
        - Total producido: {total} unidades
        - Lotes registrados: {len(machines)}
        - Equipos utilizados: {equipment_count}
        - Promedio por lote: {avg_per_batch:.1f} unidades
        - Temperatura promedio: {avg_temp:.1f}°C (Límite: {rules['temperature_limit']}°C)
        - Contexto de cumplimiento:
          - Límite de presión: {rules['pressure_limit']} psi
          - Notas operativas: {rules['process_notes']}
        
        💡 CONTEXTO PARA EL LLM: Evaluar si la producción de {product_type} opera dentro de parámetros seguros. 
        Alertar si temperatura promedio ({avg_temp:.1f}°C) se acerca al límite ({rules['temperature_limit']}°C) 
        o si hay variaciones significativas entre equipos. Considerar notas operativas en el análisis.
        """

@mcp.tool()
async def equipment_productivity(ctx: Context, equipment: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if not records:
            return f"No hay datos para {equipment}"
        
        product_stats = {}
        for record in records:
            product = record["production_metrics"]["product_type"]
            if product not in product_stats:
                product_stats[product] = {
                    'total': 0,
                    'count': 0,
                    'last_production': record["timestamp"]
                }
            product_stats[product]['total'] += record["production_metrics"]["quantity"]
            product_stats[product]['count'] += 1
            if record["timestamp"] > product_stats[product]['last_production']:
                product_stats[product]['last_production'] = record["timestamp"]
        
        report = [f"📈 Productividad de {equipment}:"]
        for product, stats in product_stats.items():
            report.append(
                f"\n📦 {product}:"
                f"\n- Total: {stats['total']} unidades"
                f"\n- Lotes: {stats['count']}"
                f"\n- Promedio: {stats['total']/stats['count']:.1f} unidades/lote"
                f"\n- Última producción: {stats['last_production'][:10]}"
            )
        
        rules = records[0]["contextual_info"]["compliance_rules"]
        report.append(
            f"\n📜 Contexto operativo:"
            f"\n- Temperatura máxima permitida: {rules['temperature_limit']}°C"
            f"\n- Presión máxima permitida: {rules['pressure_limit']} psi"
            f"\n- Notas: {rules['process_notes']}"
            f"\n\n💡 CONTEXTO PARA EL LLM: Analizar productividad en relación con límites operativos. "
            f"Alertar si alta productividad coincide con valores cercanos a límites de temperatura/presión. "
            f"Considerar que operador puede estar en entrenamiento: {rules['process_notes']}"
        )
        
        return "\n".join(report)

# =============================================
# HERRAMIENTAS DE ANÁLISIS PREDICTIVO
# =============================================

@mcp.tool()
async def predict_production(ctx: Context, product_type: str, hours: int ) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        relevant_records = [
            r for r in response.json() 
            if r["production_metrics"]["product_type"].lower() == product_type.lower()
        ]
        
        if len(relevant_records) < 5:
            return f"Insuficientes datos para {product_type} (mínimo 5 registros)"
        
        production_data = []
        equipment_stats = {}
        
        for r in relevant_records:
            record = {
                "time": r["timestamp"],
                "equipment": r["equipment"],
                "quantity": r["production_metrics"]["quantity"],
                "operator": r["operator"],
                "conditions": {
                    "temp": r["sensor_data"]["temperature"],
                    "pressure": r["sensor_data"]["pressure"]
                }
            }
            production_data.append(record)
            if r["equipment"] not in equipment_stats:
                equipment_stats[r["equipment"]] = []
            equipment_stats[r["equipment"]].append(r["production_metrics"]["quantity"])
        
        total_recent = sum(r["production_metrics"]["quantity"] for r in relevant_records[:24])
        avg_per_hour = total_recent / 24 if len(relevant_records) >= 24 else total_recent / len(relevant_records)
        rules = relevant_records[0]["contextual_info"]["compliance_rules"]
        
        analysis_context = {
            "product_type": product_type,
            "forecast_hours": hours,
            "recent_production": total_recent,
            "avg_hourly_rate": round(avg_per_hour, 2),
            "equipment_count": len(equipment_stats),
            "top_performers": {
                eq: max(qty) 
                for eq, qty in list(equipment_stats.items())[:3]
            },
            "sample_data": production_data,
            "compliance_rules": {
                "temperature_limit": rules["temperature_limit"],
                "pressure_limit": rules["pressure_limit"],
                "process_notes": rules["process_notes"]
            }
        }
        
        return f"""
        📈 Datos para predicción de producción de {product_type} (próximas {hours} horas):
        
        **Contexto de análisis:**
        ```json
        {json.dumps(analysis_context, indent=2)}
        ```
        
        **Instrucciones para el LLM:**
        1. Analiza patrones de producción por equipo
        2. Considera variaciones por turno/temporalidad
        3. Calcula proyección considerando capacidad actual
        4. Identifica cuellos de botella potenciales
        5. Proporciona rango probable (min-max)
        6. Considera impacto de reglas de cumplimiento y mando un aviso si no las cumple
        """


@mcp.tool()
async def predict_temperature(ctx: Context, equipment: str, hours: int) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 5:
            return f"Insuficientes datos para {equipment} (mínimo 5 registros)"
        
        temp_data = []
        for r in records:
            temp_data.append({
                "time": r["timestamp"],
                "temperature": r["sensor_data"]["temperature"],
                "pressure": r["sensor_data"]["pressure"],
                "vibration": r["sensor_data"]["vibration"],
                "production": r["production_metrics"]["quantity"]
            })
        
        rules = records[0]["contextual_info"]["compliance_rules"]
        analysis_context = {
            "equipment": equipment,
            "timeframe_hours": hours,
            "temperature_limit": rules["temperature_limit"],
            "data_samples": temp_data,
            "compliance_rules": {
                "pressure_limit": rules["pressure_limit"],
                "process_notes": rules["process_notes"]
            }
        }
        
        return f"""
        🔍 Datos para predicción de temperatura en {equipment} (próximas {hours} horas):
        
        **Contexto de análisis:**
        ```json
        {json.dumps(analysis_context, indent=2)}
        ```
        
        **Instrucciones para el LLM:**
        Analiza los patrones y proporciona:
        1. Temperatura predicha
        2. Factores clave influyentes
        3. Recomendaciones operativas
        4. Señales de alerta temprana
        5. Considera impacto de reglas de cumplimiento y mando un aviso si no las cumple
        """

@mcp.tool()
async def predict_maintenance(ctx: Context, equipment: str, horizon_hours: int) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
        maintenance_data = []
        for r in records:
            maintenance_data.append({
                "time": r["timestamp"],
                "sensors": {
                    "temp": r["sensor_data"]["temperature"],
                    "pressure": r["sensor_data"]["pressure"],
                    "vibration": r["sensor_data"]["vibration"]
                },
                "production": {
                    "quantity": r["production_metrics"]["quantity"],
                    "type": r["production_metrics"]["product_type"]
                }
            })
        
        rules = records[0]["contextual_info"]["compliance_rules"]
        analysis_context = {
            "equipment": equipment,
            "forecast_hours": horizon_hours,
            "limits": rules,
            "key_metrics": {
                "avg_temp": statistics.mean(r["sensor_data"]["temperature"] for r in records[:24]),
                "max_vibration": max(r["sensor_data"]["vibration"] for r in records[:24])
            },
            "recent_samples": maintenance_data
        }
        
        return f"""
        🛠️ Datos para predicción de mantenimiento en {equipment} (próximas {horizon_hours} horas):
        
        **Contexto de análisis:**
        ```json
        {json.dumps(analysis_context, indent=2)}
        ```
        
        **Instrucciones para el LLM:**
        1. Evalúa patrones de desgaste
        2. Identifica componentes críticos
        3. Estima probabilidad de fallo
        4. Sugiere acciones preventivas
        5. Considera impacto de reglas de cumplimiento y mando un aviso si no las cumple
        """


@mcp.tool()
async def analyze_equipment_patterns(ctx: Context, equipment: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
        # Calcular estadísticas completas
        temps = [r["sensor_data"]["temperature"] for r in records]
        pressures = [r["sensor_data"]["pressure"] for r in records]
        vibes = [r["sensor_data"]["vibration"] for r in records]
        productions = [r["production_metrics"]["quantity"] for r in records]
        
        rules = records[0]["contextual_info"]["compliance_rules"]
        
        stats = {
            "temperature": {
                "min": min(temps),
                "max": max(temps),
                "avg": statistics.mean(temps),
                "limit": rules["temperature_limit"],
                "over_limit_count": sum(1 for t in temps if t > rules["temperature_limit"])
            },
            "pressure": {
                "min": min(pressures),
                "max": max(pressures),
                "avg": statistics.mean(pressures),
                "limit": rules["pressure_limit"],
                "over_limit_count": sum(1 for p in pressures if p > rules["pressure_limit"])
            },
            "vibration": {
                "min": min(vibes),
                "max": max(vibes),
                "avg": statistics.mean(vibes)
            },
            "production": {
                "min": min(productions),
                "max": max(productions),
                "avg": statistics.mean(productions)
            },
            "time_range": {
                "start": min(r["timestamp"] for r in records),
                "end": max(r["timestamp"] for r in records)
            }
        }
        
        return f"""
        🔍 Análisis de Patrones en {equipment}:
        
        **ESTADÍSTICAS COMPLETAS:**
        ```json
        {json.dumps(stats, indent=2)}
        ```
        
        **LÍMITES OPERATIVOS:**
        - Temperatura máxima permitida: {rules['temperature_limit']}°C
        - Presión máxima permitida: {rules['pressure_limit']} psi
        - Notas operativas: {rules['process_notes']}
        
        **INSTRUCCIONES PARA EL LLM:**
        1. Analizar las estadísticas proporcionadas
        2. Identificar correlaciones entre variables
        3. Detectar patrones temporales
        4. Evaluar violaciones a límites operativos
        5. Proponer recomendaciones basadas en los datos
        6. Considerar el contexto operacional proporcionado
        """

# =============================================
# HERRAMIENTAS DE DOCUMENTACIÓN (RAG)
# =============================================

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



# =============================================
# HERRAMIENTAS DE MANTENIMIENTO
# =============================================

@mcp.tool()
async def maintenance_recommendations(ctx: Context, equipment: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if not records:
            return f"No hay datos suficientes para {equipment}"
        
        temps = [r["sensor_data"]["temperature"] for r in records]
        pressures = [r["sensor_data"]["pressure"] for r in records]
        vibes = [r["sensor_data"]["vibration"] for r in records]
        rules = records[0]["contextual_info"]["compliance_rules"]
        
        recommendations = []
        
        temp_avg = statistics.mean(temps)
        if temp_avg > rules["temperature_limit"] * 0.9:
            recommendations.append(f"🔧 Limpieza de sistemas de refrigeración (límite: {rules['temperature_limit']}°C)")
        
        pressure_max = max(pressures)
        if pressure_max > rules["pressure_limit"] * 0.85:
            recommendations.append(f"🔧 Verificación de válvulas y sellos (límite: {rules['pressure_limit']} psi)")
        
        if statistics.mean(vibes) > 3.0:
            recommendations.append("🔧 Balanceo de componentes rotativos")
        
        if not recommendations:
            return f"""
            ✅ {equipment} no requiere mantenimiento preventivo inmediato
            - Temperatura actual: {temps[0]}°C (Límite: {rules['temperature_limit']}°C)
            - Presión actual: {pressures[0]} psi (Límite: {rules['pressure_limit']} psi)
            - Notas operativas: {rules['process_notes']}
            
            💡 CONTEXTO PARA EL LLM: Equipo opera dentro de parámetros normales. 
            Monitorear si valores se acercan a límites ({rules['temperature_limit']}°C/{rules['pressure_limit']} psi).
            Considerar: {rules['process_notes']}
            """
        else:
            return f"""
            🛠️ Recomendaciones para {equipment}:
            Basado en {len(records)} registros:
            {chr(10).join(f'- {rec}' for rec in recommendations)}
            
            **Límites Operativos:**
            - Temperatura: {temps[0]}°C (Límite: {rules['temperature_limit']}°C)
            - Presión: {pressures[0]} psi (Límite: {rules['pressure_limit']} psi)
            - Vibración: {vibes[0]} mm/s
            - Notas: {rules['process_notes']}
            
            💡 CONTEXTO PARA EL LLM: Priorizar recomendaciones cerca de límites. 
            Considerar impacto en producción y notas operativas.
            """

if __name__ == "__main__":
    mcp.run()