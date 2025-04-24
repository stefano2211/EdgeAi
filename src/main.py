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
from pydantic import BaseModel

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

class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        if self.specific_date:
            try:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
                # Si hay specific_date, ignorar start_date y end_date
                self.start_date = None
                self.end_date = None
            except ValueError as e:
                raise ValueError(f"Formato de fecha específica inválido. Use YYYY-MM-DD: {str(e)}")
        else:
            if self.start_date:
                datetime.strptime(self.start_date, "%Y-%m-%d")
            if self.end_date:
                datetime.strptime(self.end_date, "%Y-%m-%d")
            if self.start_date and self.end_date and self.start_date > self.end_date:
                raise ValueError("La fecha de inicio no puede ser mayor que la fecha de fin")

@mcp.tool()
async def equipment_status(
    ctx: Context, 
    equipment: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """Obtiene el estado del equipo con filtros temporales."""
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    endpoint = f"{API_URL}/machines/{equipment}"
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(endpoint, params=params)
            if response.status_code == 404:
                return f"Equipo {equipment} no encontrado"
            
            machines = response.json()
            if not machines:
                return f"No se encontraron datos para {equipment}"
            
            # Validar que sea una lista
            if not isinstance(machines, list):
                machines = [machines]
            
            report = ["🏭 Estado del Equipamiento:"]
            for machine in machines:
                # Validar campos esenciales
                required_fields = {
                    'equipment': str,
                    'production_metrics': dict,
                    'sensor_data': dict,
                    'contextual_info': dict,
                    'operator': str,
                    'timestamp': str
                }
                
                missing_fields = []
                for field, field_type in required_fields.items():
                    if field not in machine or not isinstance(machine[field], field_type):
                        missing_fields.append(field)
                
                if missing_fields:
                    logger.error(f"Datos incompletos para {equipment}: Faltan {missing_fields}")
                    continue
                
                # Construir reporte
                status = (
                    f"\n🔧 {machine['equipment']} ({machine['production_metrics']['product_type']})"
                    f"\n- Operador: {machine['operator']}"
                    f"\n- Producción: {machine['production_metrics']['quantity']} unidades"
                    f"\n- Sensores:"
                    f"\n  • Temperatura: {machine['sensor_data']['temperature']}°C (Límite: {machine['contextual_info']['compliance_rules']['temperature_limit']}°C)"
                    f"\n  • Presión: {machine['sensor_data']['pressure']} psi (Límite: {machine['contextual_info']['compliance_rules']['pressure_limit']} psi)"
                    f"\n  • Vibración: {machine['sensor_data']['vibration']} mm/s"
                    f"\n- Última actualización: {machine['timestamp']}"
                    f"\n- Notas: {machine['contextual_info']['compliance_rules'].get('process_notes', 'Ninguna')}"
                )
                report.append(status)
                
                # Generar alertas
                alerts = []
                sensor_data = machine['sensor_data']
                rules = machine['contextual_info']['compliance_rules']
                
                if sensor_data['temperature'] > rules['temperature_limit']:
                    alerts.append(f"🚨 ALERTA: Temperatura ({sensor_data['temperature']}°C) excede el límite ({rules['temperature_limit']}°C)")
                
                if sensor_data['pressure'] > rules['pressure_limit']:
                    alerts.append(f"🚨 ALERTA: Presión ({sensor_data['pressure']} psi) excede el límite ({rules['pressure_limit']} psi)")
                
                if sensor_data['vibration'] > 3.5:  # Límite genérico para vibración
                    alerts.append(f"⚠️ ADVERTENCIA: Vibración elevada ({sensor_data['vibration']} mm/s)")
                
                if alerts:
                    report.append("\n" + "\n".join(alerts))
            
            return "\n".join(report)
            
        except httpx.RequestError as e:
            logger.error(f"Error de conexión: {str(e)}")
            return f"Error al conectar con la API: {str(e)}"
        except Exception as e:
            logger.error(f"Error inesperado: {str(e)}")
            return f"Error al procesar los datos: {str(e)}"

@mcp.tool()
async def production_dashboard(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/", params=params)
        machines = response.json()
        
        # Resto del código original SIN CAMBIOS
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
async def product_analysis(
    ctx: Context, 
    product_type: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/", params=params)
        machines = [m for m in response.json() if m["production_metrics"]["product_type"].lower() == product_type.lower()]
        
        # Resto del código original SIN CAMBIOS
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
async def equipment_productivity(
    ctx: Context, 
    equipment: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}", params=params)
        records = response.json()
        
        # Resto del código original SIN CAMBIOS
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
async def predict_production(
    ctx: Context, 
    product_type: str, 
    hours: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/", params=params)
        relevant_records = [
            r for r in response.json() 
            if r["production_metrics"]["product_type"].lower() == product_type.lower()
        ]
        
        # Resto del código original SIN CAMBIOS
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
        
        rules = relevant_records[0]["contextual_info"]["compliance_rules"]
        
        all_records = "\n".join(
            f"📅 {d['time']} | 🏭 {d['equipment']} | 👷 {d['operator']} | "
            f"📦 {d['quantity']} unidades | 🌡️ {d['conditions']['temp']}°C | "
            f"🌀 {d['conditions']['pressure']} psi"
            for d in production_data
        )
        
        return f"""
        📈 PREDICCIÓN DE PRODUCCIÓN - {product_type.upper()}
        ⏳ Período: Próximas {hours} horas
        📊 Registros completos ({len(relevant_records)}):
        
        {all_records}
        
        ⚠️ LÍMITES OPERATIVOS:
        • Temperatura máxima: {rules['temperature_limit']}°C
        • Presión máxima: {rules['pressure_limit']} psi
        • Notas: {rules['process_notes']}
        
        💡 INSTRUCCIONES PARA EL LLM:
        Analizar todos los registros mostrados y predecir producción considerando:
        1. Patrones históricos completos
        2. Límites operativos
        3. Variación entre equipos
        4. Contexto operacional
        """

@mcp.tool()
async def predict_temperature(
    ctx: Context, 
    equipment: str, 
    hours: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}", params=params)
        records = response.json()
        
        # Resto del código original SIN CAMBIOS
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
        
        all_readings = "\n".join(
            f"📅 {d['time']} | 🌡️ {d['temperature']}°C | 🌀 {d['pressure']} psi | "
            f"📳 {d['vibration']} mm/s | 📦 {d['production']} unidades"
            for d in temp_data
        )
        
        return f"""
        🌡️ PREDICCIÓN DE TEMPERATURA - {equipment.upper()}
        ⏳ Período: Próximas {hours} horas
        📊 Registros completos ({len(records)}):
        
        {all_readings}
        
        ⚠️ LÍMITES OPERATIVOS:
        • Temperatura máxima: {rules['temperature_limit']}°C
        • Presión máxima: {rules['pressure_limit']} psi
        • Notas: {rules['process_notes']}
        
        💡 INSTRUCCIONES PARA EL LLM:
        Analizar todos los registros mostrados y predecir temperatura considerando:
        1. Tendencia histórica completa
        2. Correlación con presión y producción
        3. Límites operativos
        4. Patrones de vibración
        """

@mcp.tool()
async def predict_maintenance(
    ctx: Context, 
    equipment: str, 
    horizon_hours: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}", params=params)
        records = response.json()
        
        # Resto del código original SIN CAMBIOS
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
        
        all_maintenance = "\n".join(
            f"📅 {d['time']} | 🌡️ {d['sensors']['temp']}°C | 🌀 {d['sensors']['pressure']} psi | "
            f"📳 {d['sensors']['vibration']} mm/s | 📦 {d['production']['quantity']} {d['production']['type']}"
            for d in maintenance_data
        )
        
        return f"""
        🛠️ PREDICCIÓN DE MANTENIMIENTO - {equipment.upper()}
        ⏳ Horizonte: Próximas {horizon_hours} horas
        📊 Registros completos ({len(records)}):
        
        {all_maintenance}
        
        ⚠️ LÍMITES OPERATIVOS:
        • Temperatura máxima: {rules['temperature_limit']}°C
        • Presión máxima: {rules['pressure_limit']} psi
        • Notas: {rules['process_notes']}
        
        💡 INSTRUCCIONES PARA EL LLM:
        Analizar todos los registros mostrados y predecir mantenimiento considerando:
        1. Patrones completos de desgaste
        2. Historial de valores de sensores
        3. Relación con producción
        4. Límites operativos
        5. Contexto de operación
        """


@mcp.tool()
async def analyze_equipment_patterns(
    ctx: Context, 
    equipment: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}", params=params)
        records = response.json()
        
        # Resto del código original SIN CAMBIOS
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
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
async def get_pdf_data(ctx: Context, request: str) -> str:
    """Busca en PDFs relevantes para la consulta"""
    async with httpx.AsyncClient() as client:
        try:
            # 1. Obtener lista de PDFs disponibles
            pdf_list_response = await client.get(f"{API_URL}/pdfs/list")
            pdf_list = pdf_list_response.json()
            
            if not pdf_list:
                return "No hay PDFs disponibles en el sistema."

            # 2. Seleccionar los más relevantes
            model = SentenceTransformer('all-MiniLM-L6-v2')
            request_embedding = model.encode(request, convert_to_tensor=True)
            
            pdf_scores = []
            for pdf in pdf_list:
                text_to_embed = f"{pdf['filename']} {pdf['description']}"
                pdf_embedding = model.encode(text_to_embed, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(request_embedding, pdf_embedding).item()
                pdf_scores.append((pdf['filename'], similarity))
            
            # Ordenar y filtrar por relevancia
            pdf_scores.sort(key=lambda x: x[1], reverse=True)
            top_pdfs = [pdf[0] for pdf in pdf_scores[:2] if pdf[1] > 0.3]

            if not top_pdfs:
                return f"No se encontraron PDFs relevantes para: '{request}'"

            # 3. Obtener contenidos
            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": top_pdfs}
            )
            
            if content_response.status_code != 200:
                return f"Error al obtener contenidos: {content_response.text}"
            
            pdf_contents = content_response.json()
            
            # 4. Preparar respuesta estructurada
            context = {
                "user_request": request,
                "pdfs": pdf_contents["pdfs"],
                "analysis_instructions": (
                    "Analiza los documentos y responde considerando:\n"
                    "1. Relevancia para la solicitud\n"
                    "2. Datos técnicos encontrados\n"
                    "3. Posibles acciones recomendadas"
                )
            }
            
            return json.dumps(context, indent=2)
            
        except Exception as e:
            logger.error(f"Error en get_pdf_data: {str(e)}")
            return f"Error al procesar la solicitud: {str(e)}"

# =============================================
# HERRAMIENTAS DE MANTENIMIENTO
# =============================================

@mcp.tool()
async def maintenance_recommendations(
    ctx: Context, 
    equipment: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    time_filter = TimeFilter(
        start_date=start_date,
        end_date=end_date,
        specific_date=specific_date
    )
    try:
        time_filter.validate_dates()
    except ValueError as e:
        return str(e)
    
    params = {}
    if time_filter.specific_date:
        params["specific_date"] = time_filter.specific_date
    else:
        if time_filter.start_date:
            params["start_date"] = time_filter.start_date
        if time_filter.end_date:
            params["end_date"] = time_filter.end_date
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}", params=params)
        records = response.json()
        
        # Resto del código original SIN CAMBIOS
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