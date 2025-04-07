import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import statistics
from sentence_transformers import SentenceTransformer, util
import torch

mcp = FastMCP("Industrial Analytics MCP")
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================
# HERRAMIENTAS DE MONITOREO EN TIEMPO REAL
# =============================================

@mcp.tool()
async def equipment_status(ctx: Context, equipment: Optional[str] = None) -> str:
    """Obtiene el estado actual de equipos específicos o todos los equipos"""
    endpoint = f"{API_URL}/machines/{equipment}" if equipment else f"{API_URL}/machines/"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint)
        if response.status_code == 404:
            return f"Equipo {equipment} no encontrado" if equipment else "No hay equipos registrados"
        
        machines = response.json() if isinstance(response.json(), list) else [response.json()]
        
        report = ["🏭 Estado del Equipamiento:"]
        for machine in machines[:15]:  # Limitar a 15 resultados
            status = (
                f"\n🔧 {machine['equipment']} ({machine['production_metrics']['product_type']})"
                f"\n- Operador: {machine['operator']}"
                f"\n- Producción: {machine['production_metrics']['quantity']} unidades"
                f"\n- Sensores: {machine['sensor_data']['temperature']}°C, "
                f"{machine['sensor_data']['pressure']} psi, "
                f"{machine['sensor_data']['vibration']} mm/s"
                f"\n- Última actualización: {machine['timestamp']}"
            )
            report.append(status)
        
        return "\n".join(report)

@mcp.tool()
async def production_dashboard(ctx: Context) -> str:
    """Muestra un resumen general de la producción"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = response.json()
        
        if not machines:
            return "No hay datos de producción disponibles"
        
        # Cálculo de métricas
        total_production = sum(m["production_metrics"]["quantity"] for m in machines)
        unique_products = {m["production_metrics"]["product_type"] for m in machines}
        active_equipment = {m["equipment"] for m in machines}
        avg_temp = statistics.mean(m["sensor_data"]["temperature"] for m in machines)
        
        return f"""
        📊 Dashboard de Producción:
        - Total producido: {total_production} unidades
        - Tipos de producto: {len(unique_products)} ({', '.join(unique_products)})
        - Equipos activos: {len(active_equipment)}
        - Temperatura promedio: {avg_temp:.1f}°C
        """

# =============================================
# HERRAMIENTAS DE ANÁLISIS DE PRODUCCIÓN
# =============================================

@mcp.tool()
async def product_analysis(ctx: Context, product_type: str) -> str:
    """Analiza la producción de un tipo de producto específico"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        machines = [m for m in response.json() if m["production_metrics"]["product_type"].lower() == product_type.lower()]
        
        if not machines:
            return f"No hay datos para el producto {product_type}"
        
        total = sum(m["production_metrics"]["quantity"] for m in machines)
        equipment_count = len({m["equipment"] for m in machines})
        avg_per_batch = total / len(machines)
        
        # Análisis de sensores para este producto
        temps = [m["sensor_data"]["temperature"] for m in machines]
        avg_temp = statistics.mean(temps)
        
        return f"""
        🍞 Análisis de {product_type}:
        - Total producido: {total} unidades
        - Lotes registrados: {len(machines)}
        - Equipos utilizados: {equipment_count}
        - Promedio por lote: {avg_per_batch:.1f} unidades
        - Temperatura promedio: {avg_temp:.1f}°C
        """

@mcp.tool()
async def equipment_productivity(ctx: Context, equipment: str) -> str:
    """Evalúa la productividad de un equipo específico"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if not records:
            return f"No hay datos para {equipment}"
        
        # Agrupar por producto
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
        
        # Generar reporte
        report = [f"📈 Productividad de {equipment}:"]
        for product, stats in product_stats.items():
            report.append(
                f"\n📦 {product}:"
                f"\n- Total: {stats['total']} unidades"
                f"\n- Lotes: {stats['count']}"
                f"\n- Promedio: {stats['total']/stats['count']:.1f} unidades/lote"
                f"\n- Última producción: {stats['last_production'][:10]}"
            )
        
        return "\n".join(report)

# =============================================
# HERRAMIENTAS DE ANÁLISIS PREDICTIVO
# =============================================

@mcp.tool()
async def predict_temperature(ctx: Context, equipment: str, hours: int = 24) -> str:
    """Predice la temperatura futura basada en tendencias recientes"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 5:
            return f"Insuficientes datos para {equipment} (mínimo 5 registros)"
        
        # Preparar datos para análisis
        temps = [r["sensor_data"]["temperature"] for r in records[:24]]  # Últimas 24 lecturas
        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records[:24]]
        time_diffs = [(timestamps[i]-timestamps[i+1]).total_seconds()/3600 for i in range(len(timestamps)-1)]
        
        # Calcular tendencia
        temp_changes = [temps[i]-temps[i+1] for i in range(len(temps)-1)]
        hourly_trend = sum(temp_changes[i]/time_diffs[i] for i in range(len(temp_changes))) / len(temp_changes)
        
        # Predicción lineal simple
        current_temp = temps[0]
        predicted_temp = current_temp + (hourly_trend * hours)
        
        # Obtener límites de compliance
        rules = records[0]["contextual_info"]["compliance_rules"]
        temp_limit = rules["temperature_limit"]
        
        # Evaluar riesgo
        risk = ""
        if predicted_temp > temp_limit:
            risk = f"🚨 ALERTA: Predicción excede límite de {temp_limit}°C"
        elif predicted_temp > temp_limit * 0.9:
            risk = f"⚠️ Advertencia: Se aproxima al límite de {temp_limit}°C"
        
        return f"""
        🔮 Predicción de Temperatura para {equipment}:
        - Temperatura actual: {current_temp}°C
        - Tendencia horaria: {'+' if hourly_trend > 0 else ''}{hourly_trend:.2f}°C/hora
        - Predicción en {hours} horas: {predicted_temp:.1f}°C
        - Límite seguro: {temp_limit}°C
        {risk}
        """

@mcp.tool()
async def predict_maintenance(ctx: Context, equipment: str, forecast_hours: int = 48) -> str:
    """Predice necesidad de mantenimiento en las próximas X horas"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
        # Analizar múltiples parámetros
        temps = [r["sensor_data"]["temperature"] for r in records[:24]]
        pressures = [r["sensor_data"]["pressure"] for r in records[:24]]
        vibrations = [r["sensor_data"]["vibration"] for r in records[:24]]
        rules = records[0]["contextual_info"]["compliance_rules"]
        
        # Calcular tendencias por hora
        def calculate_hourly_trend(values):
            changes = [values[i]-values[i+1] for i in range(len(values)-1)]
            time_diffs = [(datetime.fromisoformat(records[i]["timestamp"]) - 
                          datetime.fromisoformat(records[i+1]["timestamp"])).total_seconds()/3600 
                         for i in range(len(values)-1)]
            return sum(c/d for c,d in zip(changes, time_diffs)) / len(changes)
        
        temp_trend = calculate_hourly_trend(temps)
        pressure_trend = calculate_hourly_trend(pressures)
        vibe_trend = calculate_hourly_trend(vibrations)
        
        # Generar predicciones
        current_values = {
            'temp': temps[0],
            'pressure': pressures[0],
            'vibration': vibrations[0]
        }
        
        predicted_values = {
            'temp': current_values['temp'] + (temp_trend * forecast_hours),
            'pressure': current_values['pressure'] + (pressure_trend * forecast_hours),
            'vibration': current_values['vibration'] + (vibe_trend * forecast_hours)
        }
        
        # Evaluar riesgos
        alerts = []
        if predicted_values['temp'] > rules['temperature_limit']:
            alerts.append(f"Temperatura predicha: {predicted_values['temp']:.1f}°C > {rules['temperature_limit']}°C")
        if predicted_values['pressure'] > rules['pressure_limit']:
            alerts.append(f"Presión predicha: {predicted_values['pressure']:.1f}psi > {rules['pressure_limit']}psi")
        if predicted_values['vibration'] > 4.0:
            alerts.append(f"Vibración predicha: {predicted_values['vibration']:.1f}mm/s > 4.0mm/s")
        
        # Calcular probabilidad de fallo
        risk_score = 0
        if alerts: risk_score = min(90 + (len(alerts)*5), 100)
        
        maintenance_advice = []
        if risk_score > 70:
            maintenance_advice.append("🔧 Realizar mantenimiento preventivo inmediato")
            maintenance_advice.append("🛑 Considerar parada no programada")
        elif risk_score > 40:
            maintenance_advice.append("🔧 Programar mantenimiento pronto")
        else:
            maintenance_advice.append("✅ Operación normal - Sin mantenimiento urgente")
        
        return f"""
        🛠️ Predicción de Mantenimiento para {equipment} (próximas {forecast_hours} horas):
        
        Parámetros actuales:
        - Temp: {current_values['temp']}°C (Límite: {rules['temperature_limit']}°C)
        - Presión: {current_values['pressure']}psi (Límite: {rules['pressure_limit']}psi)
        - Vibración: {current_values['vibration']}mm/s
        
        Tendencias horarias:
        - Temp: {'+' if temp_trend > 0 else ''}{temp_trend:.3f}°C/hora
        - Presión: {'+' if pressure_trend > 0 else ''}{pressure_trend:.3f}psi/hora
        - Vibración: {'+' if vibe_trend > 0 else ''}{vibe_trend:.3f}mm/s/hora
        
        {'🚨 Alertas:' if alerts else '✅ Sin alertas'}
        {chr(10).join('- ' + alert for alert in alerts)}
        
        Riesgo estimado: {risk_score}%
        {chr(10).join(maintenance_advice)}
        """

@mcp.tool()
async def predict_production(ctx: Context, product_type: str, hours: int = 24) -> str:
    """Predice la producción esperada para un tipo de producto"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/")
        relevant_records = [
            r for r in response.json() 
            if r["production_metrics"]["product_type"].lower() == product_type.lower()
        ]
        
        if len(relevant_records) < 5:
            return f"Insuficientes datos para {product_type} (mínimo 5 registros)"
        
        # Agrupar por equipos
        equipment_data = {}
        for record in relevant_records[:24]:  # Últimas 24 horas
            eq = record["equipment"]
            if eq not in equipment_data:
                equipment_data[eq] = []
            equipment_data[eq].append(record)
        
        # Predecir por equipo
        total_predicted = 0
        equipment_predictions = []
        
        for eq, records in equipment_data.items():
            if len(records) < 3:
                continue
                
            quantities = [r["production_metrics"]["quantity"] for r in records]
            timestamps = [datetime.fromisoformat(r["timestamp"]) for r in records]
            
            # Calcular tasa de producción por hora
            time_diffs = [(timestamps[i]-timestamps[i+1]).total_seconds()/3600 for i in range(len(timestamps)-1)]
            prod_rates = [quantities[i]/time_diffs[i] for i in range(len(time_diffs))]
            avg_rate = statistics.mean(prod_rates)
            
            predicted = avg_rate * hours
            total_predicted += predicted
            equipment_predictions.append(f"- {eq}: {predicted:.1f} unidades")
        
        if not equipment_predictions:
            return "No se pudo calcular predicción (datos insuficientes)"
        
        return f"""
        🔮 Predicción de Producción para {product_type} (próximas {hours} horas):
        - Producción total estimada: {total_predicted:.1f} unidades
        - Desglose por equipo:
        {chr(10).join(equipment_predictions)}
        
        Basado en el rendimiento promedio reciente de {len(equipment_predictions)} equipos
        """

# =============================================
# HERRAMIENTAS DE DOCUMENTACIÓN (RAG)
# =============================================

@mcp.tool()
async def search_work_instructions(ctx: Context, product_type: str) -> str:
    """Busca instrucciones de trabajo para un tipo de producto"""
    async with httpx.AsyncClient() as client:
        # Obtener documentos relevantes
        response = await client.get(f"{API_URL}/pdfs/")
        pdfs = response.json()
        
        if not pdfs:
            return "No hay documentos disponibles"
        
        # Búsqueda semántica
        query = f"Instrucciones de trabajo para producto {product_type}"
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        results = []
        for pdf in pdfs:
            content = pdf["content"][:1000]  # Limitar tamaño para eficiencia
            doc_embedding = model.encode(content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            
            if similarity > 0.25:  # Umbral bajo para capturar más resultados
                results.append({
                    "filename": pdf["filename"],
                    "similarity": similarity,
                    "excerpt": content[:200] + "..."
                })
        
        if not results:
            return f"No se encontraron instrucciones para {product_type}"
        
        # Ordenar y formatear resultados
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        report = [f"📄 Instrucciones para {product_type}:"]
        for doc in results[:3]:  # Top 3 resultados
            report.append(
                f"\n📑 {doc['filename']} (Relevancia: {doc['similarity']:.0%})"
                f"\n{doc['excerpt']}"
            )
        
        return "\n".join(report)

# =============================================
# HERRAMIENTAS DE MANTENIMIENTO
# =============================================

@mcp.tool()
async def maintenance_recommendations(ctx: Context, equipment: str) -> str:
    """Genera recomendaciones de mantenimiento preventivo"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()[:10]  # Últimos 10 registros
        
        if not records:
            return f"No hay datos suficientes para {equipment}"
        
        # Análisis de tendencias
        temps = [r["sensor_data"]["temperature"] for r in records]
        pressures = [r["sensor_data"]["pressure"] for r in records]
        vibes = [r["sensor_data"]["vibration"] for r in records]
        rules = records[0]["contextual_info"]["compliance_rules"]
        
        recommendations = []
        
        # Evaluar temperatura
        temp_avg = statistics.mean(temps)
        if temp_avg > rules["temperature_limit"] * 0.9:
            recommendations.append("🔧 Limpieza de sistemas de refrigeración")
        
        # Evaluar presión
        pressure_max = max(pressures)
        if pressure_max > rules["pressure_limit"] * 0.85:
            recommendations.append("🔧 Verificación de válvulas y sellos")
        
        # Evaluar vibración
        if statistics.mean(vibes) > 3.0:
            recommendations.append("🔧 Balanceo de componentes rotativos")
        
        if not recommendations:
            return f"✅ {equipment} no requiere mantenimiento preventivo inmediato"
        
        return f"""
        🛠️ Recomendaciones para {equipment}:
        Basado en los últimos {len(records)} registros:
        {chr(10).join(f'- {rec}' for rec in recommendations)}
        
        Parámetros actuales:
        - Temperatura: {temps[0]}°C (Límite: {rules['temperature_limit']}°C)
        - Presión: {pressures[0]} psi (Límite: {rules['pressure_limit']} psi)
        - Vibración: {vibes[0]} mm/s
        """

if __name__ == "__main__":
    mcp.run()