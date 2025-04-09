import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import statistics
from sentence_transformers import SentenceTransformer, util
import torch
import json

mcp = FastMCP("Industrial Analytics MCP")
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

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
        for machine in machines[:15]:
            status = (
                f"\n🔧 {machine['equipment']} ({machine['production_metrics']['product_type']})"
                f"\n- Operador: {machine['operator']}"
                f"\n- Producción: {machine['production_metrics']['quantity']} unidades"
                f"\n- Sensores: {machine['sensor_data']['temperature']}°C (Límite: {machine['contextual_info']['compliance_rules']['temperature_limit']}°C), "
                f"{machine['sensor_data']['pressure']} psi (Límite: {machine['contextual_info']['compliance_rules']['pressure_limit']} psi), "
                f"{machine['sensor_data']['vibration']} mm/s"
                f"\n- Última actualización: {machine['timestamp']}"
                f"\n- Notas de cumplimiento: {machine['contextual_info']['compliance_rules']['process_notes']}"
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
        
        # Usar el primer equipo como referencia para compliance_rules
        rules = machines[0]["contextual_info"]["compliance_rules"]
        return f"""
        📊 Dashboard de Producción:
        - Total producido: {total_production} unidades
        - Tipos de producto: {len(unique_products)} ({', '.join(unique_products)})
        - Equipos activos: {len(active_equipment)}
        - Temperatura promedio: {avg_temp:.1f}°C (Límite típico: {rules['temperature_limit']}°C)
        - Normas de cumplimiento relevantes:
          - Límite de presión: {rules['pressure_limit']} psi
          - Certificación de operador: {'Requerida' if rules['operator_certification_required'] else 'No requerida'}
          - Notas de proceso: {rules['process_notes']}
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
        # Integrar compliance_rules
        rules = records[0]["contextual_info"]["compliance_rules"]
        report.append(
            f"\n📜 Contexto operativo:"
            f"\n- Temperatura máxima permitida: {rules['temperature_limit']}°C"
            f"\n- Presión máxima permitida: {rules['pressure_limit']} psi"
            f"\n- Notas: {rules['process_notes']}"
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
        
        for r in relevant_records[:1000]:
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
            "sample_data": production_data[:3],
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
        """


@mcp.tool()
async def predict_temperature(ctx: Context, equipment: str, hours: int) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 5:
            return f"Insuficientes datos para {equipment} (mínimo 5 registros)"
        
        temp_data = []
        for r in records[:1000]:
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
            "data_samples": temp_data[:10],
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
        """

@mcp.tool()
async def predict_maintenance(ctx: Context, equipment: str, horizon_hours: int) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
        maintenance_data = []
        for r in records[:1000]:
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
            "recent_samples": maintenance_data[:3]
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
        """

@mcp.tool()
async def analyze_equipment_patterns(ctx: Context, equipment: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()
        
        if len(records) < 10:
            return f"Insuficientes datos para {equipment} (mínimo 10 registros)"
        
        pattern_data = []
        for r in records[:1000]:
            pattern_data.append({
                "time": r["timestamp"],
                "temp": r["sensor_data"]["temperature"],
                "pressure": r["sensor_data"]["pressure"],
                "vibration": r["sensor_data"]["vibration"],
                "production": r["production_metrics"]["quantity"],
                "product_type": r["production_metrics"]["product_type"]
            })
        
        rules = records[0]["contextual_info"]["compliance_rules"]
        stats = {
            "temp_range": (min(r["sensor_data"]["temperature"] for r in records), 
                          max(r["sensor_data"]["temperature"] for r in records)),
            "pressure_avg": statistics.mean(r["sensor_data"]["pressure"] for r in records),
            "production_variation": {
                "min": min(r["production_metrics"]["quantity"] for r in records),
                "max": max(r["production_metrics"]["quantity"] for r in records)
            },
            "compliance_limits": {
                "temp_limit": rules["temperature_limit"],
                "pressure_limit": rules["pressure_limit"]
            }
        }
        
        return f"""
        🔎 Datos para análisis de patrones en {equipment}:
        
        **Resumen estadístico:**
        ```json
        {json.dumps(stats, indent=2)}
        ```
        
        **Muestras de datos temporales:**
        ```json
        {json.dumps(pattern_data[:3], indent=2)}
        ```
        [Mostrando 3 de {len(pattern_data)} registros disponibles]
        
        **Instrucciones para el LLM:**
        1. Analiza correlaciones entre variables
        2. Identifica patrones temporales
        3. Detecta anomalías significativas
        4. Sugiere optimizaciones operativas
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
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/machines/{equipment}")
        records = response.json()[:10]
        
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
            """
        else:
            return f"""
            🛠️ Recomendaciones para {equipment}:
            Basado en los últimos {len(records)} registros:
            {chr(10).join(f'- {rec}' for rec in recommendations)}
            
            Parámetros actuales:
            - Temperatura: {temps[0]}°C (Límite: {rules['temperature_limit']}°C)
            - Presión: {pressures[0]} psi (Límite: {rules['pressure_limit']} psi)
            - Vibración: {vibes[0]} mm/s
            - Notas operativas: {rules['process_notes']}
            """

if __name__ == "__main__":
    mcp.run()