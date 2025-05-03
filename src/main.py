import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime
from typing import Optional
import logging
from pydantic import BaseModel
import re
import statistics
from sentence_transformers import SentenceTransformer, util
import torch
import os

mcp = FastMCP("Industrial Analytics MCP")
API_URL = "http://api:5000"
model = SentenceTransformer('all-MiniLM-L6-v2')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        if self.specific_date:
            try:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
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
async def read_pdf(
    ctx: Context,
    machine: Optional[str] = None
) -> str:
    """
    Genera un informe con el contenido extraído de un PDF relevante para la máquina especificada.
    """
    try:
        async with httpx.AsyncClient() as client:
            pdf_list_response = await client.get(f"{API_URL}/pdfs/list")
            pdf_list = pdf_list_response.json()

            if not pdf_list:
                return (
                    "Informe de Contenido de PDF\n"
                    "==========================\n"
                    "Error: No hay PDFs disponibles en el sistema.\n"
                    "Recomendación: Cargue los manuales técnicos de las máquinas como PDFs en el sistema.\n"
                    "Sugerencia: Asegúrese de incluir un archivo con un nombre como 'NombreDeLaMáquina.pdf' (por ejemplo, 'ModelA.pdf')."
                )

            top_pdf = None
            if machine:
                exact_match = next((pdf for pdf in pdf_list if pdf['filename'].lower() == f"{machine.lower()}.pdf"), None)
                if exact_match:
                    top_pdf = exact_match['filename']
                    logger.info(f"Encontrado PDF exacto para {machine}: {top_pdf}")
                else:
                    pdf_scores = []
                    machine_embedding = model.encode(machine, convert_to_tensor=True)
                    for pdf in pdf_list:
                        text_to_embed = f"{pdf['filename']} {pdf['description']}"
                        pdf_embedding = model.encode(text_to_embed, convert_to_tensor=True)
                        similarity = util.pytorch_cos_sim(machine_embedding, pdf_embedding).item()
                        pdf_scores.append((pdf['filename'], similarity))
                    
                    logger.debug(f"Puntajes de similitud para {machine}: {pdf_scores}")
                    pdf_scores.sort(key=lambda x: x[1], reverse=True)
                    top_pdf = pdf_scores[0][0] if pdf_scores and pdf_scores[0][1] > 0.3 else None
            
            if not top_pdf and pdf_list:
                top_pdf = pdf_list[0]['filename']
            
            if not top_pdf:
                available_pdfs = [pdf['filename'] for pdf in pdf_list]
                return (
                    "Informe de Contenido de PDF\n"
                    "==========================\n"
                    f"Error: No se encontraron PDFs relevantes para la máquina '{machine}'.\n"
                    f"PDFs disponibles en el sistema: {', '.join(available_pdfs) if available_pdfs else 'Ninguno'}.\n"
                    "Recomendación: Cargue el manual técnico de la máquina como PDF en el sistema.\n"
                    f"Sugerencia: Use un nombre como '{machine}.pdf' para facilitar la identificación."
                )

            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": [top_pdf]}
            )
            if content_response.status_code != 200:
                return (
                    "Informe de Contenido de PDF\n"
                    "==========================\n"
                    f"Error: No se pudo obtener el contenido del PDF '{top_pdf}'.\n"
                    f"Detalles: {content_response.text}\n"
                    "Recomendación: Verifique la disponibilidad del archivo en el sistema o contacte al soporte técnico."
                )

            pdf_contents = content_response.json()
            content = pdf_contents["pdfs"][0]["content"]
            
            report = [
                "Informe de Contenido de PDF",
                "==========================",
                f"Máquina: {machine or 'No especificada'}",
                f"Archivo PDF: {top_pdf}",
                "",
                "Resumen Ejecutivo",
                "----------------",
                f"Se extrajo el contenido del manual técnico '{top_pdf}' asociado a la máquina '{machine or 'No especificada'}'. "
                f"El contenido incluye especificaciones técnicas y reglas de cumplimiento relevantes.",
                "",
                "Contenido Extraído",
                "-----------------",
                content,
                "",
                "Recomendaciones",
                "--------------",
                "1. Asegúrese de que el PDF contenga las reglas de cumplimiento en el formato correcto (por ejemplo, 'temperature <= 80°C, vibration <= 1.0 mm/s, defects <= 2, uptime >= 90%').",
                "2. Verifique que el manual esté actualizado con las especificaciones más recientes de la máquina.",
                "3. Si el contenido es extenso, considere dividir el PDF en secciones para facilitar su procesamiento."
            ]
            
            return "\n".join(report)
            
    except Exception as e:
        logger.error(f"No se pudo leer el PDF para la máquina '{machine}': {str(e)}")
        return (
            "Informe de Contenido de PDF\n"
            "==========================\n"
            f"Error: No se pudo leer el PDF para la máquina '{machine}'.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al equipo de soporte técnico para diagnosticar el problema.\n"
            f"Sugerencia: Asegúrese de que el archivo '{machine}.pdf' esté cargado y sea accesible."
        )

@mcp.tool()
async def check_temperature_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de la temperatura para todas las máquinas contra los límites definidos en los manuales técnicos (PDF).
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        try:
            time_filter.validate_dates()
        except ValueError as e:
            return (
                "Informe de Cumplimiento de Temperatura para Todas las Máquinas\n"
                "===========================================================\n"
                f"Error: Error en parámetros de fecha: {str(e)}.\n"
                "Recomendación: Use fechas en formato YYYY-MM-DD (por ejemplo, '2025-04-01')."
            )

        endpoint = f"{API_URL}/machines/"
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
                logs = response.json()
                if not logs:
                    return (
                        "Informe de Cumplimiento de Temperatura para Todas las Máquinas\n"
                        "===========================================================\n"
                        f"Período: {time_filter.specific_date or f'{time_filter.start_date} a {time_filter.end_date}'}\n"
                        "Resultado: No se encontraron registros.\n"
                        "Recomendación: Verifique si las máquinas estuvieron operativas durante el período especificado."
                    )
            except httpx.RequestError as e:
                logger.error(f"Error en la solicitud API: {str(e)}")
                return (
                    "Informe de Cumplimiento de Temperatura para Todas las Máquinas\n"
                    "===========================================================\n"
                    f"Error: No se pudieron recuperar datos de la API.\n"
                    f"Detalles: {str(e)}\n"
                    "Recomendación: Verifique la conexión con la API o contacte al equipo de soporte técnico."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Temperatura para Todas las Máquinas\n"
                "===========================================================\n"
                "Error: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegúrese de que los datos de producción estén registrados."
            )

        results = {}
        pdf_cache = {}
        total_records = 0

        for machine in machines:
            machine_logs = [log for log in logs if log["machine"] == machine]
            if not machine_logs:
                continue

            if machine not in pdf_cache:
                pdf_result = await read_pdf(ctx, machine=machine)
                if "Error:" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Error: ")[1].split("\n")[0],
                        "recommendation": "Cargue el manual técnico de la máquina como PDF en el sistema."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido Extraído\n-----------------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            temp_limit = 80.0
            rules_match = re.search(
                r"temperature\s*[<≤=]+\s*(\d+\.?\d*)\s*(?:°C|C)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                temp_limit = float(rules_match.group(1))
                logger.info(f"Límite de temperatura extraído del PDF para {machine}: temperature <= {temp_limit}°C")
            else:
                logger.warning(f"No se encontró el límite de temperatura en el PDF de {machine}. Usando valor por defecto: {temp_limit}°C")
                warning_message = (
                    f"Advertencia: No se encontró el límite de temperatura en el PDF de {machine}. Usando valor por defecto: <= {temp_limit}°C\n"
                    f"Contenido del PDF:\n{compliance_text}\n"
                    "Recomendación: Verifique que el PDF contenga la regla de temperatura en el formato esperado."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["temperature"] > temp_limit:
                    issue = f"Temperatura: {log['temperature']}°C excede el límite de {temp_limit}°C"
                    non_compliant_count += 1
                
                compliance_report.append({
                    "id": log["id"],
                    "date": log["date"],
                    "machine": log["machine"],
                    "production_line": log["production_line"],
                    "material": log["material"],
                    "metrics": {
                        "uptime": log["uptime"],
                        "defects": log["defects"],
                        "vibration": log["vibration"],
                        "temperature": log["temperature"],
                        "throughput": log["throughput"],
                        "inventory_level": log["inventory_level"]
                    },
                    "defect_type": log["defect_type"],
                    "compliant": not issue,
                    "issue": issue
                })

            results[machine] = {
                "compliance_report": compliance_report,
                "non_compliant_count": non_compliant_count,
                "temp_limit": temp_limit,
                "rules_found": bool(rules_match),
                "warning_message": warning_message if not rules_match else None
            }
            total_records += len(compliance_report)

        period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
        report = [
            "Informe de Cumplimiento de Temperatura para Todas las Máquinas",
            "===========================================================",
            f"Período: {period}",
            f"Total de Máquinas Analizadas: {len(machines)}",
            f"Total de Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Error: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Total de Registros: {len(data['compliance_report'])}\n"
                f"  Límite Aplicado: Temperatura <= {data['temp_limit']}°C\n"
            )
            if not data["rules_found"]:
                report.append(data["warning_message"] + "\n")
            
            report.append("  Detalles de Registros:\n")
            for i, entry in enumerate(data["compliance_report"], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Máquina: {entry['machine']}\n"
                    f"      Línea de Producción: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Tiempo Activo: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} unidades/h\n"
                    f"        - Nivel de Inventario: {entry['metrics']['inventory_level']} unidades\n"
                    f"      Tipo de Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema Detectado:\n{issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    Registros No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "------------",
            f"Se verificaron {total_records} registros de producción de {len(machines)} máquinas para el período {period} contra los límites de temperatura definidos en los manuales técnicos (PDF). "
            f"Los registros listados como 'No Conforme' exceden los límites de temperatura especificados para cada máquina. "
            "Por favor, revise los registros no conformes detallados arriba para identificar las causas de las desviaciones y tomar acciones correctivas, como verificar los sistemas de enfriamiento o recalibrar los sensores de temperatura."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"No se pudo verificar el cumplimiento de temperatura para las máquinas: {str(e)}")
        return (
            "Informe de Cumplimiento de Temperatura para Todas las Máquinas\n"
            "===========================================================\n"
            f"Error: No se pudo verificar el cumplimiento de temperatura para las máquinas.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al equipo de soporte técnico para diagnosticar el problema."
        )

@mcp.tool()
async def check_vibration_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de la vibración para todas las máquinas contra los límites definidos en los manuales técnicos (PDF).
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        try:
            time_filter.validate_dates()
        except ValueError as e:
            return (
                "Informe de Cumplimiento de Vibración para Todas las Máquinas\n"
                "=========================================================\n"
                f"Error: Error en parámetros de fecha: {str(e)}.\n"
                "Recomendación: Use fechas en formato YYYY-MM-DD (por ejemplo, '2025-04-01')."
            )

        endpoint = f"{API_URL}/machines/"
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
                logs = response.json()
                if not logs:
                    return (
                        "Informe de Cumplimiento de Vibración para Todas las Máquinas\n"
                        "=========================================================\n"
                        f"Período: {time_filter.specific_date or f'{time_filter.start_date} a {time_filter.end_date}'}\n"
                        "Resultado: No se encontraron registros.\n"
                        "Recomendación: Verifique si las máquinas estuvieron operativas durante el período especificado."
                    )
            except httpx.RequestError as e:
                logger.error(f"Error en la solicitud API: {str(e)}")
                return (
                    "Informe de Cumplimiento de Vibración para Todas las Máquinas\n"
                    "=========================================================\n"
                    f"Error: No se pudieron recuperar datos de la API.\n"
                    f"Detalles: {str(e)}\n"
                    "Recomendación: Verifique la conexión con la API o contacte al equipo de soporte técnico."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Vibración para Todas las Máquinas\n"
                "=========================================================\n"
                "Error: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegúrese de que los datos de producción estén registrados."
            )

        results = {}
        pdf_cache = {}
        total_records = 0

        for machine in machines:
            machine_logs = [log for log in logs if log["machine"] == machine]
            if not machine_logs:
                continue

            if machine not in pdf_cache:
                pdf_result = await read_pdf(ctx, machine=machine)
                if "Error:" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Error: ")[1].split("\n")[0],
                        "recommendation": "Cargue el manual técnico de la máquina como PDF en el sistema."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido Extraído\n-----------------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            vibration_limit = 1.0
            rules_match = re.search(
                r"vibration\s*[<≤=]+\s*(\d+\.?\d*)\s*(?:mm/s|mm\s*/\s*s)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                vibration_limit = float(rules_match.group(1))
                logger.info(f"Límite de vibración extraído del PDF para {machine}: vibration <= {vibration_limit} mm/s")
            else:
                logger.warning(f"No se encontró el límite de vibración en el PDF de {machine}. Usando valor por defecto: {vibration_limit} mm/s")
                warning_message = (
                    f"Advertencia: No se encontró el límite de vibración en el PDF de {machine}. Usando valor por defecto: <= {vibration_limit} mm/s\n"
                    f"Contenido del PDF:\n{compliance_text}\n"
                    "Recomendación: Verifique que el PDF contenga la regla de vibración en el formato esperado."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["vibration"] > vibration_limit:
                    issue = f"Vibración: {log['vibration']} mm/s excede el límite de {vibration_limit} mm/s"
                    non_compliant_count += 1
                
                compliance_report.append({
                    "id": log["id"],
                    "date": log["date"],
                    "machine": log["machine"],
                    "production_line": log["production_line"],
                    "material": log["material"],
                    "metrics": {
                        "uptime": log["uptime"],
                        "defects": log["defects"],
                        "vibration": log["vibration"],
                        "temperature": log["temperature"],
                        "throughput": log["throughput"],
                        "inventory_level": log["inventory_level"]
                    },
                    "defect_type": log["defect_type"],
                    "compliant": not issue,
                    "issue": issue
                })

            results[machine] = {
                "compliance_report": compliance_report,
                "non_compliant_count": non_compliant_count,
                "vibration_limit": vibration_limit,
                "rules_found": bool(rules_match),
                "warning_message": warning_message if not rules_match else None
            }
            total_records += len(compliance_report)

        period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
        report = [
            "Informe de Cumplimiento de Vibración para Todas las Máquinas",
            "=========================================================",
            f"Período: {period}",
            f"Total de Máquinas Analizadas: {len(machines)}",
            f"Total de Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Error: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Total de Registros: {len(data['compliance_report'])}\n"
                f"  Límite Aplicado: Vibración <= {data['vibration_limit']} mm/s\n"
            )
            if not data["rules_found"]:
                report.append(data["warning_message"] + "\n")
            
            report.append("  Detalles de Registros:\n")
            for i, entry in enumerate(data["compliance_report"], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Máquina: {entry['machine']}\n"
                    f"      Línea de Producción: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Tiempo Activo: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} unidades/h\n"
                    f"        - Nivel de Inventario: {entry['metrics']['inventory_level']} unidades\n"
                    f"      Tipo de Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema Detectado:\n{issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    Registros No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "------------",
            f"Se verificaron {total_records} registros de producción de {len(machines)} máquinas para el período {period} contra los límites de vibración definidos en los manuales técnicos (PDF). "
            f"Los registros listados como 'No Conforme' exceden los límites de vibración especificados para cada máquina. "
            "Por favor, revise los registros no conformes detallados arriba para identificar las causas de las desviaciones y tomar acciones correctivas, como inspeccionar componentes mecánicos o programar mantenimiento."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"No se pudo verificar el cumplimiento de vibración para las máquinas: {str(e)}")
        return (
            "Informe de Cumplimiento de Vibración para Todas las Máquinas\n"
            "=========================================================\n"
            f"Error: No se pudo verificar el cumplimiento de vibración para las máquinas.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al equipo de soporte técnico para diagnosticar el problema."
        )

@mcp.tool()
async def check_uptime_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento del tiempo activo para todas las máquinas contra los límites definidos en los manuales técnicos (PDF).
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        try:
            time_filter.validate_dates()
        except ValueError as e:
            return (
                "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas\n"
                "=============================================================\n"
                f"Error: Error en parámetros de fecha: {str(e)}.\n"
                "Recomendación: Use fechas en formato YYYY-MM-DD (por ejemplo, '2025-04-01')."
            )

        endpoint = f"{API_URL}/machines/"
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
                logs = response.json()
                if not logs:
                    return (
                        "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas\n"
                        "=============================================================\n"
                        f"Período: {time_filter.specific_date or f'{time_filter.start_date} a {time_filter.end_date}'}\n"
                        "Resultado: No se encontraron registros.\n"
                        "Recomendación: Verifique si las máquinas estuvieron operativas durante el período especificado."
                    )
            except httpx.RequestError as e:
                logger.error(f"Error en la solicitud API: {str(e)}")
                return (
                    "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas\n"
                    "=============================================================\n"
                    f"Error: No se pudieron recuperar datos de la API.\n"
                    f"Detalles: {str(e)}\n"
                    "Recomendación: Verifique la conexión con la API o contacte al equipo de soporte técnico."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas\n"
                "=============================================================\n"
                "Error: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegúrese de que los datos de producción estén registrados."
            )

        results = {}
        pdf_cache = {}
        total_records = 0

        for machine in machines:
            machine_logs = [log for log in logs if log["machine"] == machine]
            if not machine_logs:
                continue

            if machine not in pdf_cache:
                pdf_result = await read_pdf(ctx, machine=machine)
                if "Error:" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Error: ")[1].split("\n")[0],
                        "recommendation": "Cargue el manual técnico de la máquina como PDF en el sistema."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido Extraído\n-----------------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            uptime_min = 90.0
            rules_match = re.search(
                r"uptime\s*[>≥=]+\s*(\d+\.?\d*)\s*%",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                uptime_min = float(rules_match.group(1))
                logger.info(f"Límite de tiempo activo extraído del PDF para {machine}: uptime >= {uptime_min}%")
            else:
                logger.warning(f"No se encontró el límite de tiempo activo en el PDF de {machine}. Usando valor por defecto: {uptime_min}%")
                warning_message = (
                    f"Advertencia: No se encontró el límite de tiempo activo en el PDF de {machine}. Usando valor por defecto: >= {uptime_min}%\n"
                    f"Contenido del PDF:\n{compliance_text}\n"
                    "Recomendación: Verifique que el PDF contenga la regla de tiempo activo en el formato esperado."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["uptime"] < uptime_min:
                    issue = f"Tiempo Activo: {log['uptime']}% está por debajo del mínimo de {uptime_min}%"
                    non_compliant_count += 1
                
                compliance_report.append({
                    "id": log["id"],
                    "date": log["date"],
                    "machine": log["machine"],
                    "production_line": log["production_line"],
                    "material": log["material"],
                    "metrics": {
                        "uptime": log["uptime"],
                        "defects": log["defects"],
                        "vibration": log["vibration"],
                        "temperature": log["temperature"],
                        "throughput": log["throughput"],
                        "inventory_level": log["inventory_level"]
                    },
                    "defect_type": log["defect_type"],
                    "compliant": not issue,
                    "issue": issue
                })

            results[machine] = {
                "compliance_report": compliance_report,
                "non_compliant_count": non_compliant_count,
                "uptime_min": uptime_min,
                "rules_found": bool(rules_match),
                "warning_message": warning_message if not rules_match else None
            }
            total_records += len(compliance_report)

        period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
        report = [
            "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas",
            "=============================================================",
            f"Período: {period}",
            f"Total de Máquinas Analizadas: {len(machines)}",
            f"Total de Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Error: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Total de Registros: {len(data['compliance_report'])}\n"
                f"  Límite Aplicado: Tiempo Activo >= {data['uptime_min']}%\n"
            )
            if not data["rules_found"]:
                report.append(data["warning_message"] + "\n")
            
            report.append("  Detalles de Registros:\n")
            for i, entry in enumerate(data["compliance_report"], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Máquina: {entry['machine']}\n"
                    f"      Línea de Producción: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Tiempo Activo: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} unidades/h\n"
                    f"        - Nivel de Inventario: {entry['metrics']['inventory_level']} unidades\n"
                    f"      Tipo de Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema Detectado:\n{issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    Registros No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "------------",
            f"Se verificaron {total_records} registros de producción de {len(machines)} máquinas para el período {period} contra los límites de tiempo activo definidos en los manuales técnicos (PDF). "
            f"Los registros listados como 'No Conforme' están por debajo de los límites de tiempo activo especificados para cada máquina. "
            "Por favor, revise los registros no conformes detallados arriba para identificar las causas de las desviaciones y tomar acciones correctivas, como investigar paradas no planificadas o optimizar el mantenimiento preventivo."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"No se pudo verificar el cumplimiento de tiempo activo para las máquinas: {str(e)}")
        return (
            "Informe de Cumplimiento de Tiempo Activo para Todas las Máquinas\n"
            "=============================================================\n"
            f"Error: No se pudo verificar el cumplimiento de tiempo activo para las máquinas.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al equipo de soporte técnico para diagnosticar el problema."
        )

@mcp.tool()
async def check_defects_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de defectos para todas las máquinas contra los límites definidos en los manuales técnicos (PDF).
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        try:
            time_filter.validate_dates()
        except ValueError as e:
            return (
                "Informe de Cumplimiento de Defectos para Todas las Máquinas\n"
                "========================================================\n"
                f"Error: Error en parámetros de fecha: {str(e)}.\n"
                "Recomendación: Use fechas en formato YYYY-MM-DD (por ejemplo, '2025-04-01')."
            )

        endpoint = f"{API_URL}/machines/"
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
                logs = response.json()
                if not logs:
                    return (
                        "Informe de Cumplimiento de Defectos para Todas las Máquinas\n"
                        "========================================================\n"
                        f"Período: {time_filter.specific_date or f'{time_filter.start_date} a {time_filter.end_date}'}\n"
                        "Resultado: No se encontraron registros.\n"
                        "Recomendación: Verifique si las máquinas estuvieron operativas durante el período especificado."
                    )
            except httpx.RequestError as e:
                logger.error(f"Error en la solicitud API: {str(e)}")
                return (
                    "Informe de Cumplimiento de Defectos para Todas las Máquinas\n"
                    "========================================================\n"
                    f"Error: No se pudieron recuperar datos de la API.\n"
                    f"Detalles: {str(e)}\n"
                    "Recomendación: Verifique la conexión con la API o contacte al equipo de soporte técnico."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Defectos para Todas las Máquinas\n"
                "========================================================\n"
                "Error: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegúrese de que los datos de producción estén registrados."
            )

        results = {}
        pdf_cache = {}
        total_records = 0

        for machine in machines:
            machine_logs = [log for log in logs if log["machine"] == machine]
            if not machine_logs:
                continue

            if machine not in pdf_cache:
                pdf_result = await read_pdf(ctx, machine=machine)
                if "Error:" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Error: ")[1].split("\n")[0],
                        "recommendation": "Cargue el manual técnico de la máquina como PDF en el sistema."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido Extraído\n-----------------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            defects_limit = 2
            rules_match = re.search(
                r"defects\s*[<≤=]+\s*(\d+)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                defects_limit = int(rules_match.group(1))
                logger.info(f"Límite de defectos extraído del PDF para {machine}: defects <= {defects_limit}")
            else:
                logger.warning(f"No se encontró el límite de defectos en el PDF de {machine}. Usando valor por defecto: {defects_limit}")
                warning_message = (
                    f"Advertencia: No se encontró el límite de defectos en el PDF de {machine}. Usando valor por defecto: <= {defects_limit}\n"
                    f"Contenido del PDF:\n{compliance_text}\n"
                    "Recomendación: Verifique que el PDF contenga la regla de defectos en el formato esperado."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["defects"] > defects_limit:
                    issue = f"Defectos: {log['defects']} excede el límite de {defects_limit}"
                    non_compliant_count += 1
                
                compliance_report.append({
                    "id": log["id"],
                    "date": log["date"],
                    "machine": log["machine"],
                    "production_line": log["production_line"],
                    "material": log["material"],
                    "metrics": {
                        "uptime": log["uptime"],
                        "defects": log["defects"],
                        "vibration": log["vibration"],
                        "temperature": log["temperature"],
                        "throughput": log["throughput"],
                        "inventory_level": log["inventory_level"]
                    },
                    "defect_type": log["defect_type"],
                    "compliant": not issue,
                    "issue": issue
                })

            results[machine] = {
                "compliance_report": compliance_report,
                "non_compliant_count": non_compliant_count,
                "defects_limit": defects_limit,
                "rules_found": bool(rules_match),
                "warning_message": warning_message if not rules_match else None
            }
            total_records += len(compliance_report)

        period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
        report = [
            "Informe de Cumplimiento de Defectos para Todas las Máquinas",
            "=========================================================",
            f"Período: {period}",
            f"Total de Máquinas Analizadas: {len(machines)}",
            f"Total de Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Error: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Total de Registros: {len(data['compliance_report'])}\n"
                f"  Límite Aplicado: Defectos <= {data['defects_limit']}\n"
            )
            if not data["rules_found"]:
                report.append(data["warning_message"] + "\n")
            
            report.append("  Detalles de Registros:\n")
            for i, entry in enumerate(data["compliance_report"], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Máquina: {entry['machine']}\n"
                    f"      Línea de Producción: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Tiempo Activo: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} unidades/h\n"
                    f"        - Nivel de Inventario: {entry['metrics']['inventory_level']} unidades\n"
                    f"      Tipo de Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema Detectado:\n{issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    Registros No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "------------",
            f"Se verificaron {total_records} registros de producción de {len(machines)} máquinas para el período {period} contra los límites de defectos definidos en los manuales técnicos (PDF). "
            f"Los registros listados como 'No Conforme' exceden los límites de defectos especificados para cada máquina. "
            "Por favor, revise los registros no conformes detallados arriba para identificar las causas de las desviaciones y tomar acciones correctivas, como analizar el proceso de producción o implementar controles de calidad adicionales."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"No se pudo verificar el cumplimiento de defectos para las máquinas: {str(e)}")
        return (
            "Informe de Cumplimiento de Defectos para Todas las Máquinas\n"
            "========================================================\n"
            f"Error: No se pudo verificar el cumplimiento de defectos para las máquinas.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al equipo de soporte técnico para diagnosticar el problema."
        )

if __name__ == "__main__":
    mcp.run()