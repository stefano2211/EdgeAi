import httpx
from mcp.server.fastmcp import FastMCP, Context
from datetime import datetime
from typing import Optional
import logging
from pydantic import BaseModel
import re
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Inicializar FastMCP para OpenWebUI
mcp = FastMCP("Industrial Analytics MCP")

# Configuración de URLs basada en docker-compose.yml
API_URL = "http://api:5000"

# Modelo de embeddings para selección de PDFs
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo para validar filtros de tiempo
class TimeFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    specific_date: Optional[str] = None

    def validate_dates(self):
        """Valida que las fechas estén en formato YYYY-MM-DD y sean lógicas."""
        try:
            if self.specific_date:
                datetime.strptime(self.specific_date, "%Y-%m-%d")
                self.start_date = None
                self.end_date = None
            else:
                if self.start_date:
                    datetime.strptime(self.start_date, "%Y-%m-%d")
                if self.end_date:
                    datetime.strptime(self.end_date, "%Y-%m-%d")
                if self.start_date and self.end_date and self.start_date > self.end_date:
                    raise ValueError("La fecha de inicio no puede ser mayor que la fecha de fin")
        except ValueError as e:
            raise ValueError(f"Formato de fecha inválido. Use YYYY-MM-DD: {str(e)}")

@mcp.tool()
async def read_pdf(
    ctx: Context,
    machine: Optional[str] = None
) -> str:
    """
    Genera un informe con el contenido extraído de un PDF relevante para la máquina especificada.
    Formato optimizado para legibilidad por Llama 3.1 en OpenWebUI.
    """
    try:
        async with httpx.AsyncClient() as client:
            pdf_list_response = await client.get(f"{API_URL}/pdfs/list")
            pdf_list = pdf_list_response.json()

            if not pdf_list:
                return (
                    "Informe de Contenido de PDF\n"
                    "==========================\n"
                    "Estado: Error\n"
                    "Mensaje: No hay PDFs disponibles.\n"
                    "Recomendación: Suba los manuales técnicos en formato PDF al sistema.\n"
                    "Sugerencia: Use nombres como 'NombreMaquina.pdf' (ej. 'ModelA.pdf')."
                )

            top_pdf = None
            if machine:
                exact_match = next((pdf for pdf in pdf_list if pdf['filename'].lower() == f"{machine.lower()}.pdf"), None)
                if exact_match:
                    top_pdf = exact_match['filename']
                    logger.info(f"PDF exacto encontrado para {machine}: {top_pdf}")
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
                    "Estado: Error\n"
                    f"Mensaje: No se encontraron PDFs para '{machine}'.\n"
                    f"PDFs disponibles: {', '.join(available_pdfs) if available_pdfs else 'Ninguno'}.\n"
                    "Recomendación: Suba el manual técnico de la máquina como PDF.\n"
                    f"Sugerencia: Use '{machine}.pdf' para fácil identificación."
                )

            content_response = await client.get(
                f"{API_URL}/pdfs/content/",
                params={"filenames": [top_pdf]}
            )
            if content_response.status_code != 200:
                return (
                    "Informe de Contenido de PDF\n"
                    "==========================\n"
                    "Estado: Error\n"
                    f"Mensaje: No se pudo obtener el contenido de '{top_pdf}'.\n"
                    f"Detalles: {content_response.text}\n"
                    "Recomendación: Verifique la disponibilidad del archivo o contacte al soporte."
                )

            pdf_contents = content_response.json()
            content = pdf_contents["pdfs"][0]["content"]
            
            report = [
                "Informe de Contenido de PDF",
                "==========================",
                f"Máquina: {machine or 'No especificada'}",
                f"PDF: {top_pdf}",
                "",
                "Resumen",
                "-------",
                f"Contenido extraído de '{top_pdf}' para '{machine or 'No especificada'}'. Incluye especificaciones técnicas.",
                "",
                "Contenido",
                "--------",
                content,
                "",
                "Recomendaciones",
                "--------------",
                "- Asegure que el PDF tenga reglas de cumplimiento claras (ej. 'temperature <= 80°C').",
                "- Verifique que el manual esté actualizado.",
                "- Considere dividir PDFs extensos en secciones."
            ]
            
            return "\n".join(report)
            
    except Exception as e:
        logger.error(f"Error al leer PDF para '{machine}': {str(e)}")
        return (
            "Informe de Contenido de PDF\n"
            "==========================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo leer el PDF para '{machine}'.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al soporte técnico.\n"
            f"Sugerencia: Verifique que '{machine}.pdf' esté disponible."
        )

@mcp.tool()
async def check_temperature_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de temperatura para todas las máquinas contra límites en PDFs.
    Informe optimizado para Llama 3.1 en OpenWebUI.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

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
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()
            if not logs:
                period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
                return (
                    "Informe de Cumplimiento de Temperatura\n"
                    "=====================================\n"
                    f"Período: {period}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros.\n"
                    "Recomendación: Verifique si las máquinas estuvieron operativas."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Temperatura\n"
                "=====================================\n"
                "Estado: Error\n"
                "Mensaje: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegure que los datos de producción estén registrados."
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
                if "Estado: Error" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Mensaje: ")[1].split("\n")[0],
                        "recommendation": "Suba el manual técnico como PDF."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido\n--------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            temp_limit = 80.0
            rules_match = re.search(
                r"temperature\s*[<≤=]+\s*(\d+\.?\d*)\s*(?:°C|C)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                temp_limit = float(rules_match.group(1))
                logger.info(f"Límite de temperatura para {machine}: <= {temp_limit}°C")
            else:
                logger.warning(f"No se encontró límite de temperatura para {machine}. Usando {temp_limit}°C")
                warning_message = (
                    f"Advertencia: No se encontró límite de temperatura en el PDF de {machine}. Usando <= {temp_limit}°C\n"
                    "Recomendación: Verifique que el PDF tenga la regla en el formato correcto."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["temperature"] > temp_limit:
                    issue = f"Temperatura: {log['temperature']}°C excede {temp_limit}°C"
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
            "Informe de Cumplimiento de Temperatura",
            "=====================================",
            f"Período: {period}",
            f"Máquinas Analizadas: {len(machines)}",
            f"Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "-------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Estado: Error\n"
                    f"  Mensaje: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Registros: {len(data['compliance_report'])}\n"
                f"  Límite: Temperatura <= {data['temp_limit']}°C\n"
            )
            if not data["rules_found"]:
                report.append(f"  {data['warning_message']}\n")
            
            report.append("  Registros:\n")
            for i, entry in enumerate(data['compliance_report'], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Línea: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Uptime: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} u/h\n"
                    f"        - Inventario: {entry['metrics']['inventory_level']} u\n"
                    f"      Tipo Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema: {issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "-------------",
            f"Se verificaron {total_records} registros de {len(machines)} máquinas en {period} contra límites de temperatura en PDFs. "
            "Los registros 'No Conforme' exceden los límites. Revise los detalles para tomar acciones correctivas."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error en check_temperature_compliance: {str(e)}")
        return (
            "Informe de Cumplimiento de Temperatura\n"
            "=====================================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo verificar el cumplimiento.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al soporte técnico."
        )

@mcp.tool()
async def check_vibration_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de vibración para todas las máquinas contra límites en PDFs.
    Informe optimizado para Llama 3.1 en OpenWebUI.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

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
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()
            if not logs:
                period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
                return (
                    "Informe de Cumplimiento de Vibración\n"
                    "===================================\n"
                    f"Período: {period}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros.\n"
                    "Recomendación: Verifique si las máquinas estuvieron operativas."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Vibración\n"
                "===================================\n"
                "Estado: Error\n"
                "Mensaje: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegure que los datos de producción estén registrados."
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
                if "Estado: Error" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Mensaje: ")[1].split("\n")[0],
                        "recommendation": "Suba el manual técnico como PDF."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido\n--------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            vibration_limit = 1.0
            rules_match = re.search(
                r"vibration\s*[<≤=]+\s*(\d+\.?\d*)\s*(?:mm/s|mm\s*/\s*s)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                vibration_limit = float(rules_match.group(1))
                logger.info(f"Límite de vibración para {machine}: <= {vibration_limit} mm/s")
            else:
                logger.warning(f"No se encontró límite de vibración para {machine}. Usando {vibration_limit} mm/s")
                warning_message = (
                    f"Advertencia: No se encontró límite de vibración en el PDF de {machine}. Usando <= {vibration_limit} mm/s\n"
                    "Recomendación: Verifique que el PDF tenga la regla en el formato correcto."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["vibration"] > vibration_limit:
                    issue = f"Vibración: {log['vibration']} mm/s excede {vibration_limit} mm/s"
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
            "Informe de Cumplimiento de Vibración",
            "===================================",
            f"Período: {period}",
            f"Máquinas Analizadas: {len(machines)}",
            f"Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "-------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Estado: Error\n"
                    f"  Mensaje: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Registros: {len(data['compliance_report'])}\n"
                f"  Límite: Vibración <= {data['vibration_limit']} mm/s\n"
            )
            if not data["rules_found"]:
                report.append(f"  {data['warning_message']}\n")
            
            report.append("  Registros:\n")
            for i, entry in enumerate(data['compliance_report'], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Línea: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Uptime: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} u/h\n"
                    f"        - Inventario: {entry['metrics']['inventory_level']} u\n"
                    f"      Tipo Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema: {issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "-------------",
            f"Se verificaron {total_records} registros de {len(machines)} máquinas en {period} contra límites de vibración en PDFs. "
            "Los registros 'No Conforme' exceden los límites. Revise los detalles para tomar acciones correctivas."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error en check_vibration_compliance: {str(e)}")
        return (
            "Informe de Cumplimiento de Vibración\n"
            "===================================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo verificar el cumplimiento.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al soporte técnico."
        )

@mcp.tool()
async def check_uptime_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de tiempo activo para todas las máquinas contra límites en PDFs.
    Informe optimizado para Llama 3.1 en OpenWebUI.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

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
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()
            if not logs:
                period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
                return (
                    "Informe de Cumplimiento de Uptime\n"
                    "================================\n"
                    f"Período: {period}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros.\n"
                    "Recomendación: Verifique si las máquinas estuvieron operativas."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Uptime\n"
                "================================\n"
                "Estado: Error\n"
                "Mensaje: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegure que los datos de producción estén registrados."
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
                if "Estado: Error" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Mensaje: ")[1].split("\n")[0],
                        "recommendation": "Suba el manual técnico como PDF."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido\n--------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            uptime_min = 90.0
            rules_match = re.search(
                r"uptime\s*[>≥=]+\s*(\d+\.?\d*)\s*%",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                uptime_min = float(rules_match.group(1))
                logger.info(f"Límite de uptime para {machine}: >= {uptime_min}%")
            else:
                logger.warning(f"No se encontró límite de uptime para {machine}. Usando {uptime_min}%")
                warning_message = (
                    f"Advertencia: No se encontró límite de uptime en el PDF de {machine}. Usando >= {uptime_min}%\n"
                    "Recomendación: Verifique que el PDF tenga la regla en el formato correcto."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["uptime"] < uptime_min:
                    issue = f"Uptime: {log['uptime']}% menor a {uptime_min}%"
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
            "Informe de Cumplimiento de Uptime",
            "================================",
            f"Período: {period}",
            f"Máquinas Analizadas: {len(machines)}",
            f"Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "-------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Estado: Error\n"
                    f"  Mensaje: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Registros: {len(data['compliance_report'])}\n"
                f"  Límite: Uptime >= {data['uptime_min']}%\n"
            )
            if not data["rules_found"]:
                report.append(f"  {data['warning_message']}\n")
            
            report.append("  Registros:\n")
            for i, entry in enumerate(data['compliance_report'], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Línea: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Uptime: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} u/h\n"
                    f"        - Inventario: {entry['metrics']['inventory_level']} u\n"
                    f"      Tipo Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema: {issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "-------------",
            f"Se verificaron {total_records} registros de {len(machines)} máquinas en {period} contra límites de uptime en PDFs. "
            "Los registros 'No Conforme' están por debajo del límite. Revise los detalles para tomar acciones correctivas."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error en check_uptime_compliance: {str(e)}")
        return (
            "Informe de Cumplimiento de Uptime\n"
            "================================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo verificar el cumplimiento.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al soporte técnico."
        )

@mcp.tool()
async def check_defects_compliance(
    ctx: Context,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_date: Optional[str] = None
) -> str:
    """
    Verifica el cumplimiento de defectos para todas las máquinas contra límites en PDFs.
    Informe optimizado para Llama 3.1 en OpenWebUI.
    """
    try:
        time_filter = TimeFilter(
            start_date=start_date,
            end_date=end_date,
            specific_date=specific_date
        )
        time_filter.validate_dates()

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
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            logs = response.json()
            if not logs:
                period = time_filter.specific_date or f"{time_filter.start_date} a {time_filter.end_date}"
                return (
                    "Informe de Cumplimiento de Defectos\n"
                    "==================================\n"
                    f"Período: {period}\n"
                    "Estado: Sin datos\n"
                    "Mensaje: No se encontraron registros.\n"
                    "Recomendación: Verifique si las máquinas estuvieron operativas."
                )

        machines = sorted(set(log["machine"] for log in logs))
        if not machines:
            return (
                "Informe de Cumplimiento de Defectos\n"
                "==================================\n"
                "Estado: Error\n"
                "Mensaje: No se encontraron máquinas en los registros.\n"
                "Recomendación: Asegure que los datos de producción estén registrados."
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
                if "Estado: Error" in pdf_result:
                    results[machine] = {
                        "error": pdf_result.split("Mensaje: ")[1].split("\n")[0],
                        "recommendation": "Suba el manual técnico como PDF."
                    }
                    continue
                pdf_cache[machine] = pdf_result.split("Contenido\n--------\n")[1].split("\n\nRecomendaciones\n--------------")[0]

            compliance_text = pdf_cache[machine]
            
            defects_limit = 2
            rules_match = re.search(
                r"defects\s*[<≤=]+\s*(\d+)",
                compliance_text,
                re.IGNORECASE
            )
            if rules_match:
                defects_limit = int(rules_match.group(1))
                logger.info(f"Límite de defectos para {machine}: <= {defects_limit}")
            else:
                logger.warning(f"No se encontró límite de defectos para {machine}. Usando {defects_limit}")
                warning_message = (
                    f"Advertencia: No se encontró límite de defectos en el PDF de {machine}. Usando <= {defects_limit}\n"
                    "Recomendación: Verifique que el PDF tenga la regla en el formato correcto."
                )

            compliance_report = []
            non_compliant_count = 0
            for log in machine_logs:
                issue = None
                if log["defects"] > defects_limit:
                    issue = f"Defectos: {log['defects']} excede {defects_limit}"
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
            "Informe de Cumplimiento de Defectos",
            "==================================",
            f"Período: {period}",
            f"Máquinas Analizadas: {len(machines)}",
            f"Registros Analizados: {total_records}",
            "",
            "Detalles por Máquina",
            "-------------------"
        ]

        for machine, data in results.items():
            if "error" in data:
                report.append(
                    f"Máquina: {machine}\n"
                    f"  Estado: Error\n"
                    f"  Mensaje: {data['error']}\n"
                    f"  Recomendación: {data['recommendation']}\n"
                )
                continue
            
            report.append(
                f"Máquina: {machine}\n"
                f"  Registros: {len(data['compliance_report'])}\n"
                f"  Límite: Defectos <= {data['defects_limit']}\n"
            )
            if not data["rules_found"]:
                report.append(f"  {data['warning_message']}\n")
            
            report.append("  Registros:\n")
            for i, entry in enumerate(data['compliance_report'], 1):
                status = "Conforme" if entry["compliant"] else "No Conforme"
                issue_text = f"    - {entry['issue']}" if entry["issue"] else "    Ninguno"
                report.append(
                    f"    Registro {i}:\n"
                    f"      ID: {entry['id']}\n"
                    f"      Fecha: {entry['date']}\n"
                    f"      Línea: {entry['production_line']}\n"
                    f"      Material: {entry['material']}\n"
                    f"      Métricas:\n"
                    f"        - Uptime: {entry['metrics']['uptime']}%\n"
                    f"        - Defectos: {entry['metrics']['defects']}\n"
                    f"        - Vibración: {entry['metrics']['vibration']} mm/s\n"
                    f"        - Temperatura: {entry['metrics']['temperature']}°C\n"
                    f"        - Rendimiento: {entry['metrics']['throughput']} u/h\n"
                    f"        - Inventario: {entry['metrics']['inventory_level']} u\n"
                    f"      Tipo Defecto: {entry['defect_type']}\n"
                    f"      Estado: {status}\n"
                    f"      Problema: {issue_text}\n"
                )
            
            report.append(
                f"  Resumen:\n"
                f"    No Conformes: {data['non_compliant_count']}\n"
            )

        report.extend([
            "Instrucciones",
            "-------------",
            f"Se verificaron {total_records} registros de {len(machines)} máquinas en {period} contra límites de defectos en PDFs. "
            "Los registros 'No Conforme' exceden los límites. Revise los detalles para tomar acciones correctivas."
        ])
        
        return "\n".join(report)
        
    except Exception as e:
        logger.error(f"Error en check_defects_compliance: {str(e)}")
        return (
            "Informe de Cumplimiento de Defectos\n"
            "==================================\n"
            "Estado: Error\n"
            f"Mensaje: No se pudo verificar el cumplimiento.\n"
            f"Detalles: {str(e)}\n"
            "Recomendación: Contacte al soporte técnico."
        )

if __name__ == "__main__":
    mcp.run()