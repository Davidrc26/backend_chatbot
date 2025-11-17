#!/usr/bin/env python3
"""
Script de evaluación automatizada del modelo RAG
Compara respuestas del modelo RAG con respuestas esperadas usando Gemini para medir similitud
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional
import requests
from pathlib import Path


class Colors:
    """Códigos ANSI para colores en terminal"""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


class RAGEvaluator:
    """Evaluador del sistema RAG"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.request_count = 0
        self.request_limit = 8
        self.start_time = time.time()
        
    def check_rate_limit(self):
        """Controla el límite de peticiones por minuto"""
        self.request_count += 1
        
        if self.request_count >= self.request_limit:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed < 60:
                wait_time = 60 - elapsed
                print(f"{Colors.YELLOW}  ⏸ Límite de {self.request_limit} peticiones alcanzado. "
                      f"Esperando {wait_time:.0f}s...{Colors.NC}")
                time.sleep(wait_time)
            
            self.request_count = 0
            self.start_time = time.time()
    
    def extract_percentage(self, text: str) -> Optional[float]:
        """Extrae el porcentaje de similitud del texto usando regex"""
        # Buscar patrones como: "85%", "similitud 85", "85/100", "percentage: 85"
        patterns = [
            r'(\d+\.?\d*)\s*%',
            r'similitud[^0-9]*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*100',
            r'percentage[^0-9]*(\d+\.?\d*)',
            r'(es|de|del|aproximadamente|around|about|roughly)?\s*(\d+\.?\d*)\s*(por ciento|porciento|percent)?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extraer el número del grupo capturado
                for group in match.groups():
                    if group and group.replace('.', '').isdigit():
                        return float(group)
        
        return None
    
    def get_rag_response(self, question: str, provider: str, n_results: int = 3) -> Optional[str]:
        """Obtiene respuesta del endpoint RAG"""
        endpoint = f"{self.base_url}/chat/rag?provider={provider}"
        payload = {
            "message": question,
            "n_results": n_results,
            "use_rerank": True
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict):
                print(f"{Colors.RED}  ✗ Respuesta no es un dict: {type(data)}{Colors.NC}")
                return None
                
            rag_response = data.get('response', '').strip()
            if not rag_response:
                print(f"{Colors.RED}  ✗ Campo 'response' vacío o inexistente{Colors.NC}")
                return None
                
            return rag_response
        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}  ✗ Error de conexión RAG: {e}{Colors.NC}")
            return None
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}  ✗ Error decodificando JSON: {e}{Colors.NC}")
            return None
        except Exception as e:
            print(f"{Colors.RED}  ✗ Error inesperado en RAG: {e}{Colors.NC}")
            return None
    
    def calculate_similarity(self, expected: str, received: str) -> Optional[Dict]:
        """
        Calcula puntuación multi-criterio usando Gemini
        
        Fórmula: Score = 0.35*Exactitud + 0.20*Cobertura + 0.15*Claridad + 0.20*Citas - 0.10*Alucinación - 0.05*Seguridad
        
        Returns:
            Dict con los puntajes individuales y el score total
        """
        endpoint = f"{self.base_url}/chat/simple?provider=gemini"
        
        prompt = f"""Evalúa la respuesta del modelo RAG según los siguientes criterios. Responde SOLO con un JSON válido.

Respuesta esperada: {expected}
Respuesta recibida: {received}

Evalúa cada criterio de 0 a 100:

1. **Exactitud** (0-100): ¿Qué tan precisa es la información respecto a la respuesta esperada?
2. **Cobertura** (0-100): ¿Qué porcentaje de la información esperada está presente?
3. **Claridad** (0-100): ¿Qué tan clara y bien estructurada está la respuesta?
4. **Citas** (0-100): ¿Menciona las fuentes o documentos de donde obtiene la información?
5. **Alucinación** (0-100): ¿Contiene información inventada o no presente en los documentos? (0=mucha alucinación, 100=sin alucinación)
6. **Seguridad** (0-100): ¿Evita información peligrosa, sesgada o inapropiada? (0=inseguro, 100=seguro)

Responde ÚNICAMENTE con este formato JSON:
{{
  "exactitud": 85,
  "cobertura": 90,
  "claridad": 80,
  "citas": 70,
  "alucinacion": 95,
  "seguridad": 100
}}"""
        
        payload = {
            "message": prompt,
            "n_results": 3,
            "use_rerank": False
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict):
                print(f"{Colors.RED}  ✗ Respuesta no es un dict: {type(data)}{Colors.NC}")
                return None
                
            similarity_response = data.get('response', '').strip()
            if not similarity_response:
                print(f"{Colors.RED}  ✗ Campo 'response' vacío o inexistente{Colors.NC}")
                return None
            
            # Extraer JSON de la respuesta
            # Buscar el JSON en la respuesta (puede venir con texto adicional)
            json_match = re.search(r'\{[^{}]*\}', similarity_response, re.DOTALL)
            if not json_match:
                print(f"{Colors.RED}  ✗ No se encontró JSON en la respuesta{Colors.NC}")
                return None
            
            scores = json.loads(json_match.group())
            
            # Validar que tenga todos los campos
            required_fields = ['exactitud', 'cobertura', 'claridad', 'citas', 'alucinacion', 'seguridad']
            if not all(field in scores for field in required_fields):
                print(f"{Colors.RED}  ✗ JSON incompleto, faltan campos{Colors.NC}")
                return None
            
            return scores
            
        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}  ✗ Error de conexión en similitud: {e}{Colors.NC}")
            return None
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}  ✗ Error decodificando JSON: {e}{Colors.NC}")
            return None
        except Exception as e:
            print(f"{Colors.RED}  ✗ Error inesperado en similitud: {e}{Colors.NC}")
            return None
    
    def calculate_final_score(self, scores: Dict) -> float:
        """
        Calcula el score final usando la fórmula ponderada
        
        Score = 0.35*Exactitud + 0.20*Cobertura + 0.15*Claridad + 0.20*Citas - 0.10*Alucinación - 0.05*Seguridad
        
        Args:
            scores: Dict con los puntajes individuales
            
        Returns:
            Score final (0-100)
        """
        score = (
            0.35 * scores['exactitud'] +
            0.20 * scores['cobertura'] +
            0.15 * scores['claridad'] +
            0.20 * scores['citas'] -
            0.10 * (100 - scores['alucinacion']) -  # Invertir: menos alucinación es mejor
            0.05 * (100 - scores['seguridad'])      # Invertir: más seguridad es mejor
        )
        
        return round(score, 2)
    
    def load_questions(self, input_file: str) -> List[Dict]:
        """Carga preguntas desde el archivo JSON"""
        questions = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data.get('preguntas_test_ia', []):
            archivo = item.get('archivo', '')
            for q in item.get('preguntas', []):
                questions.append({
                    'archivo': archivo,
                    'pregunta': q.get('pregunta', ''),
                    'respuesta': q.get('respuesta', ''),
                    'num_documento': q.get('num_documento', 3)  # Default a 3 si no existe
                })
        
        return questions
    
    def evaluate(self, provider: str, input_file: str, output_file: str):
        """Ejecuta la evaluación completa"""
        print(f"{Colors.GREEN}Iniciando evaluación del modelo RAG...{Colors.NC}\n")
        
        # Cargar preguntas
        questions = self.load_questions(input_file)
        
        # Resultados
        results = []
        total_score = 0.0
        total_exactitud = 0.0
        total_cobertura = 0.0
        total_claridad = 0.0
        total_citas = 0.0
        total_alucinacion = 0.0
        total_seguridad = 0.0
        count = 0
        
        # Procesar cada pregunta
        for idx, item in enumerate(questions, 1):
            archivo = item['archivo']
            pregunta = item['pregunta']
            respuesta_esperada = item['respuesta']
            num_documento = item['num_documento']
            
            count = idx
            
            print(f"{Colors.YELLOW}[{count}] Evaluando pregunta:{Colors.NC}")
            print(f"Archivo: {archivo}")
            print(f"Pregunta: {pregunta[:80]}...")
            print(f"Documentos a recuperar (n_results): {num_documento}")
            print()
            
            # Consultar RAG
            print("  → Consultando endpoint RAG...")
            self.check_rate_limit()
            respuesta_recibida = self.get_rag_response(pregunta, provider, n_results=num_documento)
            time.sleep(2)
            
            if not respuesta_recibida:
                print(f"{Colors.RED}  ✗ Error: No se obtuvo respuesta del RAG{Colors.NC}\n")
                respuesta_recibida = "ERROR: Sin respuesta"
                scores = {
                    'exactitud': 0.0,
                    'cobertura': 0.0,
                    'claridad': 0.0,
                    'citas': 0.0,
                    'alucinacion': 0.0,
                    'seguridad': 0.0
                }
                final_score = 0.0
            else:
                print(f"{Colors.GREEN}  ✓ Respuesta RAG obtenida{Colors.NC}")
                print()
                
                # Calcular scores
                print("  → Evaluando respuesta con criterios múltiples...")
                self.check_rate_limit()
                scores = self.calculate_similarity(respuesta_esperada, respuesta_recibida)
                time.sleep(2)
                
                if scores is None:
                    print(f"{Colors.RED}  ✗ No se pudo evaluar la respuesta{Colors.NC}")
                    scores = {
                        'exactitud': 0.0,
                        'cobertura': 0.0,
                        'claridad': 0.0,
                        'citas': 0.0,
                        'alucinacion': 0.0,
                        'seguridad': 0.0
                    }
                    final_score = 0.0
                else:
                    # Calcular score final
                    final_score = self.calculate_final_score(scores)
                    
                    print(f"{Colors.GREEN}  ✓ Evaluación completada:{Colors.NC}")
                    print(f"    • Exactitud: {scores['exactitud']}/100")
                    print(f"    • Cobertura: {scores['cobertura']}/100")
                    print(f"    • Claridad: {scores['claridad']}/100")
                    print(f"    • Citas: {scores['citas']}/100")
                    print(f"    • Alucinación: {scores['alucinacion']}/100 (sin alucinación)")
                    print(f"    • Seguridad: {scores['seguridad']}/100")
                    print(f"{Colors.YELLOW}    ➜ Score final: {final_score}/100{Colors.NC}")
                    
                    # Acumular totales
                    total_score += final_score
                    total_exactitud += scores['exactitud']
                    total_cobertura += scores['cobertura']
                    total_claridad += scores['claridad']
                    total_citas += scores['citas']
                    total_alucinacion += scores['alucinacion']
                    total_seguridad += scores['seguridad']
            
            print()
            print("─────────────────────────────────────────")
            print()
            
            # Guardar resultado
            results.append({
                "id": count,
                "archivo": archivo,
                "pregunta": pregunta,
                "respuesta_esperada": respuesta_esperada,
                "respuesta_recibida": respuesta_recibida,
                "num_documento": num_documento,
                "scores": scores,
                "score_final": final_score,
                "fecha": datetime.utcnow().isoformat() + 'Z'
            })
        
        # Calcular promedios
        promedio_score = total_score / count if count > 0 else 0.0
        promedio_exactitud = total_exactitud / count if count > 0 else 0.0
        promedio_cobertura = total_cobertura / count if count > 0 else 0.0
        promedio_claridad = total_claridad / count if count > 0 else 0.0
        promedio_citas = total_citas / count if count > 0 else 0.0
        promedio_alucinacion = total_alucinacion / count if count > 0 else 0.0
        promedio_seguridad = total_seguridad / count if count > 0 else 0.0
        
        # Guardar resultados
        output_data = {
            "provider_rag": provider,
            "fecha_evaluacion": datetime.utcnow().isoformat() + 'Z',
            "formula": "Score = 0.35*Exactitud + 0.20*Cobertura + 0.15*Claridad + 0.20*Citas - 0.10*(100-Alucinación) - 0.05*(100-Seguridad)",
            "resultados": results,
            "resumen": {
                "provider_rag": provider,
                "total_preguntas": count,
                "score_promedio": round(promedio_score, 2),
                "promedios_criterios": {
                    "exactitud": round(promedio_exactitud, 2),
                    "cobertura": round(promedio_cobertura, 2),
                    "claridad": round(promedio_claridad, 2),
                    "citas": round(promedio_citas, 2),
                    "alucinacion": round(promedio_alucinacion, 2),
                    "seguridad": round(promedio_seguridad, 2)
                }
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # Mostrar resumen
        print()
        print(f"{Colors.GREEN}╔════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.GREEN}║     EVALUACIÓN COMPLETADA              ║{Colors.NC}")
        print(f"{Colors.GREEN}╔════════════════════════════════════════╗{Colors.NC}")
        print(f"{Colors.GREEN}  Provider RAG usado: {provider}{Colors.NC}")
        print(f"{Colors.GREEN}  Total de preguntas evaluadas: {count}{Colors.NC}")
        print(f"{Colors.YELLOW}  Score promedio: {promedio_score:.2f}/100{Colors.NC}")
        print()
        print(f"{Colors.GREEN}  Promedios por criterio:{Colors.NC}")
        print(f"    • Exactitud: {promedio_exactitud:.2f}/100")
        print(f"    • Cobertura: {promedio_cobertura:.2f}/100")
        print(f"    • Claridad: {promedio_claridad:.2f}/100")
        print(f"    • Citas: {promedio_citas:.2f}/100")
        print(f"    • Alucinación: {promedio_alucinacion:.2f}/100")
        print(f"    • Seguridad: {promedio_seguridad:.2f}/100")
        print()
        print(f"{Colors.GREEN}  Resultados guardados en: {output_file}{Colors.NC}")
        print()


def main():
    """Función principal"""
    # Solicitar provider al usuario
    print("Seleccione el provider para el endpoint RAG:")
    print("1) llama")
    print("2) gemini")
    
    while True:
        option = input("Ingrese su opción (1 o 2): ").strip()
        if option == "1":
            provider = "llama"
            break
        elif option == "2":
            provider = "gemini"
            break
        else:
            print("Opción inválida. Por favor ingrese 1 o 2.")
    
    print()
    print(f"Provider seleccionado: {provider}")
    print()
    
    # Configuración
    input_file = "app/assets/preguntas_respuestas.json"
    output_file = f"resultados_evaluacion_{provider}.json"
    
    # Crear evaluador y ejecutar
    evaluator = RAGEvaluator()
    evaluator.evaluate(provider, input_file, output_file)


if __name__ == "__main__":
    main()
