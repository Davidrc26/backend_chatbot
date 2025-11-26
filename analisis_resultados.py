#!/usr/bin/env python3
"""
Script para analizar y comparar resultados de evaluación RAG
"""
import json
from typing import Dict, List
from pathlib import Path


class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def analyze_results(file_path: str) -> Dict:
    """Analiza un archivo de resultados"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    resultados = data['resultados']
    valid_results = [r for r in resultados if r['score_final'] > 0]
    zero_results = [r for r in resultados if r['score_final'] == 0]
    
    if not valid_results:
        return None
    
    analysis = {
        'provider': data.get('provider_rag', 'unknown'),
        'total_preguntas': len(resultados),
        'preguntas_validas': len(valid_results),
        'preguntas_con_zero': len(zero_results),
        'tasa_exito': (len(valid_results) / len(resultados)) * 100,
        'avg_score': sum(r['score_final'] for r in valid_results) / len(valid_results),
        'avg_exactitud': sum(r['scores']['exactitud'] for r in valid_results) / len(valid_results),
        'avg_cobertura': sum(r['scores']['cobertura'] for r in valid_results) / len(valid_results),
        'avg_claridad': sum(r['scores']['claridad'] for r in valid_results) / len(valid_results),
        'avg_citas': sum(r['scores']['citas'] for r in valid_results) / len(valid_results),
        'avg_alucinacion': sum(r['scores']['alucinacion'] for r in valid_results) / len(valid_results),
        'avg_seguridad': sum(r['scores']['seguridad'] for r in valid_results) / len(valid_results),
        'max_score': max(r['score_final'] for r in valid_results),
        'min_score': min(r['score_final'] for r in valid_results),
        'top5': sorted(valid_results, key=lambda x: x['score_final'], reverse=True)[:5],
        'bottom5': sorted(valid_results, key=lambda x: x['score_final'])[:5]
    }
    
    return analysis


def print_analysis(analysis: Dict):
    """Imprime el análisis de forma bonita"""
    if not analysis:
        print(f"{Colors.RED}No hay datos válidos para analizar{Colors.NC}")
        return
    
    provider = analysis['provider'].upper()
    
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
    print(f"{Colors.CYAN}📊 ANÁLISIS DE RESULTADOS - {provider}{Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
    
    # Estadísticas generales
    print(f"{Colors.YELLOW}📈 ESTADÍSTICAS GENERALES:{Colors.NC}")
    print(f"  • Total de preguntas: {analysis['total_preguntas']}")
    print(f"  • Preguntas válidas (score > 0): {Colors.GREEN}{analysis['preguntas_validas']}{Colors.NC}")
    print(f"  • Preguntas con score = 0: {Colors.RED}{analysis['preguntas_con_zero']}{Colors.NC}")
    print(f"  • Tasa de éxito: {Colors.GREEN}{analysis['tasa_exito']:.1f}%{Colors.NC}")
    print()
    
    # Promedios
    print(f"{Colors.YELLOW}📊 PROMEDIOS (solo scores > 0):{Colors.NC}")
    print(f"  {Colors.GREEN}🎯 Score Final: {analysis['avg_score']:.2f}/100{Colors.NC}")
    print(f"  • Exactitud:     {analysis['avg_exactitud']:.2f}/100")
    print(f"  • Cobertura:     {analysis['avg_cobertura']:.2f}/100")
    print(f"  • Claridad:      {analysis['avg_claridad']:.2f}/100")
    print(f"  • Citas:         {analysis['avg_citas']:.2f}/100")
    print(f"  • Sin Alucinación: {analysis['avg_alucinacion']:.2f}/100")
    print(f"  • Seguridad:     {analysis['avg_seguridad']:.2f}/100")
    print()
    
    # Rango
    print(f"{Colors.YELLOW}📐 RANGO DE SCORES:{Colors.NC}")
    print(f"  • Máximo: {Colors.GREEN}{analysis['max_score']:.1f}{Colors.NC}")
    print(f"  • Mínimo: {Colors.RED}{analysis['min_score']:.1f}{Colors.NC}")
    print(f"  • Rango:  {analysis['max_score'] - analysis['min_score']:.1f} puntos")
    print()
    
    # Top 5
    print(f"{Colors.GREEN}🏆 TOP 5 MEJORES SCORES:{Colors.NC}")
    for i, r in enumerate(analysis['top5'], 1):
        print(f"  {i}. {Colors.GREEN}{r['score_final']:.1f}{Colors.NC} - Q{r['id']}: {r['pregunta'][:55]}...")
    print()
    
    # Bottom 5
    print(f"{Colors.YELLOW}⚠️  5 SCORES MÁS BAJOS (pero > 0):{Colors.NC}")
    for i, r in enumerate(analysis['bottom5'], 1):
        print(f"  {i}. {Colors.YELLOW}{r['score_final']:.1f}{Colors.NC} - Q{r['id']}: {r['pregunta'][:55]}...")
    print()


def compare_providers(analyses: List[Dict]):
    """Compara múltiples providers"""
    if len(analyses) < 2:
        return
    
    print(f"\n{Colors.CYAN}{'=' * 70}{Colors.NC}")
    print(f"{Colors.CYAN}⚖️  COMPARACIÓN ENTRE PROVIDERS{Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 70}{Colors.NC}\n")
    
    # Tabla comparativa
    print(f"{'Métrica':<20} | ", end="")
    for a in analyses:
        print(f"{a['provider'].upper():^15} | ", end="")
    print()
    print("-" * 70)
    
    metrics = [
        ('Tasa de éxito', 'tasa_exito', '%'),
        ('Score promedio', 'avg_score', '/100'),
        ('Exactitud', 'avg_exactitud', '/100'),
        ('Cobertura', 'avg_cobertura', '/100'),
        ('Claridad', 'avg_claridad', '/100'),
        ('Citas', 'avg_citas', '/100'),
        ('Sin Alucinación', 'avg_alucinacion', '/100'),
        ('Seguridad', 'avg_seguridad', '/100'),
    ]
    
    for label, key, unit in metrics:
        print(f"{label:<20} | ", end="")
        values = [a[key] for a in analyses]
        best_idx = values.index(max(values))
        
        for i, a in enumerate(analyses):
            val = a[key]
            if i == best_idx:
                print(f"{Colors.GREEN}{val:>10.2f}{unit:>5}{Colors.NC} | ", end="")
            else:
                print(f"{val:>10.2f}{unit:>5} | ", end="")
        print()
    
    print()
    
    # Ganador general
    best_provider = max(analyses, key=lambda x: x['avg_score'])
    print(f"{Colors.GREEN}🏆 GANADOR GENERAL: {best_provider['provider'].upper()}{Colors.NC}")
    print(f"   Score promedio: {best_provider['avg_score']:.2f}/100")
    print()


def main():
    """Función principal"""
    # Buscar archivos de resultados
    files = list(Path('.').glob('resultados_evaluacion_*.json'))
    
    if not files:
        print(f"{Colors.RED}No se encontraron archivos de resultados{Colors.NC}")
        return
    
    print(f"\n{Colors.CYAN}🔍 Archivos de resultados encontrados:{Colors.NC}")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    print()
    
    analyses = []
    
    for file_path in files:
        try:
            analysis = analyze_results(str(file_path))
            if analysis:
                analyses.append(analysis)
                print_analysis(analysis)
        except Exception as e:
            print(f"{Colors.RED}Error analizando {file_path}: {e}{Colors.NC}")
    
    # Comparar si hay múltiples providers
    if len(analyses) >= 2:
        compare_providers(analyses)
    elif len(analyses) == 1:
        print(f"\n{Colors.YELLOW}💡 Ejecuta la evaluación con otro provider para comparar{Colors.NC}")
        print(f"   Providers disponibles: llama, gemini")
        print()


if __name__ == "__main__":
    main()
