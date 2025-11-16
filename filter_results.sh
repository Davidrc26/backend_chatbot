#!/bin/bash

# Script para filtrar resultados con similitud = 0

INPUT_FILE="${1:-resultados_evaluacion_gemini.json}"
OUTPUT_FILE="${INPUT_FILE%.json}_filtered.json"

# Verificar que el archivo existe
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Archivo $INPUT_FILE no encontrado"
    exit 1
fi

# Filtrar resultados con similitud = 0
jq '{
    provider_rag: .provider_rag,
    fecha_evaluacion: .fecha_evaluacion,
    resultados: [.resultados[] | select(.similitud == 0)],
    resumen: {
        provider_rag: .resumen.provider_rag,
        total_preguntas_originales: .resumen.total_preguntas,
        total_preguntas_filtradas: ([.resultados[] | select(.similitud == 0)] | length),
        similitud_promedio_original: .resumen.similitud_promedio,
        similitud_promedio_filtrada: 0
    }
}' "$INPUT_FILE" > "$OUTPUT_FILE"

# Mostrar estadísticas
total_original=$(jq '.resumen.total_preguntas' "$INPUT_FILE")
total_filtrado=$(jq '.resumen.total_preguntas_filtradas' "$OUTPUT_FILE")
similitud_original=$(jq '.resumen.similitud_promedio_original' "$OUTPUT_FILE")

echo ""
echo "✅ Filtrado completado"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Estadísticas:"
echo "   Total original: $total_original preguntas"
echo "   Total filtrado: $total_filtrado preguntas (similitud = 0)"
echo "   Con similitud > 0: $((total_original - total_filtrado)) preguntas"
echo ""
echo "   Similitud promedio original: ${similitud_original}%"
echo ""
echo "📁 Archivo generado: $OUTPUT_FILE"
echo ""
