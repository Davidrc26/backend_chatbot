#!/bin/bash

# Script de evaluación automatizada del modelo RAG
# Compara respuestas del modelo RAG con respuestas esperadas usando Gemini para medir similitud

# Solicitar provider al usuario
echo "Seleccione el provider para el endpoint RAG:"
echo "1) llama"
echo "2) gemini"
read -p "Ingrese su opción (1 o 2): " option

case $option in
    1)
        RAG_PROVIDER="llama"
        ;;
    2)
        RAG_PROVIDER="gemini"
        ;;
    *)
        echo "Opción inválida. Usando 'llama' por defecto."
        RAG_PROVIDER="llama"
        ;;
esac

echo ""
echo "Provider seleccionado: $RAG_PROVIDER"
echo ""

# Configuración
API_BASE_URL="http://192.168.1.40:8000/api/v1"
RAG_ENDPOINT="${API_BASE_URL}/chat/rag?provider=${RAG_PROVIDER}"
SIMPLE_ENDPOINT="${API_BASE_URL}/chat/simple?provider=gemini"
INPUT_FILE="app/assets/preguntas_respuestas.json"
OUTPUT_FILE="resultados_evaluacion_${RAG_PROVIDER}.json"
TEMP_FILE="temp_evaluation.json"

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Inicializar archivo de resultados
echo "{" > "$OUTPUT_FILE"
echo "  \"provider_rag\": \"$RAG_PROVIDER\"," >> "$OUTPUT_FILE"
echo "  \"fecha_evaluacion\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"," >> "$OUTPUT_FILE"
echo "  \"resultados\": [" >> "$OUTPUT_FILE"

# Contador de evaluaciones
count=0
total_similitud=0
request_count=0
request_limit=8
start_time=$(date +%s)

# Función para controlar rate limit
check_rate_limit() {
    request_count=$((request_count + 1))
    
    if [ $request_count -ge $request_limit ]; then
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -lt 60 ]; then
            wait_time=$((60 - elapsed))
            echo -e "${YELLOW}  ⏸ Límite de $request_limit peticiones alcanzado. Esperando ${wait_time}s...${NC}"
            sleep $wait_time
        fi
        
        request_count=0
        start_time=$(date +%s)
    fi
}

# Función para extraer porcentaje de similitud usando regex
extract_percentage() {
    local text="$1"
    percentage=$(echo "$text" | grep -oP '(\d+\.?\d*)\s*%|similitud[^0-9]*(\d+\.?\d*)|(\d+\.?\d*)\s*/\s*100|percentage[^0-9]*(\d+\.?\d*)' | grep -oP '\d+\.?\d*' | head -1)
    
    if [ -z "$percentage" ]; then
        percentage=$(echo "$text" | grep -oiP '(es|de|del|aproximadamente|around|about|roughly)?\s*(\d+\.?\d*)\s*(por ciento|porciento|percent)?' | grep -oP '\d+\.?\d*' | head -1)
    fi
    
    echo "$percentage"
}

# Función para hacer petición al endpoint RAG
get_rag_response() {
    local question="$1"
    local json_payload=$(python3 -c "import json, sys; print(json.dumps({'message': sys.argv[1], 'n_results': 3, 'use_rerank': True}))" "$question")
    
    local response=$(curl -s -X POST "$RAG_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    echo "$response" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('response', ''))"
}

# Función para calcular similitud usando Gemini
calculate_similarity() {
    local expected="$1"
    local received="$2"
    
    local prompt="Compara estas dos respuestas y dame SOLO un porcentaje numérico de similitud (0-100) considerando tanto el contenido semántico como la información clave que transmiten. Respuesta esperada: $expected Respuesta recibida: $received Responde ÚNICAMENTE con un número entre 0 y 100 seguido del símbolo %. Ejemplo: 85%"
    
    local json_payload=$(python3 -c "import json, sys; print(json.dumps({'message': sys.argv[1], 'n_results': 3, 'use_rerank': False}))" "$prompt")
    
    local response=$(curl -s -X POST "$SIMPLE_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    echo "$response" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data.get('response', ''))"
}

# Leer el archivo JSON y procesar cada pregunta
echo -e "${GREEN}Iniciando evaluación del modelo RAG...${NC}\n"

# Crear archivo temporal con las preguntas
TEMP_QUESTIONS="${TEMP_FILE}.questions"
python3 -c "
import json
with open('$INPUT_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data.get('preguntas_test_ia', []):
    archivo = item.get('archivo', '')
    for q in item.get('preguntas', []):
        print(json.dumps({'archivo': archivo, 'pregunta': q.get('pregunta', ''), 'respuesta': q.get('respuesta', '')}, ensure_ascii=False))
" > "$TEMP_QUESTIONS"

# Procesar cada pregunta
while IFS= read -r item; do
    archivo=$(echo "$item" | python3 -c "import json, sys; print(json.loads(sys.stdin.read()).get('archivo', ''))")
    pregunta=$(echo "$item" | python3 -c "import json, sys; print(json.loads(sys.stdin.read()).get('pregunta', ''))")
    respuesta_esperada=$(echo "$item" | python3 -c "import json, sys; print(json.loads(sys.stdin.read()).get('respuesta', ''))")
    
    count=$((count + 1))
    
    echo -e "${YELLOW}[$count] Evaluando pregunta:${NC}"
    echo "Archivo: $archivo"
    echo "Pregunta: ${pregunta:0:80}..."
    echo ""
    
    echo "  → Consultando endpoint RAG..."
    check_rate_limit
    respuesta_recibida=$(get_rag_response "$pregunta")
    sleep 2
    
    if [ -z "$respuesta_recibida" ]; then
        echo -e "${RED}  ✗ Error: No se obtuvo respuesta del RAG${NC}\n"
        respuesta_recibida="ERROR: Sin respuesta"
        similitud="0"
    else
        echo -e "${GREEN}  ✓ Respuesta RAG obtenida${NC}"
        echo ""
        
        echo "  → Calculando similitud con Gemini..."
        check_rate_limit
        similitud_response=$(calculate_similarity "$respuesta_esperada" "$respuesta_recibida")
        sleep 2
        
        similitud=$(extract_percentage "$similitud_response")
        
        if [ -z "$similitud" ]; then
            echo -e "${RED}  ✗ No se pudo extraer porcentaje${NC}"
            similitud="0"
        else
            echo -e "${GREEN}  ✓ Similitud calculada: ${similitud}%${NC}"
            total_similitud=$(echo "$total_similitud + $similitud" | bc)
        fi
    fi
    
    echo ""
    echo "─────────────────────────────────────────"
    echo ""
    
    pregunta_escaped=$(echo "$pregunta" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))")
    respuesta_esperada_escaped=$(echo "$respuesta_esperada" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))")
    respuesta_recibida_escaped=$(echo "$respuesta_recibida" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))")
    archivo_escaped=$(echo "$archivo" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read()))")
    
    if [ $count -gt 1 ]; then
        echo "    ," >> "$OUTPUT_FILE"
    fi
    
    echo "    {" >> "$OUTPUT_FILE"
    echo "      \"id\": $count," >> "$OUTPUT_FILE"
    echo "      \"archivo\": $archivo_escaped," >> "$OUTPUT_FILE"
    echo "      \"pregunta\": $pregunta_escaped," >> "$OUTPUT_FILE"
    echo "      \"respuesta_esperada\": $respuesta_esperada_escaped," >> "$OUTPUT_FILE"
    echo "      \"respuesta_recibida\": $respuesta_recibida_escaped," >> "$OUTPUT_FILE"
    echo "      \"similitud\": $similitud," >> "$OUTPUT_FILE"
    echo "      \"fecha\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"" >> "$OUTPUT_FILE"
    echo "    }" >> "$OUTPUT_FILE"
    
done < "$TEMP_QUESTIONS"

rm -f "$TEMP_QUESTIONS"

if [ $count -gt 0 ]; then
    promedio=$(echo "scale=2; $total_similitud / $count" | bc)
else
    promedio="0"
fi

echo "" >> "$OUTPUT_FILE"
echo "  ]," >> "$OUTPUT_FILE"
echo "  \"resumen\": {" >> "$OUTPUT_FILE"
echo "    \"provider_rag\": \"$RAG_PROVIDER\"," >> "$OUTPUT_FILE"
echo "    \"total_preguntas\": $count," >> "$OUTPUT_FILE"
echo "    \"similitud_promedio\": $promedio" >> "$OUTPUT_FILE"
echo "  }" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     EVALUACIÓN COMPLETADA              ║${NC}"
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}  Provider RAG usado: $RAG_PROVIDER${NC}"
echo -e "${GREEN}  Total de preguntas evaluadas: $count${NC}"
echo -e "${GREEN}  Similitud promedio: ${promedio}%${NC}"
echo -e "${GREEN}  Resultados guardados en: $OUTPUT_FILE${NC}"
echo ""
