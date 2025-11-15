# Evaluación Automatizada del Modelo RAG

Script de evaluación automatizada que compara respuestas del modelo RAG con respuestas esperadas utilizando Gemini para medir la similitud semántica.

## 📋 Requisitos

### Sistema Operativo
- **Linux** (Probado en Ubuntu/Debian)
- **macOS** (Compatible)
- **Windows**: Requiere WSL (Windows Subsystem for Linux)

### Dependencias
- `bash` (shell)
- `python3` (para procesamiento JSON)
- `curl` (para peticiones HTTP)
- `bc` (para cálculos matemáticos)

### Instalación de Dependencias

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y python3 curl bc
```

**macOS:**
```bash
brew install python3 curl bc
```

**Windows (WSL):**
```bash
sudo apt-get update
sudo apt-get install -y python3 curl bc
```

## ⚙️ Configuración

### 1. Configurar el Host de la API

Edita el archivo `evaluate_rag.sh` y modifica la línea 29 con la URL de tu API:

```bash
API_BASE_URL="http://TU_HOST:8000/api/v1"
```

**Ejemplos de configuración:**

- **Localhost:** `API_BASE_URL="http://localhost:8000/api/v1"`
- **Red local:** `API_BASE_URL="http://192.168.1.40:8000/api/v1"`
- **Servidor remoto:** `API_BASE_URL="http://mi-servidor.com:8000/api/v1"`
- **Docker:** `API_BASE_URL="http://host.docker.internal:8000/api/v1"`

### 2. Preparar el Archivo de Preguntas

Crea o edita el archivo `preguntas_respuestas.json` con el siguiente formato:

```json
{
  "preguntas_test_ia": [
    {
      "archivo": "nombre_del_documento.pdf",
      "preguntas": [
        {
          "pregunta": "¿Cuál es la pregunta de prueba?",
          "respuesta": "Respuesta esperada del modelo"
        }
      ]
    }
  ]
}
```

### 3. Dar Permisos de Ejecución

```bash
chmod +x evaluate_rag.sh
```

## 🚀 Uso

### Ejecutar la Evaluación

```bash
./evaluate_rag.sh
```

El script te pedirá que selecciones el provider:

```
Seleccione el provider para el endpoint RAG:
1) llama
2) gemini
Ingrese su opción (1 o 2):
```

### Opciones de Provider

- **llama**: Utiliza el modelo Llama para RAG
- **gemini**: Utiliza el modelo Gemini para RAG

## 📊 Resultados

### Archivo de Salida

Los resultados se guardan en: `resultados_evaluacion_[provider].json`

**Ejemplo:** `resultados_evaluacion_gemini.json`

### Formato de Resultados

```json
{
  "provider_rag": "gemini",
  "fecha_evaluacion": "2025-11-15T10:30:00Z",
  "resultados": [
    {
      "id": 1,
      "archivo": "documento.pdf",
      "pregunta": "¿Pregunta de prueba?",
      "respuesta_esperada": "Respuesta esperada",
      "respuesta_recibida": "Respuesta del modelo",
      "similitud": 85.5,
      "fecha": "2025-11-15T10:30:15Z"
    }
  ],
  "resumen": {
    "provider_rag": "gemini",
    "total_preguntas": 10,
    "similitud_promedio": 82.3
  }
}
```

### Interpretación de Resultados

- **similitud**: Porcentaje de similitud semántica (0-100)
- **similitud_promedio**: Promedio de todas las evaluaciones
- Valores > 80%: Excelente similitud
- Valores 60-80%: Buena similitud
- Valores < 60%: Requiere revisión

## 🔧 Características Técnicas

### Control de Rate Limit

El script implementa control automático de tasa de peticiones:

- **Límite:** 30 peticiones por minuto
- **Pausa automática:** Espera cuando se alcanza el límite
- **Delay entre peticiones:** 2 segundos entre cada llamada

### Procesamiento JSON

Utiliza Python3 para el procesamiento JSON (no requiere `jq`):
- Parsing de archivos de entrada
- Creación de payloads para API
- Extracción de respuestas
- Escape de strings para JSON de salida

## 🐛 Solución de Problemas

### Error: "Permission denied"

```bash
chmod +x evaluate_rag.sh
```

### Error: "python3: command not found"

Instala Python3 según tu sistema operativo (ver sección de dependencias).

### Error: "curl: command not found"

Instala curl según tu sistema operativo (ver sección de dependencias).

### Error: "Connection refused"

Verifica que:
1. La API esté ejecutándose
2. El host y puerto sean correctos en `API_BASE_URL`
3. No haya firewall bloqueando la conexión

### Error: "Rate limit exceeded"

El script ya maneja esto automáticamente. Si sigues teniendo problemas:
- Aumenta el `sleep` entre peticiones (líneas 147 y 159)
- Reduce el `request_limit` (línea 51)


## 🔄 Flujo de Ejecución

1. Selección del provider (llama/gemini)
2. Carga de preguntas desde `preguntas_respuestas.json`
3. Por cada pregunta:
   - Consulta al endpoint RAG
   - Espera 2 segundos
   - Calcula similitud con Gemini
   - Espera 2 segundos
   - Verifica límite de peticiones
4. Guarda resultados en archivo JSON
5. Muestra resumen en consola

## 📄 Licencia

Este script es de uso libre para evaluación de modelos RAG.

## 👤 Autor

Script de evaluación automatizada para modelos RAG.

---

**Nota:** Asegúrate de configurar correctamente el `API_BASE_URL` antes de ejecutar el script.
