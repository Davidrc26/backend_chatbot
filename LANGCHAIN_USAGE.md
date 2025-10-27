# 📚 Guía de Uso: LangChain PDF Processing

## 🚀 Endpoint: `/documents/upload-file-langchain`

Este endpoint procesa PDFs usando **LangChain** con **UnstructuredPDFLoader** para una extracción más robusta y detallada.

---

## 📋 Parámetros

### Obligatorios:
- **`file`** (File): Archivo PDF a procesar

### Opcionales:
- **`provider`** (str, default: "llama"): 
  - `"llama"`: Usa Ollama con Llama 3.1
  - `"gemini"`: Usa Google Gemini 2.0 Flash

- **`chunk_size`** (int, default: 1000): Tamaño de cada chunk de texto

- **`chunk_overlap`** (int, default: 200): Overlap entre chunks consecutivos

- **`extraction_mode`** (str, default: "elements"): Modo de extracción del PDF
  - `"elements"`: 🎯 **Recomendado** - Divide por elementos (títulos, párrafos, tablas)
  - `"single"`: Todo el documento como un solo texto
  - `"paged"`: Divide por páginas

### Metadata Opcional:
- **`author`** (str): Autor del documento
- **`category`** (str): Categoría (ej: "Tutorial", "Manual", "Artículo")
- **`tags`** (str): Tags separados por comas (ej: "python,machine-learning,nlp")
- **`year`** (str): Año del documento

---

## 🔧 Ejemplos de Uso

### 1️⃣ Uso Básico (con Llama/Ollama)

```bash
curl -X POST "http://localhost:8000/documents/upload-file-langchain" \
  -F "file=@mi_documento.pdf" \
  -F "provider=llama"
```

### 2️⃣ Uso con Gemini y Metadata Completa

```bash
curl -X POST "http://localhost:8000/documents/upload-file-langchain" \
  -F "file=@tutorial_fastapi.pdf" \
  -F "provider=gemini" \
  -F "author=John Doe" \
  -F "category=Tutorial" \
  -F "tags=python,fastapi,api,backend" \
  -F "year=2024"
```

### 3️⃣ Configuración Personalizada de Chunks

```bash
curl -X POST "http://localhost:8000/documents/upload-file-langchain" \
  -F "file=@libro_largo.pdf" \
  -F "provider=llama" \
  -F "chunk_size=1500" \
  -F "chunk_overlap=300" \
  -F "extraction_mode=paged"
```

### 4️⃣ Usando Python `requests`

```python
import requests

url = "http://localhost:8000/documents/upload-file-langchain"

with open("mi_documento.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "provider": "gemini",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "extraction_mode": "elements",
        "author": "Jane Smith",
        "category": "Research Paper",
        "tags": "nlp,transformers,bert",
        "year": "2024"
    }
    
    response = requests.post(url, files=files, data=data)
    print(response.json())
```

### 5️⃣ Usando JavaScript `fetch`

```javascript
const formData = new FormData();
formData.append('file', pdfFile); // pdfFile es un File object
formData.append('provider', 'llama');
formData.append('chunk_size', '1000');
formData.append('chunk_overlap', '200');
formData.append('extraction_mode', 'elements');
formData.append('author', 'Juan Pérez');
formData.append('category', 'Manual');
formData.append('tags', 'python,langchain,rag');
formData.append('year', '2024');

const response = await fetch('http://localhost:8000/documents/upload-file-langchain', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result);
```

---

## 📊 Respuesta Exitosa

```json
{
    "id": "documents_langchain_llama",
    "message": "PDF procesado con LangChain (UnstructuredPDFLoader). 15 documentos extraídos, 42 chunks creados. | Autor: John Doe | Categoría: Tutorial | Año: 2024"
}
```

### Campos de la Respuesta:
- **`id`**: Nombre de la colección en ChromaDB (`documents_langchain_{provider}`)
- **`message`**: Resumen del procesamiento con estadísticas y metadata

---

## 🎯 Modos de Extracción - ¿Cuál Usar?

### `"elements"` (Recomendado) 🌟
- **Mejor para**: PDFs con estructura compleja (artículos, manuales, libros)
- **Ventaja**: Preserva la estructura del documento (títulos, secciones, tablas)
- **Desventaja**: Puede ser más lento

### `"single"`
- **Mejor para**: PDFs cortos o cuando quieres todo el contenido junto
- **Ventaja**: Más rápido
- **Desventaja**: Pierde estructura del documento

### `"paged"`
- **Mejor para**: PDFs donde la división por página es importante
- **Ventaja**: Mantiene contexto de página
- **Desventaja**: Puede cortar secciones lógicas

---

## 🔍 Consultar los Documentos

Después de subir documentos con LangChain, puedes consultarlos usando los endpoints de chat:

### Endpoint RAG Simple: `/chat/rag`

```bash
curl -X POST "http://localhost:8000/chat/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Qué dice el documento sobre FastAPI?",
    "provider": "llama",
    "use_langchain": true
  }'
```

### Endpoint RAG Conversacional: `/chat/conversation`

```bash
curl -X POST "http://localhost:8000/chat/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explícame el concepto principal",
    "provider": "gemini",
    "chat_history": [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte?"}
    ],
    "use_langchain": true
  }'
```

---

## 📦 Colecciones en ChromaDB

Los documentos se guardan en colecciones separadas por provider:

- **Llama/Ollama**: `documents_langchain_llama`
- **Gemini**: `documents_langchain_gemini`

Esto permite comparar la calidad de embeddings entre providers.

---

## ⚠️ Consideraciones

### Tamaño de Chunks
- **Chunks pequeños (500-800)**: Mejor precisión, más chunks, búsqueda más específica
- **Chunks medianos (1000-1500)**: Balance entre contexto y precisión ✅
- **Chunks grandes (2000+)**: Más contexto, menos chunks, menos precisión

### Overlap
- **Overlap bajo (100-150)**: Menos redundancia
- **Overlap medio (200-250)**: Balance recomendado ✅
- **Overlap alto (300+)**: Más contexto en límites, más redundancia

### Providers
- **Llama (Ollama)**: Gratuito, local, rápido, buena calidad
- **Gemini**: API de Google, excelente calidad, requiere API key

---

## 🐛 Errores Comunes

### Error 400: "Provider debe ser 'llama' o 'gemini'"
- Solución: Usa solo `"llama"` o `"gemini"` (minúsculas)

### Error 400: "Solo se aceptan archivos PDF"
- Solución: Asegúrate que el archivo tenga extensión `.pdf`

### Error 400: "extraction_mode debe ser 'elements', 'single' o 'paged'"
- Solución: Usa uno de los tres modos válidos

### Error 500: Error procesando el archivo
- Posibles causas:
  - PDF corrupto o protegido con contraseña
  - Dependencias de `unstructured` no instaladas
  - ChromaDB no disponible
  - API key de Gemini inválida (si usas provider="gemini")

---

## 🔗 Integración con Frontend

### Ejemplo completo en Vue.js/React:

```javascript
async function uploadPDF(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Valores por defecto
    const {
        provider = 'llama',
        chunkSize = 1000,
        chunkOverlap = 200,
        extractionMode = 'elements',
        author,
        category,
        tags,
        year
    } = options;
    
    formData.append('provider', provider);
    formData.append('chunk_size', chunkSize.toString());
    formData.append('chunk_overlap', chunkOverlap.toString());
    formData.append('extraction_mode', extractionMode);
    
    if (author) formData.append('author', author);
    if (category) formData.append('category', category);
    if (tags) formData.append('tags', tags);
    if (year) formData.append('year', year);
    
    try {
        const response = await fetch('http://localhost:8000/documents/upload-file-langchain', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ PDF procesado:', result);
        return result;
        
    } catch (error) {
        console.error('❌ Error:', error);
        throw error;
    }
}

// Uso:
await uploadPDF(pdfFile, {
    provider: 'gemini',
    extractionMode: 'elements',
    author: 'John Doe',
    category: 'Tutorial',
    tags: 'python,fastapi,langchain',
    year: '2024'
});
```

---

## 📈 Comparación: LangChain vs Custom

| Característica | `/upload-file-langchain` | `/upload-file-own` |
|----------------|--------------------------|---------------------|
| **Extracción** | UnstructuredPDFLoader | PyPDF2 |
| **Chunking** | RecursiveCharacterTextSplitter | Custom split |
| **Detección de estructura** | ✅ Sí (títulos, tablas, etc.) | ❌ No |
| **Configuración** | Más opciones | Básica |
| **Rendimiento** | Similar | Similar |
| **Uso recomendado** | PDFs complejos | PDFs simples |

---

## 📞 Soporte

Para más información sobre LangChain y el servicio:
- Ver código: `app/services/langchain_service.py`
- Ver endpoint: `app/api/documents_route.py`
- Documentación LangChain: https://python.langchain.com/

---

✨ **¡Listo para procesar PDFs con LangChain!**
