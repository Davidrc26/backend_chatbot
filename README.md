# Backend Chatbot - FastAPI

Backend para sistema de chatbot construido con FastAPI.

## 🚀 Características

- FastAPI framework
- Estructura modular y escalable
- Configuración basada en variables de entorno
- CORS configurado
- API REST con versionado
- Documentación automática (Swagger/OpenAPI)

## 📋 Requisitos

- Python 3.8+
- pip

## 🔧 Instalación

1. Clonar el repositorio
2. Crear entorno virtual:
```bash
python -m venv venv
```

3. Activar entorno virtual:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

5. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus configuraciones
```

## 🚀 Ejecutar el proyecto

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 Documentación

Una vez ejecutado el servidor, accede a:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/api/v1/openapi.json

## 📁 Estructura del Proyecto

```
backend_chatbot/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       └── api.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   ├── schemas/
│   └── services/
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```
