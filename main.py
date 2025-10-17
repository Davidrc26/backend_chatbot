from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import documents_route

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(documents_route.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {"message": "Bienvenido al Backend Chatbot API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
