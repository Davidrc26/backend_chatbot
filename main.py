from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api import documents_route, chat_route
from app.telegram.bot import telegram_bot
import asyncio

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Incluir routers
app.include_router(documents_route.router, prefix=settings.API_V1_STR)
app.include_router(chat_route.router, prefix=settings.API_V1_STR)

# Guardar la tarea global para poder esperarla si es necesario
_telegram_task: asyncio.Task | None = None

@app.on_event("startup")
async def startup_event():
    global _telegram_task
    # arrancar bot en background para no bloquear el arranque de FastAPI
    _telegram_task = asyncio.create_task(telegram_bot.start())

@app.on_event("shutdown")
async def shutdown_event():
    # indicar al bot que se detenga y esperar a que termine la tarea
    try:
        await telegram_bot.stop()
        if _telegram_task:
            # dar tiempo a la tarea para limpiar correctamente
            await asyncio.wait_for(_telegram_task, timeout=10.0)
    except asyncio.TimeoutError:
        # no bloquear el shutdown indefinidamente
        pass


@app.get("/")
async def root():
    return {"message": "Bienvenido al Backend Chatbot API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
