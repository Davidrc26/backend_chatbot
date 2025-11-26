from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import documents_route, chat_route
from app.telegram.bot import telegram_bot
import asyncio


# Lifespan context manager para manejar startup y shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: arrancar bot en background
    telegram_task = asyncio.create_task(telegram_bot.start())
    
    yield  # Aquí la aplicación está corriendo
    
    # Shutdown: detener el bot
    try:
        await telegram_bot.stop()
        # Dar tiempo a la tarea para limpiar correctamente
        await asyncio.wait_for(telegram_task, timeout=10.0)
    except asyncio.TimeoutError:
        # No bloquear el shutdown indefinidamente
        pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
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


@app.get("/")
async def root():
    return {"message": "Bienvenido al Backend Chatbot API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
