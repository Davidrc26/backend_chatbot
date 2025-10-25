# app/telegram/bot.py
import asyncio
from typing import Optional
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction

from app.core.config import settings
from .session import UserState, UserSession, user_sessions

# Timeouts
HTTP_TIMEOUT = 60.0  # timeout para llamadas al backend (httpx)
TELEGRAM_TIMEOUT = getattr(settings, "TELEGRAM_REQUEST_TIMEOUT", 30)  # para PTB

# BACKEND base (ya incluye settings.API_V1_STR)
BACKEND_BASE_URL = f"{getattr(settings, 'BACKEND_BASE_URL', 'http://localhost:8000')}{settings.API_V1_STR}"


class TelegramBot:
    def __init__(self):
        # Aumentamos timeouts en el client de telegram para evitar TimedOut cuando RAG+rerank toma tiempo
        self.application = (
            Application.builder()
            .token(settings.TELEGRAM_BOT_TOKEN)
            .read_timeout(TELEGRAM_TIMEOUT)
            .write_timeout(TELEGRAM_TIMEOUT)
            .connect_timeout(TELEGRAM_TIMEOUT)
            .pool_timeout(TELEGRAM_TIMEOUT)
            .build()
        )
        self._http_client: Optional[httpx.AsyncClient] = None
        self._started = False
        self._setup_handlers()

    def _setup_handlers(self):
        # Comandos
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("politica", self.cmd_politica))
        self.application.add_handler(CommandHandler("fuentes", self.cmd_fuentes))
        self.application.add_handler(CommandHandler("reset", self.cmd_reset))
        self.application.add_handler(CommandHandler("modo", self.cmd_modo))
        self.application.add_handler(CommandHandler("provider", self.cmd_provider))
        # Mensajes y callbacks
        self.application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_message))
        self.application.add_handler(CallbackQueryHandler(self.handle_buttons))
        # Error handler
        self.application.add_error_handler(self.handle_error)

    async def _ensure_http_client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    # ---------- lifecycle ----------
    async def start(self):
        if self._started:
            return
        await self.application.initialize()
        await self._ensure_http_client()
        await self.application.start()
        self._started = True
        print("🤖 Bot de Telegram: iniciado (integrado con Uvicorn/ FastAPI)")

        # polling en background (usamos get_updates loop para evitar cerrar el event loop)
        async def polling_loop():
            bot = self.application.bot
            offset = None
            while True:
                try:
                    updates = await bot.get_updates(offset=offset, timeout=10)
                    for update in updates:
                        offset = update.update_id + 1
                        await self.application.process_update(update)
                except Exception as e:
                    # no levantamos excepciones que detengan el servidor; log y retry
                    print("⚠️ Error en polling:", e)
                    await asyncio.sleep(2)

        asyncio.create_task(polling_loop())

    async def stop(self):
        try:
            await self.application.stop()
            await self.application.shutdown()
        except Exception as e:
            print("⚠️ Error deteniendo Telegram app:", e)
        finally:
            if self._http_client:
                await self._http_client.aclose()
            print("🤖 Bot detenido")

    # ---------- comandos ----------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        # crear session si no existe
        session = UserSession()
        user_sessions[user_id] = session
        session.state = UserState.PENDING_OPTIN

        await update.message.reply_text(
            "👋 Bienvenido. Para usar el bot debes aceptar la política de tratamiento de datos.\n\n"
            "Escribe *ACEPTO* para continuar o *SALIR* para terminar. Usa /politica para ver el resumen.",
            parse_mode="Markdown",
        )

    async def cmd_politica(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📄 Política de tratamiento de datos (resumen):\n"
            "- No almacenamos tu número ni tu username.\n"
            "- Guardamos solo un session_id anónimo y el historial de chat si activas el modo extendido.\n"
            "- Tus mensajes pueden usarse de forma anónima para mejorar el servicio.\n\n"
            "Si estás de acuerdo escribe *ACEPTO*.",
            parse_mode="Markdown",
        )

    async def cmd_fuentes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        session = user_sessions.get(user_id)
        if not session or not session.last_sources:
            await update.message.reply_text("No hay fuentes disponibles todavía.")
            return
        text = "🔎 Fuentes encontradas:\n" + "\n".join(f"- {s}" for s in session.last_sources)
        await update.message.reply_text(text)

    async def cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        session = user_sessions.get(user_id)
        if session:
            session.chat_history = []
            session.last_sources = []
        await update.message.reply_text("✅ Historial y fuentes eliminados para esta sesión.")

    async def cmd_modo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        session = user_sessions.get(user_id)
        if not session:
            session = UserSession()
            user_sessions[user_id] = session
        session.mode = "extendido" if session.mode == "breve" else "breve"
        await update.message.reply_text(f"🔁 Modo cambiado a: *{session.mode}*", parse_mode="Markdown")

    async def cmd_provider(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        session = user_sessions.get(user_id) or UserSession()
        user_sessions[user_id] = session
        session.provider = "gemini" if session.provider == "llama" else "llama"
        await update.message.reply_text(f"🔁 Provider cambiado a: *{session.provider}*", parse_mode="Markdown")

    # ---------- callbacks / botones ----------
    async def handle_buttons(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        session = user_sessions.get(user_id) or UserSession()
        user_sessions[user_id] = session

        data = query.data

        # provider selection
        if data in ("llama", "gemini"):
            session.provider = data
            await query.edit_message_text(f"✅ Modelo: *{session.provider.upper()}*", parse_mode="Markdown")
            # ask mode
            keyboard = [
                [
                    InlineKeyboardButton("Breve (RAG)", callback_data="modo_breve"),
                    InlineKeyboardButton("Extendido (conversacional)", callback_data="modo_extendido"),
                ]
            ]
            await query.message.reply_text("Selecciona el modo de respuesta:", reply_markup=InlineKeyboardMarkup(keyboard))
            return

        # mode selection
        if data.startswith("modo_"):
            if data == "modo_breve":
                session.mode = "breve"
            else:
                session.mode = "extendido"
                # ensure chat_history exists
                if not hasattr(session, "chat_history"):
                    session.chat_history = []
            session.state = UserState.ACTIVE
            await query.edit_message_text(f"✅ Modo seleccionado: *{session.mode}*", parse_mode="Markdown")
            await query.message.reply_text(
                f"Configuración lista — Modelo: *{session.provider.upper()}* | Modo: *{session.mode}*\n"
                "Puedes comenzar a preguntar.",
                parse_mode="Markdown",
            )
            return

    # ---------- mensajes ----------
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        text = (update.message.text or "").strip()
        session = user_sessions.get(user_id)

        if not session:
            await update.message.reply_text("Escribe /start para comenzar.")
            return

        # exit
        if text.upper() == "SALIR":
            session.state = UserState.EXITED
            await update.message.reply_text("👋 Has salido. Usa /start para volver.")
            return

        # opt-in
        if session.state in (UserState.NEW, UserState.PENDING_OPTIN):
            if text.upper() == "ACEPTO":
                # ask provider via buttons
                keyboard = [[
                    InlineKeyboardButton("LLaMA", callback_data="llama"),
                    InlineKeyboardButton("Gemini", callback_data="gemini"),
                ]]
                await update.message.reply_text("Selecciona el modelo que deseas usar:", reply_markup=InlineKeyboardMarkup(keyboard))
                return
            else:
                await update.message.reply_text("Debes escribir *ACEPTO* o *SALIR* para continuar.", parse_mode="Markdown")
                return

        if session.state != UserState.ACTIVE:
            await update.message.reply_text("Por favor selecciona primero un modelo y modo.")
            return

        # indicate typing
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        except Exception:
            # no crítico, seguimos
            pass

        provider = session.provider

        # Build payloads according to your backend contract
        if session.mode == "breve":
            # RAG (short) -> /chat/rag
            endpoint = f"{BACKEND_BASE_URL}/chat/rag?provider={provider}"
            payload = {"message": text, "n_results": 3, "use_rerank": False}

        else:
            # Extendido -> conversation endpoint (historial + RAG + rerank)
            endpoint = f"{BACKEND_BASE_URL}/chat/conversation?provider={provider}"
            # ensure chat_history exists
            session.chat_history = getattr(session, "chat_history", [])
            payload = {
                "message": text,
                "chat_history": session.chat_history,
                "use_rag": True,
                "n_results": 5,
                "use_rerank": True,
            }

        # Call backend
        try:
            await self._ensure_http_client()
            resp = await self._http_client.post(endpoint, json=payload)
        except Exception as e:
            await update.message.reply_text(f"❗ Error conectando al backend: {e}")
            return

        if resp.status_code != 200:
            await update.message.reply_text(f"❗ Backend error: {resp.status_code} - {resp.text}")
            return

        try:
            data = resp.json()
        except Exception:
            await update.message.reply_text("❗ El backend devolvió una respuesta no válida.")
            return

        # Extract answer and sources according to your ChatResponse schema
        answer = data.get("response") or data.get("answer") or "No se recibió respuesta."
        sources = data.get("sources", []) or data.get("found_documents", []) or []

        # persist history (if extendido) and sources
        if session.mode == "extendido":
            session.chat_history.append({"role": "user", "content": text})
            session.chat_history.append({"role": "assistant", "content": answer})

        if sources:
            session.last_sources = sources

        # Reply
        await update.message.reply_text(answer)
        if sources:
            await update.message.reply_text("🔎 Se encontraron fuentes. Usa /fuentes para verlas.")

    # ---------- error handler ----------
    async def handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        # Log the error (the application also logs it). Try to notify the user.
        try:
            print("Error in handler:", context.error)
            if isinstance(update, Update) and update.effective_message:
                await update.effective_message.reply_text("⚠️ Ocurrió un error interno. Intenta más tarde.")
        except Exception:
            pass


# Export instance
telegram_bot = TelegramBot()
