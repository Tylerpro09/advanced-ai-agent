"""Optional Telegram connector. Run: python connectors/telegram_bot.py"""
from __future__ import annotations

import asyncio
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from app.config import settings
from app.core.agent import Agent
from app.core.memory import MemoryStore
from app.core.rag import RAGStore

memory = MemoryStore(settings.database_path)
agent = Agent(memory, RAGStore(settings.database_path))

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.message or not update.message.text:
        return
    uid = int(update.effective_user.id)
    if settings.telegram_users and uid not in settings.telegram_users:
        return
    result = await agent.chat(f"telegram:{uid}", f"chat:{update.effective_chat.id}", update.message.text)
    await update.message.reply_text(result["answer"][:4000])


def main():
    if not settings.telegram_bot_token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()

if __name__ == "__main__":
    main()
