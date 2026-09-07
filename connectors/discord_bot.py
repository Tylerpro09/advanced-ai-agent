"""Optional Discord connector. Run: python connectors/discord_bot.py"""
from __future__ import annotations

import discord

from app.config import settings
from app.core.agent import Agent
from app.core.memory import MemoryStore
from app.core.rag import RAGStore

memory = MemoryStore(settings.database_path)
agent = Agent(memory, RAGStore(settings.database_path))

class Client(discord.Client):
    async def on_ready(self):
        print(f"Discord connected as {self.user}")

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.content:
            return
        uid = int(message.author.id)
        if settings.discord_users and uid not in settings.discord_users:
            return
        # In guilds, answer only when mentioned. DMs are answered directly.
        if message.guild and self.user not in message.mentions:
            return
        text = message.content.replace(f"<@{self.user.id}>", "").strip() if self.user else message.content
        result = await agent.chat(f"discord:{uid}", f"channel:{message.channel.id}", text)
        await message.reply(result["answer"][:1900], mention_author=False)


def main():
    if not settings.discord_bot_token:
        raise SystemExit("Set DISCORD_BOT_TOKEN in .env")
    intents = discord.Intents.default(); intents.message_content = True
    Client(intents=intents).run(settings.discord_bot_token)

if __name__ == "__main__":
    main()
