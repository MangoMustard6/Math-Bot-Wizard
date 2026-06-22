"""
MathBot — Production Discord Math Assistant
Entry point: initializes the bot, loads the MathAssistant Cog, and syncs
the application command tree for Slash Command support.
"""

import asyncio
import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler()],
)

# Silence noisy third-party loggers
logging.getLogger("discord.gateway").setLevel(logging.WARNING)
logging.getLogger("discord.http").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger("mathbot.main")


# ---------------------------------------------------------------------------
# Bot subclass
# ---------------------------------------------------------------------------

class MathBot(commands.Bot):
    """Custom Bot subclass for the Math Assistant."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # Required for prefix-command parsing

        prefix = os.getenv("DEFAULT_PREFIX", "!")

        super().__init__(
            command_prefix=prefix,
            intents=intents,
            help_command=None,           # Suppress default help; Cog owns /help_math
            description="A production-grade math assistant bot powered by SymPy.",
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """
        Called once before the bot logs in.
        Loads all Cogs and syncs the global application command tree.
        """
        logger.info("Running setup_hook — loading Cogs …")

        try:
            await self.load_extension("cogs.math_assistant")
            logger.info("Cog loaded: cogs.math_assistant")
        except Exception:
            logger.exception("Failed to load cog 'cogs.math_assistant'")
            raise

        logger.info("Syncing global application command tree …")
        try:
            synced = await self.tree.sync()
            logger.info("Slash commands synced: %d command(s) registered.", len(synced))
        except discord.HTTPException:
            logger.exception("Failed to sync application command tree.")

    async def on_ready(self) -> None:
        logger.info(
            "Bot online — logged in as %s (ID: %s)", self.user, self.user.id
        )
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="equations | /help_math",
            )
        )

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        """Global fallback error handler for prefix commands."""
        if isinstance(error, commands.CommandNotFound):
            return  # Silently ignore unknown prefix commands
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                f"⚠️ Missing argument: `{error.param.name}`. "
                f"Use `/help_math` for usage details."
            )
        elif isinstance(error, commands.BadArgument):
            await ctx.send(f"⚠️ Bad argument: {error}")
        else:
            logger.error("Unhandled command error in '%s': %s", ctx.command, error)
            await ctx.send("❌ An unexpected error occurred. Please try again.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error(
            "DISCORD_TOKEN is not set. "
            "Add it as a Replit Secret or in your .env file."
        )
        raise RuntimeError("DISCORD_TOKEN environment variable is missing.")

    bot = MathBot()
    async with bot:
        await bot.start(token)


if __name__ == "__main__":
    asyncio.run(main())
