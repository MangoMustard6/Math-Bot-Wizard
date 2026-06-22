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
        intents.message_content = True

        prefix = os.getenv("DEFAULT_PREFIX", "!")

        super().__init__(
            command_prefix=prefix,
            intents=intents,
            help_command=None,
            description="A production-grade math assistant bot powered by SymPy.",
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """Loads all Cogs. Guild sync happens in on_ready so the guild object is available."""
        logger.info("Running setup_hook — loading Cogs …")
        try:
            await self.load_extension("cogs.math_assistant")
            logger.info("Cog loaded: cogs.math_assistant")
        except Exception:
            logger.exception("Failed to load cog 'cogs.math_assistant'")
            raise

    async def on_ready(self) -> None:
        logger.info("Bot online — logged in as %s (ID: %s)", self.user, self.user.id)

        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="equations | /help_math",
            )
        )

        # ── Guild-specific sync (instant) ─────────────────────────────
        # If GUILD_ID is set, copy and sync commands to that guild immediately.
        # This makes ALL slash commands appear within seconds, not hours.
        guild_id_str = os.getenv("GUILD_ID")
        if guild_id_str:
            try:
                guild = discord.Object(id=int(guild_id_str))
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(
                    "Guild sync complete — %d command(s) available instantly in guild %s.",
                    len(synced), guild_id_str,
                )
            except Exception:
                logger.exception("Guild sync failed. Falling back to global sync.")
                await self._global_sync()
        else:
            # No GUILD_ID set — do a global sync (propagates within ~1 hour)
            await self._global_sync()

    async def _global_sync(self) -> None:
        try:
            synced = await self.tree.sync()
            logger.info(
                "Global sync complete — %d command(s) registered "
                "(may take up to 1 hour to appear in Discord).",
                len(synced),
            )
        except discord.HTTPException:
            logger.exception("Failed to sync application command tree.")

    async def on_command_error(
        self, ctx: commands.Context, error: commands.CommandError
    ) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(
                f"⚠️ Missing argument: `{error.param.name}`. "
                f"Use `/help_math` for usage details."
            )
        elif isinstance(error, commands.BadArgument):
            await ctx.send(f"⚠️ Bad argument: {error}")
        elif isinstance(error, commands.NotOwner):
            await ctx.send("🔒 That command is restricted to the bot owner.")
        else:
            logger.error("Unhandled command error in '%s': %s", ctx.command, error)
            await ctx.send("❌ An unexpected error occurred. Please try again.")


# ---------------------------------------------------------------------------
# Owner-only sync command (instant guild sync on demand)
# ---------------------------------------------------------------------------

bot_instance: MathBot | None = None


@commands.command(name="sync")
@commands.is_owner()
async def sync_commands(ctx: commands.Context) -> None:
    """
    Owner-only prefix command: !sync
    Instantly copies all global commands to the current guild and syncs them.
    Use this whenever new slash commands are added and you can't wait for
    Discord's global propagation delay (~1 hour).
    """
    await ctx.send("⏳ Syncing slash commands to this server …")
    try:
        guild = discord.Object(id=ctx.guild.id)
        bot_instance.tree.copy_global_to(guild=guild)
        synced = await bot_instance.tree.sync(guild=guild)
        await ctx.send(
            f"✅ Synced **{len(synced)} command(s)** to this server instantly!\n"
            f"All slash commands should now appear in the `/` menu."
        )
        logger.info(
            "Manual guild sync by owner — %d command(s) synced to guild %s.",
            len(synced), ctx.guild.id,
        )
    except Exception as exc:
        await ctx.send(f"❌ Sync failed: `{exc}`")
        logger.error("Manual sync error: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    global bot_instance

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error(
            "DISCORD_TOKEN is not set. "
            "Add it as a Replit Secret or in your .env file."
        )
        raise RuntimeError("DISCORD_TOKEN environment variable is missing.")

    bot_instance = MathBot()
    bot_instance.add_command(sync_commands)

    try:
        async with bot_instance:
            await bot_instance.start(token)
    except discord.PrivilegedIntentsRequired:
        logger.error(
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  PRIVILEGED INTENT NOT ENABLED\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  The 'Message Content Intent' must be enabled in the\n"
            "  Discord Developer Portal before the bot can start.\n\n"
            "  Steps:\n"
            "    1. Go to https://discord.com/developers/applications\n"
            "    2. Select your application → Bot\n"
            "    3. Scroll to 'Privileged Gateway Intents'\n"
            "    4. Enable: MESSAGE CONTENT INTENT\n"
            "    5. Save Changes, then restart this bot.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        raise SystemExit(1)
    except discord.LoginFailure:
        logger.error(
            "Invalid DISCORD_TOKEN — check the token in your Replit Secrets."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
