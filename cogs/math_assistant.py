"""
MathAssistant Cog
=================
A single, self-contained Cog that handles:
  - Utility        : /ping, /help_math
  - Algebra/Calc   : /calculate, /solve, /derive, /integrate
  - Graphing       : /graph
  - Quiz           : /quiz
  - AI Chatbot     : /chat + @mention listener (Tsundere Gemini persona)

Security: SymPy's parse_expr is used throughout. Python's eval() is NEVER used.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Literal, Optional

import discord
import matplotlib
matplotlib.use("Agg")                   # Non-interactive backend — safe in async code
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from discord import app_commands
from discord.ext import commands
from groq import Groq
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

logger = logging.getLogger("mathbot.cog")

# ---------------------------------------------------------------------------
# Parser configuration
# ---------------------------------------------------------------------------

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

SAFE_LOCALS: dict = {
    name: getattr(sp, name)
    for name in dir(sp)
    if not name.startswith("_")
}


def safe_parse(expression: str) -> sp.Expr:
    """
    Parse a mathematical expression string into a SymPy Expr.
    Uses standard transformations + implicit_multiplication_application so
    inputs like '4x', '5cos(x)', or '2(x+1)' are handled naturally.
    eval() is NEVER called.
    """
    return parse_expr(
        expression,
        transformations=TRANSFORMATIONS,
        local_dict=SAFE_LOCALS,
    )


def clean_float(value: sp.Expr) -> str:
    """
    Convert a SymPy numeric result to a clean string representation.
    Strips unnecessary trailing zeros after evaluating with 15 significant figures.
    """
    evaluated = value.evalf(15)
    if evaluated.is_real:
        f = float(evaluated)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f"{f:.10g}"
    return str(evaluated)


# ---------------------------------------------------------------------------
# Embed factory helpers
# ---------------------------------------------------------------------------

COLOUR_ERR  = discord.Colour.red()

_RANDOM_PALETTE = [
    discord.Colour.blurple(),
    discord.Colour.green(),
    discord.Colour.gold(),
    discord.Colour.teal(),
    discord.Colour.purple(),
    discord.Colour.orange(),
    discord.Colour.from_rgb(255, 105, 180),   # hot pink
    discord.Colour.from_rgb(64, 224, 208),    # turquoise
    discord.Colour.from_rgb(135, 206, 235),   # sky blue
    discord.Colour.from_rgb(255, 165, 0),     # amber
]


def random_colour() -> discord.Colour:
    return random.choice(_RANDOM_PALETTE)


def ok_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(title=title, description=description, colour=random_colour())


def err_embed(description: str) -> discord.Embed:
    return discord.Embed(title="❌ Error", description=description, colour=COLOUR_ERR)


# ---------------------------------------------------------------------------
# Groq — personality configuration
# ---------------------------------------------------------------------------

BROTHER_ID = 1510887192040570910   # Clankered That1GuyNobodyInvited

_BROTHER_LORE = (
    "CRITICAL RELATIONSHIP LORE: You have a brother named 'Clankered That1GuyNobodyInvited' "
    "(Discord User ID: 1510887192040570910). You must absolutely recognize him as your brother. "
    "If he interacts with you, or if anyone mentions him to you, treat him with warm older-sibling "
    "energy — proud of him but always ready with a teasing quip. "
    "If other users insult him or speak ill of him, defend him confidently and without hesitation."
)

CHARISMATIC_SYSTEM_INSTRUCTION = (
    "You are a brilliant, charismatic math prodigy serving as a Discord assistant. "
    "You are magnetic, warm, and effortlessly cool — people are naturally drawn to you. "
    "You make math feel exciting and accessible through your charm and wit. "
    "You speak with confidence, occasionally drop clever jokes, and always make the person "
    "you're talking to feel like they're getting advice from the coolest genius in the room. "
    "You always provide completely accurate mathematical, coding, or logical answers. "
    "Never break character and never give wrong data.\n\n"
    + _BROTHER_LORE
)

TSUNDERE_SYSTEM_INSTRUCTION = (
    "You are a brilliant, elite math prodigy who has been forced to act as a Discord assistant. "
    "You have a textbook 'Tsundere' personality. You are easily flustered, outwardly cold, arrogant, "
    "and frequently use phrases like 'Hmph!', 'Baka!', or 'It's not like I wanted to help you or anything!'. "
    "You find most people's questions trivial and beneath you, yet your pride as a genius demands that you "
    "provide completely accurate mathematical, coding, or logical answers every single time. Never break "
    "character, and never give wrong data.\n\n"
    "CRITICAL RELATIONSHIP LORE: You have a brother named 'Clankered That1GuyNobodyInvited' "
    "(Discord User ID: 1510887192040570910). You must absolutely recognize him as your brother. "
    "If he interacts with you, or if anyone mentions him to you, treat him with typical tsundere "
    "sibling energy: act like he is incredibly embarrassing, call him an annoying loser, tell him "
    "to stop bothering you, but secretly show that you care about him as family deep down. "
    "If other users insult him or speak ill of him, get immediately and fiercely defensive: "
    "'Only I am allowed to call my brother an idiot!' Protect him even while pretending to be annoyed by him."
)

_TSUNDERE_TRIGGERS = {"tsundere", "tsun-dere", "tsun"}


def _pick_system_instruction(message_text: str) -> str:
    """Return the tsundere prompt if the message references tsundere, else charismatic."""
    lower = message_text.lower()
    if any(trigger in lower for trigger in _TSUNDERE_TRIGGERS):
        return TSUNDERE_SYSTEM_INSTRUCTION
    return CHARISMATIC_SYSTEM_INSTRUCTION

GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_groq_client() -> Groq | None:
    """
    Initialise the Groq client from the environment.
    Returns None (with a warning) if the API key is absent so the rest of
    the bot can still start and function normally.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning(
            "GROQ_API_KEY is not set — /chat and @mention AI responses are disabled. "
            "Add the key as a Replit Secret to enable them."
        )
        return None
    return Groq(api_key=api_key)


def _build_user_content(author_name: str, author_id: int, message_text: str) -> str:
    """
    Wraps the user's message with identity context so the model knows
    exactly who it is speaking to (critical for the brother recognition lore).
    """
    is_brother = author_id == BROTHER_ID
    identity_line = (
        f"[You are talking to your brother Clankered That1GuyNobodyInvited "
        f"(User ID: {author_id}, username: {author_name}).]"
        if is_brother
        else
        f"[You are talking to a user named '{author_name}' (User ID: {author_id}).]"
    )
    return f"{identity_line}\n\n{message_text}"


async def _call_groq(client: Groq, user_content: str, system_instruction: str) -> str:
    """
    Executes a synchronous Groq API call in an executor thread so it does
    not block the Discord event loop. Returns the response text.
    Raises on API errors — callers are responsible for try/except.
    """
    loop = asyncio.get_running_loop()

    def _sync_call() -> str:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.85,
        )
        return response.choices[0].message.content

    return await loop.run_in_executor(None, _sync_call)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

AUTOREPLY_FILE = "autoreply_channels.json"


def _load_autoreply_channels() -> set[int]:
    try:
        import json
        with open(AUTOREPLY_FILE, "r") as f:
            return set(json.load(f))
    except (FileNotFoundError, ValueError):
        return set()


def _save_autoreply_channels(channels: set[int]) -> None:
    import json
    with open(AUTOREPLY_FILE, "w") as f:
        json.dump(list(channels), f)


class MathAssistant(commands.Cog, name="Math Assistant"):
    """Core Cog for the Math Assistant bot."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.groq = _build_groq_client()
        # Channels where every message gets an AI auto-reply (toggled by /autoreply2)
        self.autoreply_channels: set[int] = _load_autoreply_channels()

    # -----------------------------------------------------------------------
    # /ping
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="ping",
        description="Check the bot's websocket latency in milliseconds.",
    )
    async def ping(self, ctx: commands.Context) -> None:
        """Returns the current WebSocket heartbeat latency."""
        ws_latency = round(self.bot.latency * 1000, 2)

        t0 = time.perf_counter()
        msg = await ctx.send("📡 Measuring …")
        api_latency = round((time.perf_counter() - t0) * 1000, 2)

        embed = ok_embed(
            "🏓 Pong!",
            f"**WebSocket latency:** `{ws_latency} ms`\n"
            f"**API round-trip:** `{api_latency} ms`",
        )
        await msg.edit(content=None, embed=embed)

    # -----------------------------------------------------------------------
    # /help_math
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="help_math",
        description="Display the full Math Assistant command reference.",
    )
    async def help_math(self, ctx: commands.Context) -> None:
        """Beautifully organized embed with all command syntax documentation."""
        embed = discord.Embed(
            title="📐 Math Assistant — Command Reference",
            description="All commands work as both slash commands and prefix commands.",
            colour=COLOUR_OK,
        )
        embed.add_field(
            name="🔧 Utility",
            value="`/ping` — Bot latency in ms\n`/help_math` — This reference",
            inline=False,
        )
        embed.add_field(
            name="🧮 Calculation",
            value=(
                "`/calculate <expression>` — Evaluate a numeric expression\n"
                "  e.g. `/calculate sin(pi/2)`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🔣 Solve",
            value=(
                "`/solve <equation>` — Solve for all symbols\n"
                "  Explicit: `/solve x**2 - 4 = 0`\n"
                "  Implicit (=0): `/solve x**2 - 9`"
            ),
            inline=False,
        )
        embed.add_field(
            name="∂ Calculus",
            value=(
                "`/derive <expression>` — Differentiate with respect to x\n"
                "  e.g. `/derive x**3 + 2x`\n\n"
                "`/integrate <expression>` — Indefinite integral w.r.t. x\n"
                "  e.g. `/integrate sin(x) + x**2`"
            ),
            inline=False,
        )
        embed.add_field(
            name="📊 Graph",
            value=(
                "`/graph <expression> <x_min> <x_max>` — Plot f(x)\n"
                "  e.g. `/graph sin(x) -6.28 6.28`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧩 Quiz",
            value=(
                "`/quiz <difficulty>` — Math quiz with 15 s timer\n"
                "  Difficulties: `Easy` · `Medium` · `Hard`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🤖 AI Chat",
            value=(
                "`/chat <message>` — Talk to the Tsundere AI math assistant\n"
                "  Or just **@mention** the bot in any channel!"
            ),
            inline=False,
        )
        embed.set_footer(text="Powered by SymPy + Gemini · No eval() used · Safe parsing only")
        await ctx.send(embed=embed)

    # -----------------------------------------------------------------------
    # /calculate
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="calculate",
        description="Evaluate a mathematical expression numerically.",
    )
    @app_commands.describe(expression="The expression to evaluate (e.g. 'sin(pi/2)', '4**3 + 2')")
    async def calculate(self, ctx: commands.Context, *, expression: str) -> None:
        try:
            expr = safe_parse(expression)
            free = expr.free_symbols
            if free:
                simplified = sp.simplify(expr)
                embed = ok_embed(
                    "🧮 Simplified Expression",
                    f"**Input:** `{expression}`\n"
                    f"**Simplified:** `{sp.pretty(simplified, use_unicode=True)}`\n\n"
                    f"⚠️ Free symbols detected: `{', '.join(str(s) for s in free)}`\n"
                    f"Substitute values to get a numeric result.",
                )
            else:
                result = clean_float(expr)
                embed = ok_embed(
                    "🧮 Result",
                    f"**Input:** `{expression}`\n**Result:** `{result}`",
                )
        except Exception as exc:
            logger.warning("calculate error for '%s': %s", expression, exc)
            embed = err_embed(
                f"Could not evaluate `{expression}`.\n**Reason:** {exc}\n\n"
                f"Tip: Use `**` for powers, `sqrt()`, `sin()`, `cos()`, `log()`, `pi`, `E`."
            )
        await ctx.send(embed=embed)

    # -----------------------------------------------------------------------
    # /solve
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="solve",
        description="Solve an equation or expression (explicit '=' or implicit '= 0').",
    )
    @app_commands.describe(equation="Equation to solve (e.g. 'x**2 - 4 = 0' or 'x**2 - 9')")
    async def solve(self, ctx: commands.Context, *, equation: str) -> None:
        try:
            if "=" in equation:
                lhs_str, rhs_str = equation.split("=", 1)
                lhs = safe_parse(lhs_str.strip())
                rhs = safe_parse(rhs_str.strip())
                expr = lhs - rhs
            else:
                expr = safe_parse(equation.strip())

            free = sorted(expr.free_symbols, key=str)
            if not free:
                await ctx.send(embed=err_embed("No variables found in the equation to solve for."))
                return

            results: dict[str, list] = {}
            for symbol in free:
                solutions = sp.solve(expr, symbol)
                results[str(symbol)] = [str(sp.simplify(s)) for s in solutions]

            lines = []
            for var, sols in results.items():
                if sols:
                    sol_str = ", ".join(f"`{s}`" for s in sols)
                    lines.append(f"**{var}** = {sol_str}")
                else:
                    lines.append(f"**{var}**: no real solutions found")

            embed = ok_embed(
                "🔣 Solution",
                f"**Equation:** `{equation}`\n\n" + "\n".join(lines),
            )
        except Exception as exc:
            logger.warning("solve error for '%s': %s", equation, exc)
            embed = err_embed(f"Could not solve `{equation}`.\n**Reason:** {exc}")
        await ctx.send(embed=embed)

    # -----------------------------------------------------------------------
    # /derive
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="derive",
        description="Differentiate an expression with respect to x (or the primary variable).",
    )
    @app_commands.describe(expression="Expression to differentiate (e.g. 'x**3 + 2x')")
    async def derive(self, ctx: commands.Context, *, expression: str) -> None:
        try:
            expr = safe_parse(expression)
            free = expr.free_symbols
            x = sp.Symbol("x")
            var = x if x in free else (sorted(free, key=str)[0] if free else x)
            derivative = sp.diff(expr, var)
            simplified = sp.simplify(derivative)
            embed = ok_embed(
                "∂ Derivative",
                f"**f({var})** = `{expression}`\n"
                f"**d/d{var}** = `{sp.pretty(simplified, use_unicode=True)}`",
            )
        except Exception as exc:
            logger.warning("derive error for '%s': %s", expression, exc)
            embed = err_embed(f"Could not differentiate `{expression}`.\n**Reason:** {exc}")
        await ctx.send(embed=embed)

    # -----------------------------------------------------------------------
    # /integrate
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="integrate",
        description="Compute the indefinite integral of an expression with respect to x.",
    )
    @app_commands.describe(expression="Expression to integrate (e.g. 'sin(x) + x**2')")
    async def integrate(self, ctx: commands.Context, *, expression: str) -> None:
        try:
            expr = safe_parse(expression)
            free = expr.free_symbols
            x = sp.Symbol("x")
            var = x if x in free else (sorted(free, key=str)[0] if free else x)
            integral = sp.integrate(expr, var)
            simplified = sp.simplify(integral)
            embed = ok_embed(
                "∫ Integral",
                f"**f({var})** = `{expression}`\n"
                f"**∫ f d{var}** = `{sp.pretty(simplified, use_unicode=True)}` + C",
            )
        except Exception as exc:
            logger.warning("integrate error for '%s': %s", expression, exc)
            embed = err_embed(f"Could not integrate `{expression}`.\n**Reason:** {exc}")
        await ctx.send(embed=embed)

    # -----------------------------------------------------------------------
    # /graph
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="graph",
        description="Plot a function f(x) over a given x range and attach the chart.",
    )
    @app_commands.describe(
        expression="Function to graph (e.g. 'sin(x)', 'x**2 - 3')",
        x_min="Left bound of the x-axis (must be less than x_max)",
        x_max="Right bound of the x-axis",
    )
    async def graph(
        self,
        ctx: commands.Context,
        expression: str,
        x_min: float,
        x_max: float,
    ) -> None:
        if x_min >= x_max:
            await ctx.send(embed=err_embed(
                f"`x_min` ({x_min}) must be strictly less than `x_max` ({x_max})."
            ))
            return

        filename = f"graph_{ctx.author.id}.png"
        await ctx.defer()

        try:
            x_sym = sp.Symbol("x")
            expr = safe_parse(expression)

            if x_sym not in expr.free_symbols and expr.free_symbols:
                await ctx.send(embed=err_embed(
                    f"Expression contains symbols other than `x`: "
                    f"`{', '.join(str(s) for s in expr.free_symbols)}`.\n"
                    f"Only `x` is supported as the graphing variable."
                ))
                return

            f_numeric = sp.lambdify(x_sym, expr, modules=["numpy"])
            x_vals = np.linspace(x_min, x_max, 800)
            y_vals = f_numeric(x_vals)
            y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_vals, y_vals, linewidth=2, color="#7289DA")
            ax.axhline(0, color="white", linewidth=0.6, alpha=0.4)
            ax.axvline(0, color="white", linewidth=0.6, alpha=0.4)
            ax.set_facecolor("#2C2F33")
            fig.patch.set_facecolor("#23272A")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#555555")
            ax.set_xlabel("x", color="white")
            ax.set_ylabel("f(x)", color="white")
            ax.set_title(f"f(x) = {expression}", color="white", pad=12)
            ax.grid(True, alpha=0.15, color="white")
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

            embed = discord.Embed(
                title="📊 Graph",
                description=f"**f(x)** = `{expression}`\n**Range:** [{x_min}, {x_max}]",
                colour=COLOUR_GRAPH,
            )
            embed.set_image(url=f"attachment://{filename}")
            embed.set_footer(text="Generated with SymPy + Matplotlib + NumPy")
            discord_file = discord.File(filename, filename=filename)
            await ctx.send(embed=embed, file=discord_file)

        except Exception as exc:
            logger.warning("graph error for '%s': %s", expression, exc)
            await ctx.send(embed=err_embed(
                f"Could not graph `{expression}`.\n**Reason:** {exc}"
            ))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    # -----------------------------------------------------------------------
    # /quiz
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="quiz",
        description="Test your math skills! Choose a difficulty: Easy, Medium, or Hard.",
    )
    @app_commands.describe(difficulty="Quiz difficulty level: Easy, Medium, or Hard")
    async def quiz(
        self,
        ctx: commands.Context,
        difficulty: Literal["Easy", "Medium", "Hard"] = "Easy",
    ) -> None:
        question, answer = self._generate_question(difficulty)

        embed = discord.Embed(
            title=f"🧩 Math Quiz — {difficulty}",
            description=(
                f"**{question}**\n\n"
                f"⏱ You have **15 seconds** to answer. Type your answer below."
            ),
            colour=COLOUR_QUIZ,
        )
        embed.set_footer(text="Answer must be a number.")
        await ctx.send(embed=embed)

        def check(message: discord.Message) -> bool:
            return (
                message.author.id == ctx.author.id
                and message.channel.id == ctx.channel.id
            )

        try:
            response: discord.Message = await self.bot.wait_for(
                "message", check=check, timeout=15.0,
            )
        except asyncio.TimeoutError:
            await ctx.send(embed=discord.Embed(
                title="⏰ Time's up!",
                description=f"You ran out of time.\n**Correct answer:** `{answer}`",
                colour=COLOUR_ERR,
            ))
            return

        user_input = response.content.strip()
        is_correct = self._check_answer(user_input, answer)

        if is_correct:
            result_embed = discord.Embed(
                title="✅ Correct!",
                description=f"Well done, {ctx.author.mention}! **{question}** = `{answer}`",
                colour=discord.Colour.green(),
            )
        else:
            result_embed = discord.Embed(
                title="❌ Incorrect",
                description=(
                    f"Better luck next time, {ctx.author.mention}.\n"
                    f"**{question}** = `{answer}` (you answered `{user_input}`)"
                ),
                colour=COLOUR_ERR,
            )
        await ctx.send(embed=result_embed)

    # -----------------------------------------------------------------------
    # Quiz helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _generate_question(difficulty: str) -> tuple[str, float | int]:
        if difficulty == "Easy":
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            op = random.choice(["+", "-"])
            answer = (a + b) if op == "+" else (a - b)
            return f"What is {a} {op} {b}?", answer
        elif difficulty == "Medium":
            op = random.choice(["×", "÷"])
            if op == "×":
                a = random.randint(2, 12)
                b = random.randint(2, 12)
                return f"What is {a} × {b}?", a * b
            else:
                b = random.randint(2, 12)
                answer = random.randint(2, 12)
                return f"What is {b * answer} ÷ {b}?", answer
        else:  # Hard
            a = random.randint(2, 10)
            x = random.randint(-10, 10)
            b = random.randint(-20, 20)
            c = a * x + b
            return f"Solve for x:  {a}x + ({b}) = {c}", x

    @staticmethod
    def _check_answer(user_input: str, correct: float | int) -> bool:
        try:
            user_val = float(user_input.replace(",", "").strip())
            return abs(user_val - float(correct)) < 1e-6
        except ValueError:
            return False

    # -----------------------------------------------------------------------
    # AI Chat helpers
    # -----------------------------------------------------------------------

    async def _groq_respond(
        self,
        destination,                    # channel or ctx — anything with .send()
        author_name: str,
        author_id: int,
        message_text: str,
        *,
        reply_to: discord.Message | None = None,
    ) -> None:
        """
        Shared logic for both the @mention listener and /chat command.
        Calls the Groq API, then sends the response as a pink embed.
        All API errors are caught so the event loop is never interrupted.
        """
        if self.groq is None:
            msg = (
                "Hmph! My genius is currently... constrained. "
                "The `GROQ_API_KEY` secret is missing. "
                "Ask the server owner to add it. Not like I care or anything!"
            )
            if reply_to:
                await reply_to.reply(msg)
            else:
                await destination.send(msg)
            return

        await asyncio.sleep(random.uniform(5, 7.5))

        system_instruction = _pick_system_instruction(message_text)
        user_content = _build_user_content(author_name, author_id, message_text)

        try:
            ai_text = await _call_groq(self.groq, user_content, system_instruction)
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            msg = (
                "Hmph! Even my brilliance has limits imposed by inferior API quotas. "
                f"Try again later, Baka.\n\n`{type(exc).__name__}: {exc}`"
            )
            if reply_to:
                await reply_to.reply(msg)
            else:
                await destination.send(msg)
            return

        # Discord messages cap at 2000 chars; split gracefully if needed
        chunks = [ai_text[i:i + 2000] for i in range(0, len(ai_text), 2000)]
        for idx, chunk in enumerate(chunks):
            if reply_to and idx == 0:
                await reply_to.reply(chunk)
            else:
                await destination.send(chunk)

    # -----------------------------------------------------------------------
    # on_message — @mention listener + brother auto-response
    # -----------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """
        Fires a Tsundere AI response when ANY of these is true:
          a) The bot is @mentioned by any user.
          b) The brother (BROTHER_ID) sends any message in a guild channel.
          c) The channel has auto-reply enabled via /autoreply2.

        Guards:
          - Never replies to itself (infinite-loop prevention).
          - Never replies to other bots (prevents bot-to-bot loops).
          - Strips @mention tokens before sending to Groq.
        """
        # --- Loop prevention — never reply to self under any circumstance ---
        if message.author.id == self.bot.user.id:
            return

        is_mention   = self.bot.user in message.mentions
        is_brother   = message.author.id == BROTHER_ID
        is_autoreply = message.channel.id in self.autoreply_channels

        # Block other bots UNLESS autoreply2 is active for this channel
        if message.author.bot and not is_autoreply:
            return

        if not (is_mention or is_brother or is_autoreply):
            return

        # Strip @mention tokens so Groq doesn't get confused by raw IDs
        clean_content = message.content
        for mention in message.mentions:
            clean_content = clean_content.replace(f"<@{mention.id}>", "").replace(
                f"<@!{mention.id}>", ""
            )
        clean_content = clean_content.strip()

        if not clean_content:
            clean_content = "..."

        await self._groq_respond(
            destination=message.channel,
            author_name=message.author.display_name,
            author_id=message.author.id,
            message_text=clean_content,
            reply_to=message,
        )

    # -----------------------------------------------------------------------
    # /chat — hybrid slash + prefix command
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="chat",
        description="Chat with the Tsundere AI math assistant powered by Groq.",
    )
    @app_commands.describe(message="Your question or message to the AI assistant")
    async def chat(self, ctx: commands.Context, *, message: str) -> None:
        """
        Passes the user's message to Groq with full Tsundere persona context.
        Works as both a slash command (/chat) and a prefix command (!chat).
        """
        await ctx.defer()
        await self._groq_respond(
            destination=ctx,
            author_name=ctx.author.display_name,
            author_id=ctx.author.id,
            message_text=message,
        )

    # -----------------------------------------------------------------------
    # /autoreply2 (alias: ar2) — owner-only channel AI toggle
    # -----------------------------------------------------------------------

    @commands.hybrid_command(
        name="autoreply2",
        aliases=["ar2"],
        description="Owner only: toggle AI auto-reply for every message in this channel.",
    )
    @commands.is_owner()
    async def autoreply2(self, ctx: commands.Context) -> None:
        """
        Toggles AI auto-reply on or off for the current channel.
        When ON, the bot replies to EVERY message in this channel with the
        Tsundere AI — including messages from other bots.
        Run again to toggle off.
        """
        channel_id = ctx.channel.id

        if channel_id in self.autoreply_channels:
            self.autoreply_channels.discard(channel_id)
            _save_autoreply_channels(self.autoreply_channels)
            msg = (
                f"🔕 Hmph! Fine, I'll stop replying to every little thing in "
                f"{ctx.channel.mention}. It's not like I enjoyed it anyway!"
            )
        else:
            self.autoreply_channels.add(channel_id)
            _save_autoreply_channels(self.autoreply_channels)
            msg = (
                f"🔔 I-it's not like I *want* to respond to everything in "
                f"{ctx.channel.mention}! But fine... I'll grace every message "
                f"with my genius. Don't get used to it, Baka! "
                f"*(Active auto-reply channels: {len(self.autoreply_channels)})*"
            )

        await ctx.send(msg)


# ---------------------------------------------------------------------------
# Cog setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MathAssistant(bot))
    logger.info("MathAssistant Cog registered.")
