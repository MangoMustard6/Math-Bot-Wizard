"""
MathAssistant Cog
=================
A single, self-contained Cog that handles:
  - Utility        : /ping, /help_math
  - Algebra/Calc   : /calculate, /solve, /derive, /integrate
  - Graphing       : /graph
  - Quiz           : /quiz

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
    # If the result is a real float with negligible imaginary part, simplify
    if evaluated.is_real:
        f = float(evaluated)
        # Display integers without decimal noise
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f"{f:.10g}"
    return str(evaluated)


# ---------------------------------------------------------------------------
# Embed factory helpers
# ---------------------------------------------------------------------------

COLOUR_OK    = discord.Colour.blurple()
COLOUR_ERR   = discord.Colour.red()
COLOUR_GRAPH = discord.Colour.green()
COLOUR_QUIZ  = discord.Colour.gold()


def ok_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(title=title, description=description, colour=COLOUR_OK)


def err_embed(description: str) -> discord.Embed:
    return discord.Embed(title="❌ Error", description=description, colour=COLOUR_ERR)


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class MathAssistant(commands.Cog, name="Math Assistant"):
    """Core Cog for the Math Assistant bot."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

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

        # Measure round-trip API latency
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
        prefix = ctx.prefix or "!"
        p = prefix  # short alias

        embed = discord.Embed(
            title="📐 Math Assistant — Command Reference",
            description="All commands work as both slash commands and prefix commands.",
            colour=COLOUR_OK,
        )
        embed.add_field(
            name="🔧 Utility",
            value=(
                f"`/ping` — Bot latency in ms\n"
                f"`/help_math` — This reference"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧮 Calculation",
            value=(
                f"`/calculate <expression>` — Evaluate a numeric expression\n"
                f"  e.g. `/calculate 4x + 3` where x=2 → use explicit values\n"
                f"  e.g. `/calculate sin(pi/2)`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🔣 Solve",
            value=(
                f"`/solve <equation>` — Solve for all symbols\n"
                f"  Explicit: `/solve x**2 - 4 = 0`\n"
                f"  Implicit (=0): `/solve x**2 - 9`"
            ),
            inline=False,
        )
        embed.add_field(
            name="∂ Calculus",
            value=(
                f"`/derive <expression>` — Differentiate with respect to x\n"
                f"  e.g. `/derive x**3 + 2x`\n\n"
                f"`/integrate <expression>` — Indefinite integral w.r.t. x\n"
                f"  e.g. `/integrate sin(x) + x**2`"
            ),
            inline=False,
        )
        embed.add_field(
            name="📊 Graph",
            value=(
                f"`/graph <expression> <x_min> <x_max>` — Plot f(x)\n"
                f"  e.g. `/graph sin(x) -6.28 6.28`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧩 Quiz",
            value=(
                f"`/quiz <difficulty>` — Math quiz with 15 s timer\n"
                f"  Difficulties: `Easy` · `Medium` · `Hard`"
            ),
            inline=False,
        )
        embed.set_footer(text="Powered by SymPy · No eval() used · Safe parsing only")
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
        """
        Parses and evaluates a numeric expression using SymPy.
        Supports natural notation like 4x, 5cos(x), implicit multiplication.
        """
        try:
            expr = safe_parse(expression)
            free = expr.free_symbols
            if free:
                # If symbols remain, attempt numerical substitution is impossible;
                # return the simplified symbolic form instead.
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
                    f"**Input:** `{expression}`\n"
                    f"**Result:** `{result}`",
                )
        except Exception as exc:
            logger.warning("calculate error for '%s': %s", expression, exc)
            embed = err_embed(
                f"Could not evaluate `{expression}`.\n"
                f"**Reason:** {exc}\n\n"
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
        """
        Solves an equation for all free symbols.
        Accepts either an explicit 'lhs = rhs' form or an implicit '= 0' form.
        """
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
                embed = err_embed("No variables found in the equation to solve for.")
                await ctx.send(embed=embed)
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
            embed = err_embed(
                f"Could not solve `{equation}`.\n**Reason:** {exc}"
            )
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
        """
        Computes the analytical derivative of the expression.
        Defaults to differentiating with respect to 'x'; falls back to
        whichever symbol is found if 'x' is absent.
        """
        try:
            expr = safe_parse(expression)
            free = expr.free_symbols

            # Pick the differentiation variable: prefer x, else first alphabetically
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
        """
        Computes the indefinite integral analytically using SymPy.
        The constant of integration is noted but not appended to the result.
        """
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
        """
        Vectorizes a SymPy expression via lambdify, evaluates over a numpy
        linspace, and renders a Matplotlib figure attached to the reply embed.
        The temporary file is always cleaned up in a finally block.
        """
        # --- Input validation -------------------------------------------------
        if x_min >= x_max:
            await ctx.send(embed=err_embed(
                f"`x_min` ({x_min}) must be strictly less than `x_max` ({x_max})."
            ))
            return

        filename = f"graph_{ctx.author.id}.png"

        await ctx.defer()          # Acknowledge early; graph generation may take >3 s

        try:
            # --- Parse & lambdify -----------------------------------------
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

            # --- Generate points -------------------------------------------
            x_vals = np.linspace(x_min, x_max, 800)
            y_vals = f_numeric(x_vals)

            # Mask non-finite values (poles, discontinuities)
            y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)

            # --- Render chart ----------------------------------------------
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

            # --- Build embed and attach file ------------------------------
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
            # Always remove the temp file — no disk leaks
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
        """
        Generates a randomised math question at the chosen difficulty level.
        Waits up to 15 seconds for the user's reply. Validates the answer
        by author ID, channel, and numeric comparison.
        """
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

        # --- Response filter ----------------------------------------------
        def check(message: discord.Message) -> bool:
            return (
                message.author.id == ctx.author.id
                and message.channel.id == ctx.channel.id
            )

        try:
            response: discord.Message = await self.bot.wait_for(
                "message",
                check=check,
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            timeout_embed = discord.Embed(
                title="⏰ Time's up!",
                description=f"You ran out of time.\n**Correct answer:** `{answer}`",
                colour=COLOUR_ERR,
            )
            await ctx.send(embed=timeout_embed)
            return

        # --- Evaluate the response ----------------------------------------
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
        """
        Returns a (question_string, numeric_answer) tuple.

        Easy  — addition / subtraction with small integers
        Medium — multiplication / division with clean integer results
        Hard  — linear equations of the form ax + b = c  (solve for x)
        """
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
                answer = a * b
                return f"What is {a} × {b}?", answer
            else:
                # Guarantee integer result: pick divisor then scale
                b = random.randint(2, 12)
                answer = random.randint(2, 12)
                a = b * answer
                return f"What is {a} ÷ {b}?", answer

        else:  # Hard
            # ax + b = c  →  x = (c - b) / a
            a = random.randint(2, 10)
            x = random.randint(-10, 10)
            b = random.randint(-20, 20)
            c = a * x + b
            answer = x
            return f"Solve for x:  {a}x + ({b}) = {c}", answer

    @staticmethod
    def _check_answer(user_input: str, correct: float | int) -> bool:
        """
        Numerically compares the user's string input against the correct answer.
        Accepts integer and float representations with a small tolerance.
        """
        try:
            user_val = float(user_input.replace(",", "").strip())
            return abs(user_val - float(correct)) < 1e-6
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Cog setup
# ---------------------------------------------------------------------------

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MathAssistant(bot))
    logger.info("MathAssistant Cog registered.")
