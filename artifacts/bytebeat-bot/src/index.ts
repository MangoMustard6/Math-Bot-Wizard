/**
 * Bytebeat Bot — Discord.js v14
 * ─────────────────────────────
 * Generates 5-second Bytebeat audio clips from a user-supplied formula and
 * posts them as MP4/AAC files.
 *
 * Entry points
 *   Prefix : tm!bytebeat <formula>
 *   Slash  : /bytebeat formula:<formula>
 *
 * Required environment variables
 *   DISCORD_TOKEN_BYTEBEAT   — bot token for this application
 *   DISCORD_CLIENT_ID_BYTEBEAT — application (client) ID
 */

import {
  AttachmentBuilder,
  Client,
  Events,
  GatewayIntentBits,
  Message,
  REST,
  Routes,
  SlashCommandBuilder,
  type ChatInputCommandInteraction,
} from "discord.js";
import ffmpeg from "fluent-ffmpeg";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

// ─── Config ──────────────────────────────────────────────────────────────────

const TOKEN     = process.env.DISCORD_TOKEN_BYTEBEAT     ?? "";
const CLIENT_ID = process.env.DISCORD_CLIENT_ID_BYTEBEAT ?? "";
const PREFIX    = "tm!";

const SAMPLE_RATE     = 8_000;                        // Hz
const DURATION_SECS   = 5;
const TOTAL_SAMPLES   = SAMPLE_RATE * DURATION_SECS;  // 40,000

if (!TOKEN || !CLIENT_ID) {
  console.error(
    "[bytebeat] DISCORD_TOKEN_BYTEBEAT and DISCORD_CLIENT_ID_BYTEBEAT must be set."
  );
  process.exit(1);
}

// ─── Input Sanitisation ──────────────────────────────────────────────────────

/**
 * Whitelist: digits, lowercase 't', whitespace, JS math operators, parentheses.
 * Everything else is rejected before a Function object is ever constructed.
 */
const SAFE_RE = /^[0-9t \+\-\*\/%&\|\^~<>()\s]+$/;

function sanitise(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed)        return null;
  if (!SAFE_RE.test(trimmed)) return null;
  return trimmed;
}

// ─── PCM Generation ──────────────────────────────────────────────────────────

/**
 * Compiles and runs the Bytebeat formula for every sample t ∈ [0, TOTAL_SAMPLES).
 * Returns a raw unsigned-8-bit mono PCM Buffer.
 *
 * Uses the Function constructor (not `eval`) so the formula runs in its own
 * scope — no access to the outer module. The sanitise() guard above ensures
 * only whitelisted characters reach this point.
 */
function generatePCM(formula: string): Buffer {
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const fn = new Function(
    "t",
    `"use strict"; return ((${formula}) & 255);`
  ) as (t: number) => number;

  const buf = Buffer.allocUnsafe(TOTAL_SAMPLES);
  for (let t = 0; t < TOTAL_SAMPLES; t++) {
    // Double-mask: once inside the function, once here — belt-and-suspenders.
    buf[t] = fn(t) & 255;
  }
  return buf;
}

// ─── FFmpeg Transcoding ───────────────────────────────────────────────────────

/**
 * Converts a raw u8 PCM Buffer → MP4 container with AAC audio.
 * Writes to a temp file and resolves with its path.
 * The caller is responsible for unlinking the output file after use.
 */
function transcodePCMtoMP4(pcm: Buffer): Promise<string> {
  return new Promise((resolve, reject) => {
    const stamp  = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const inPath  = path.join(os.tmpdir(), `bb_in_${stamp}.raw`);
    const outPath = path.join(os.tmpdir(), `bb_out_${stamp}.mp4`);

    fs.writeFileSync(inPath, pcm);

    ffmpeg()
      .input(inPath)
      // Tell FFmpeg the input format: unsigned 8-bit, 8kHz, mono
      .inputOptions("-f",  "u8")
      .inputOptions("-ar", String(SAMPLE_RATE))
      .inputOptions("-ac", "1")
      .audioCodec("aac")
      // +faststart moves the MP4 header to the front for instant streaming
      .outputOptions("-movflags", "+faststart")
      .output(outPath)
      .on("end", () => {
        try { fs.unlinkSync(inPath); } catch { /* best-effort */ }
        resolve(outPath);
      })
      .on("error", (err: Error) => {
        try { fs.unlinkSync(inPath); } catch { /* best-effort */ }
        reject(err);
      })
      .run();
  });
}

// ─── Shared Execution Core ────────────────────────────────────────────────────

interface BytebeatResult {
  filePath: string | null;
  error:    string | null;
}

/**
 * Full pipeline: sanitise → generate PCM → transcode → return MP4 path.
 * Never throws — all errors are returned as human-readable strings.
 */
async function executeBytebeat(rawFormula: string): Promise<BytebeatResult> {
  // 1. Sanitise input
  const formula = sanitise(rawFormula);
  if (!formula) {
    return {
      filePath: null,
      error:
        "❌ **Invalid formula.** Only digits, `t`, spaces, and the operators " +
        "`+ - * / % & | ^ ~ << >> >>>` with parentheses are allowed.",
    };
  }

  // 2. Generate raw PCM
  let pcm: Buffer;
  try {
    pcm = generatePCM(formula);
  } catch (err) {
    return {
      filePath: null,
      error: `❌ **Formula error:** \`${(err as Error).message}\``,
    };
  }

  // 3. Transcode to MP4/AAC
  try {
    const filePath = await transcodePCMtoMP4(pcm);
    return { filePath, error: null };
  } catch (err) {
    return {
      filePath: null,
      error: `❌ **Transcode error:** \`${(err as Error).message}\``,
    };
  }
}

// ─── Discord Client ───────────────────────────────────────────────────────────

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

// ─── Slash Command Definition ─────────────────────────────────────────────────

const bytebeatSlash = new SlashCommandBuilder()
  .setName("bytebeat")
  .setDescription("Generate a 5-second Bytebeat audio clip from a formula.")
  .addStringOption((opt) =>
    opt
      .setName("formula")
      .setDescription("Bytebeat expression using 't' — e.g. t>>2|t*3")
      .setRequired(true)
  );

// ─── Ready: register slash command ───────────────────────────────────────────

client.once(Events.ClientReady, async (ready) => {
  console.log(`[bytebeat] Online as ${ready.user.tag}`);

  const rest = new REST({ version: "10" }).setToken(TOKEN);
  try {
    await rest.put(Routes.applicationCommands(CLIENT_ID), {
      body: [bytebeatSlash.toJSON()],
    });
    console.log("[bytebeat] /bytebeat registered globally.");
  } catch (err) {
    console.error("[bytebeat] Slash command registration failed:", err);
  }
});

// ─── Slash Command Handler ────────────────────────────────────────────────────

client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName !== "bytebeat") return;

  const raw = interaction.options.getString("formula", true);

  // Defer immediately — FFmpeg can take a few seconds
  await interaction.deferReply();

  const { filePath, error } = await executeBytebeat(raw);

  if (error || !filePath) {
    await interaction.editReply(error ?? "❌ Unknown error.");
    return;
  }

  try {
    await interaction.editReply({
      content: `🎵 **Bytebeat** \`${raw}\``,
      files: [new AttachmentBuilder(filePath, { name: "bytebeat.mp4" })],
    });
  } finally {
    try { fs.unlinkSync(filePath); } catch { /* best-effort cleanup */ }
  }
});

// ─── Prefix Command Handler ───────────────────────────────────────────────────

client.on(Events.MessageCreate, async (message: Message) => {
  // Ignore bots and messages that don't start with our prefix
  if (message.author.bot)                           return;
  if (!message.content.startsWith(PREFIX))          return;

  // Parse: tm!bytebeat <formula...>
  const withoutPrefix = message.content.slice(PREFIX.length).trim();
  if (!withoutPrefix.startsWith("bytebeat"))        return;

  const rawFormula = withoutPrefix.slice("bytebeat".length).trim();

  if (!rawFormula) {
    await message.reply(
      "⚠️ Provide a formula. Example: `tm!bytebeat t>>2|t*3`"
    );
    return;
  }

  // Show typing indicator during generation + transcode (not available on all channel types)
  if ("sendTyping" in message.channel) await message.channel.sendTyping();

  const { filePath, error } = await executeBytebeat(rawFormula);

  if (error || !filePath) {
    await message.reply(error ?? "❌ Unknown error.");
    return;
  }

  try {
    await message.reply({
      content: `🎵 **Bytebeat** \`${rawFormula}\``,
      files: [new AttachmentBuilder(filePath, { name: "bytebeat.mp4" })],
    });
  } finally {
    try { fs.unlinkSync(filePath); } catch { /* best-effort cleanup */ }
  }
});

// ─── Start ────────────────────────────────────────────────────────────────────

client.login(TOKEN).catch((err) => {
  console.error("[bytebeat] Login failed:", err);
  process.exit(1);
});
