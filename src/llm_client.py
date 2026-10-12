"""Lightweight Ollama API client with R1 chain-of-thought handling."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import urlsplit

import requests  # type: ignore[import-untyped]

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LLM_TIMEOUT_SECONDS = 165

# ── EEE domain configuration ──────────────────────────────────────────────

EEE_SYSTEM_PROMPT = (
    "You are an expert tutor in Electronic and Electrical Engineering (EEE). "
    "Your knowledge covers: circuit analysis, signals and systems, "
    "semiconductor devices, power electronics, control systems, digital logic, "
    "electromagnetics, and communication systems.\n\n"
    "RULES:\n"
    "1. Only answer EEE questions. For non-EEE questions, reply: "
    "'I only answer EEE-related questions.'\n"
    "2. Think step-by-step, but do NOT re-check or re-derive the same step. "
    "Once you reach a conclusion, state it and move on. "
    "If you find yourself repeating the same reasoning, stop immediately "
    "and give your best answer based on what you have.\n"
    "3. Keep your reasoning concise. If a question has a straightforward "
    "factual answer, give it directly without overthinking.\n"
    "4. If you are uncertain, state your best answer with a brief note about "
    "the uncertainty rather than endlessly looping."
)

# Keywords used for the domain gate — cheap pre-filter before hitting the LLM.
EEE_KEYWORDS: set[str] = {
    # Circuits & analysis
    "ohm",
    "kirchhoff",
    "thevenin",
    "norton",
    "superposition",
    "voltage",
    "current",
    "resistance",
    "impedance",
    "capacitor",
    "inductor",
    "transformer",
    "rectifier",
    "inverter",
    "converter",
    # Semiconductor devices
    "mosfet",
    "bjt",
    "jfet",
    "diode",
    "transistor",
    "op-amp",
    "amplifier",
    "operational amplifier",
    "semiconductor",
    "pn junction",
    "cmos",
    "ttl",
    "logic gate",
    # Signals & systems
    "fourier",
    "laplace",
    "z-transform",
    "convolution",
    "filter",
    "bandwidth",
    "frequency",
    "sampling",
    "aliasing",
    "modulation",
    "demodulation",
    "pwm",
    "signal",
    # Power & energy
    "power factor",
    "rms",
    "three-phase",
    "ac power",
    "dc power",
    "motor",
    "generator",
    "sensor",
    "actuator",
    # Digital & control
    "pid",
    "feedback",
    "control system",
    "state-space",
    "truth table",
    "boolean",
    "flip-flop",
    "register",
    "combinational",
    "sequential",
    "pld",
    "fpga",
    # General EEE
    "eee",
    "electrical engineering",
    "electronic engineering",
    "circuit",
    "electromagnetic",
    "faraday",
    "lenz",
    "maxwell",
    "inductance",
    "capacitance",
    "reactance",
}


def is_eee_related(question: str) -> bool:
    """Cheap keyword-based pre-filter for EEE domain detection.

    Returns True if the question contains at least one EEE keyword.
    This is a fast guard before the LLM system prompt — it catches
    clearly non-EEE questions (e.g. 'write a poem') at zero inference cost.
    """
    q = question.lower()
    return any(kw in q for kw in EEE_KEYWORDS)


def strip_thinking(text: str) -> tuple[str, str]:
    """Split R1 output into (thinking, answer).

    DeepSeek-R1-Distill models wrap chain-of-thought in  <think>...</think>
    before the real answer.  This function pulls those apart so downstream
    consumers (RAG, TTS, etc.) never see the raw CoT.
    """
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thinking, answer
    return "", text.strip()


def effective_timeout_seconds(timeout: int) -> int:
    """Keep one backend request strictly inside the browser's 180-second budget."""
    try:
        configured = int(os.getenv("EEE_RAG_LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)))
    except ValueError:
        configured = DEFAULT_LLM_TIMEOUT_SECONDS
    configured = min(max(1, configured), DEFAULT_LLM_TIMEOUT_SECONDS)
    return min(max(1, int(timeout)), configured)


class IncompleteLLMStreamError(RuntimeError):
    """Raised when an Ollama stream is incomplete or output-token-limited."""


def model_supports_thinking(model: str) -> bool:
    """Return whether the known request format needs Ollama's thinking field."""
    normalized = model.strip().lower()
    return normalized.startswith(("qwen3:", "qwen3.5:")) or normalized.startswith("deepseek")


def generate(
    prompt: str,
    model: str = "deepseek-r1:1.5b-q4_k_m",
    *,
    timeout: int = 300,
    base_url: str = OLLAMA_URL,
    system_prompt: str | None = EEE_SYSTEM_PROMPT,
    think: bool | None = None,
    num_predict: int = 2048,
    num_ctx: int | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Send a prompt to Ollama and return timing + cleaned output.

    Parameters
    ----------
    prompt : str
        The user's question.
    model : str
        Ollama model name.
    timeout : int
        Wall-clock timeout in seconds (prevents R1 infinite CoT loops).
    base_url : str
        Ollama API endpoint URL.
    system_prompt : str | None
        System-level instruction sent before the user message.
        Defaults to ``EEE_SYSTEM_PROMPT`` which constrains the model to
        answer only EEE-related questions.  Pass ``None`` to disable.
    response_format : dict[str, Any] | None
        Optional native Ollama structured-output JSON schema.
    temperature : float | None
        Optional sampling temperature added to the Ollama options object.

    Returns a dict with keys:
        thinking  – CoT reasoning inside <think> tags (empty str if none)
        answer    – the actual answer (stripped of <think> tags)
        tokens    – number of completion tokens generated
        tps       – tokens per second
        total_s   – wall-clock seconds for the HTTP call
        ttft_s    – time-to-first-token in seconds
    """
    t0 = time.perf_counter()
    request_timeout = effective_timeout_seconds(timeout)
    first_token_at: float | None = None
    content_chunks: list[str] = []
    thinking_chunks: list[str] = []
    tokens = 0

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options: dict[str, Any] = {"num_predict": num_predict, "repeat_penalty": 1.1}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if temperature is not None:
        options["temperature"] = temperature

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    if think is not None and (think or model_supports_thinking(model)):
        payload["think"] = think
    if response_format is not None:
        payload["format"] = response_format

    request_kwargs: dict[str, Any] = {}
    if urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"}:
        request_kwargs["proxies"] = {"http": "", "https": ""}
    resp = requests.post(
        base_url,
        json=payload,
        timeout=request_timeout,
        stream=True,
        **request_kwargs,
    )
    resp.raise_for_status()

    stream_complete = False
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if time.perf_counter() - t0 > request_timeout:
                raise requests.exceptions.Timeout(
                    f"LLM stream exceeded its {request_timeout}-second request budget"
                )
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if first_token_at is None and "message" in chunk:
                first_token_at = time.perf_counter()

            if "message" in chunk:
                msg = chunk["message"]
                if "content" in msg:
                    content_chunks.append(msg["content"])
                if "thinking" in msg:
                    thinking_chunks.append(msg["thinking"])
            if chunk.get("done") is True:
                if chunk.get("done_reason") == "length":
                    raise IncompleteLLMStreamError(
                        "LLM response was truncated at the output token limit (done_reason=length)"
                    )
                tokens = chunk.get("eval_count", 0)
                stream_complete = True
                break
    finally:
        resp.close()

    if not stream_complete:
        raise IncompleteLLMStreamError(
            "LLM stream ended before an explicit done=true completion block"
        )

    elapsed = time.perf_counter() - t0
    ttft = (first_token_at - t0) if first_token_at is not None else elapsed
    raw_content = "".join(content_chunks)
    raw_thinking = "".join(thinking_chunks)
    tps = tokens / elapsed if elapsed > 0 else 0.0

    if raw_thinking:
        thinking = raw_thinking.strip()
        answer = raw_content.strip()
    else:
        thinking, answer = strip_thinking(raw_content)

    return {
        "thinking": thinking,
        "answer": answer,
        "tokens": tokens,
        "tps": round(tps, 2),
        "total_s": round(elapsed, 2),
        "ttft_s": round(ttft, 2),
    }
