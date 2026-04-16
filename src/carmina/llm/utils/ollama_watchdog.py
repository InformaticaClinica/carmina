"""
Ollama Watchdog — resilience helpers for Ollama local inference.

Responsibilities:
- Detect stalled / timed-out Ollama requests.
- Kill the hanging Ollama process (VRAM/RAM cleanup) and restart it.
- Validate that a response is genuine (not a leaked prompt echo / template residual).
- Provide a retry loop that survives transient OOM stalls.
"""

import logging
import re
import subprocess
import sys
import time

import requests

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tunable defaults (can be overridden via env vars in local_provider.py)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_REQUEST_TIMEOUT = 300          # seconds to wait for /api/chat response
DEFAULT_RESTART_WAIT    = 12           # seconds to wait after restarting Ollama
DEFAULT_MAX_RETRIES     = 3            # total attempts per call before giving up


# ──────────────────────────────────────────────────────────────────────────────
# Bad-response patterns — things that mean Ollama echoed the prompt / failed
# ──────────────────────────────────────────────────────────────────────────────
# These are checked against the *stripped* model response
_BAD_RESPONSE_PATTERNS = [
    # Literal XML/template tags from our prompts
    re.compile(r"<user_prompt\s*>",         re.IGNORECASE),
    re.compile(r"<input_text\s*>",          re.IGNORECASE),
    re.compile(r"<task\s*>",                re.IGNORECASE),
    re.compile(r"<system_prompt\s*>",       re.IGNORECASE),
    # Template placeholders left unresolved
    re.compile(r"input\s+missing",         re.IGNORECASE),
    re.compile(r"\[input_text\]",          re.IGNORECASE),
    re.compile(r"\{\{\s*input",            re.IGNORECASE),
    # Model confessing it got no input
    re.compile(r"no (input|text) (was |)provided", re.IGNORECASE),
    # Truncated / incomplete generation sentinel
    re.compile(r"^\.\.\.\s*$"),
]


def is_bad_response(text: str) -> tuple[bool, str]:
    """
    Return (True, reason) when *text* looks like a corrupt/echoed-prompt
    response that should be retried; (False, '') otherwise.

    An *empty* response is NOT considered bad here — the caller may legitimately
    receive an empty string when there are no entities to anonymise.
    """
    if not text:
        return False, ""

    for pattern in _BAD_RESPONSE_PATTERNS:
        if pattern.search(text):
            return True, f"response matches bad-pattern '{pattern.pattern}'"

    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
# Ollama process management
# ──────────────────────────────────────────────────────────────────────────────

def _kill_ollama_windows() -> bool:
    """Kill all ollama.exe processes on Windows. Returns True if something was killed."""
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "ollama.exe", "/T"],
            capture_output=True, text=True, timeout=15
        )
        killed = result.returncode == 0
        if killed:
            logger.warning("OllamaWatchdog: ollama.exe killed via taskkill.")
        else:
            logger.debug(f"OllamaWatchdog: taskkill stderr → {result.stderr.strip()}")
        return killed
    except Exception as exc:
        logger.error(f"OllamaWatchdog: could not kill Ollama on Windows: {exc}")
        return False


def _kill_ollama_unix() -> bool:
    """Kill all ollama processes on Linux/macOS. Returns True if something was killed."""
    try:
        result = subprocess.run(
            ["pkill", "-9", "-x", "ollama"],
            capture_output=True, text=True, timeout=15
        )
        killed = result.returncode == 0
        if killed:
            logger.warning("OllamaWatchdog: 'ollama' process killed via pkill.")
        return killed
    except Exception as exc:
        logger.error(f"OllamaWatchdog: could not kill Ollama on Unix: {exc}")
        return False


def kill_ollama() -> bool:
    """Platform-aware Ollama process kill. Returns True if a process was killed."""
    if sys.platform.startswith("win"):
        return _kill_ollama_windows()
    return _kill_ollama_unix()


def _start_ollama_unix() -> bool:
    """Start ollama serve in the background on Linux/macOS."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("OllamaWatchdog: 'ollama serve' launched.")
        return True
    except FileNotFoundError:
        logger.warning("OllamaWatchdog: 'ollama' binary not found in PATH — cannot auto-restart.")
        return False
    except Exception as exc:
        logger.error(f"OllamaWatchdog: failed to start ollama serve: {exc}")
        return False


def _start_ollama_windows() -> bool:
    """Start ollama.exe on Windows (it registers as a service/tray, so just launching is enough)."""
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        logger.info("OllamaWatchdog: 'ollama serve' launched on Windows.")
        return True
    except FileNotFoundError:
        logger.warning("OllamaWatchdog: 'ollama' not found in PATH — cannot auto-restart on Windows.")
        return False
    except Exception as exc:
        logger.error(f"OllamaWatchdog: Windows start failed: {exc}")
        return False


def restart_ollama(base_url: str, wait: int = DEFAULT_RESTART_WAIT) -> bool:
    """
    Kill Ollama, wait, restart it, then wait until /api/version responds.

    Returns True if Ollama is reachable after the restart, False otherwise.
    """
    logger.warning("OllamaWatchdog: initiating Ollama restart sequence…")
    kill_ollama()
    time.sleep(3)  # give the OS time to free VRAM/RAM

    if sys.platform.startswith("win"):
        started = _start_ollama_windows()
    else:
        started = _start_ollama_unix()

    if not started:
        logger.error("OllamaWatchdog: could not start Ollama — continuing without restart.")
        return False

    # Poll until it's back up
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/version", timeout=3)
            if r.status_code == 200:
                logger.info(f"OllamaWatchdog: Ollama is back (version {r.json().get('version','?')}).")
                # Extra cooldown so the model context is fully cleared
                time.sleep(2)
                return True
        except Exception:
            pass
        time.sleep(1)

    logger.error("OllamaWatchdog: Ollama did not come back within the wait window.")
    return False


def unload_model(base_url: str, ollama_model: str) -> None:
    """
    Ask Ollama to evict the model from VRAM by setting keep_alive=0.
    This is the clean way to free VRAM without killing the process.
    """
    try:
        payload = {"model": ollama_model, "keep_alive": 0}
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=10)
        if r.status_code == 200:
            logger.info(f"OllamaWatchdog: model '{ollama_model}' unloaded from VRAM.")
        else:
            logger.warning(f"OllamaWatchdog: unload returned {r.status_code}")
    except Exception as exc:
        logger.warning(f"OllamaWatchdog: could not unload model: {exc}")
