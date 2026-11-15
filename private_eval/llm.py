"""LLM transports with a disk cache: OpenAI-compatible API (Caila by default) and local Ollama."""
import hashlib
import json
import os
import re
import socket
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CAILA_URL = "https://caila.io/api/adapters/openai"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
API_MODELS = ROOT / "research" / "experiments" / "api_models.json"
API_ALLOWED = ["qwen3-235b-instruct"]
READ_CACHE = True
OLLAMA_MODELS = {"avibe-q8": "hf.co/NightForger/avibe-GGUF:Q8_0", "t-lite-q8": "t-lite-it-2.1:q8_0"}


def load_env(path=HERE / ".env"):
    """Keys from the gitignored private_eval/.env; variables already set in the shell take priority."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name and not name.startswith("#"):
            os.environ.setdefault(name.strip(), value.strip())


load_env()


def _cache_path(kind, payload):
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return CACHE / kind / digest[:2] / f"{digest}.json"


class ApiClient:
    """OpenAI-compatible chat completions. Caila expects the bare key in Authorization."""

    def __init__(self, model, base_url=None, key_env="CAILA_API_KEY", bearer=False, timeout=90, retries=3):
        if model not in API_ALLOWED and os.environ.get("ALLOW_ANY_API_MODEL") != "1":
            raise SystemExit(f"Платная модель {model} не в списке разрешённых {API_ALLOWED}. "
                             "Добавьте её в API_ALLOWED в private_eval/llm.py или задайте ALLOW_ANY_API_MODEL=1")
        catalog = {m["name"]: m for m in json.loads(API_MODELS.read_text(encoding="utf-8"))}
        entry = catalog.get(model, {"name": model, "model": model, "params": {"temperature": 0}})
        self.name, self.model_id, self.params = entry["name"], entry["model"], dict(entry.get("params", {}))
        self.base_url = (base_url or os.environ.get("API_BASE_URL") or CAILA_URL).rstrip("/")
        key = os.environ.get(key_env, "")
        if not key:
            raise SystemExit(f"Нет ключа: задайте {key_env} в окружении или строкой {key_env}=... в private_eval/.env")
        self.auth = f"Bearer {key}" if bearer else key
        self.timeout, self.retries = timeout, retries
        self.tag = self.model_id

    def check(self):
        """Free catalog request before a paid run, so a blocked API never turns into a run of Stage 1 fallbacks."""
        from urllib.parse import urlsplit
        host = urlsplit(self.base_url).hostname
        try:
            address = socket.gethostbyname(host)
        except OSError:
            raise SystemExit(f"Не удаётся найти адрес {host}: проверьте сеть") from None
        vpn = address.startswith(("198.18.", "198.19."))
        hint = (f" Адрес {host} подменён VPN ({address}), запрос уходит в туннель. Добавьте {host} в прямое "
                "подключение в правилах VPN или задайте HTTPS_PROXY с российским адресом.") if vpn else ""
        try:
            with urlopen(Request(self.base_url + "/models", headers={"Authorization": self.auth,
                                                                     "Accept": "application/json"}),
                         timeout=30) as resp:
                json.load(resp)
        except HTTPError as exc:
            body = exc.read(300).decode("utf-8", "replace")
            exc.close()
            if exc.code == 403 and "<html" in body.lower():
                raise SystemExit("API закрыт для текущего IP (HTTP 403 от nginx, Caila пускает только российские "
                                 "адреса)." + hint) from None
            if exc.code in (401, 403):
                raise SystemExit(f"API отклонил ключ (HTTP {exc.code}). Проверьте ключ в private_eval/.env") from None
            raise SystemExit(f"API ответил HTTP {exc.code} на проверку доступа") from None
        except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
            raise SystemExit(f"API недоступен: {type(exc).__name__}." + hint) from None

    def chat(self, system, user, max_tokens=96, temperature=None):
        payload = {**self.params, "model": self.model_id, "stream": False, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if temperature is not None:
            payload["temperature"] = temperature
        path = _cache_path("api", [self.base_url, payload])
        if READ_CACHE and path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        body = json.dumps(payload, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": self.auth}
        start, error = time.perf_counter(), None
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(self.base_url + "/chat/completions", data=body, headers=headers),
                             timeout=self.timeout) as resp:
                    value = json.load(resp)
                choice = (value.get("choices") or [{}])[0]
                usage = value.get("usage") or {}
                rec = {"raw": (choice.get("message") or {}).get("content"), "finish": choice.get("finish_reason"),
                       "seconds": time.perf_counter() - start, "input_tokens": usage.get("prompt_tokens", 0),
                       "output_tokens": usage.get("completion_tokens", 0), "error": None}
                if rec["raw"] is None:
                    rec["error"] = "empty_response"
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
                return {**rec, "cached": False}
            except HTTPError as exc:
                error = f"HTTP {exc.code}"
                exc.close()
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    break
            except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
                error = type(exc).__name__
            time.sleep(min(2 ** attempt, 10))
        return {"raw": None, "seconds": time.perf_counter() - start, "input_tokens": 0, "output_tokens": 0,
                "error": error, "cached": False}


class OllamaClient:
    def __init__(self, model="avibe-q8", num_ctx=8192, timeout=600):
        self.name, self.tag, self.num_ctx, self.timeout = model, OLLAMA_MODELS.get(model, model), num_ctx, timeout

    def _post(self, path, payload):
        req = Request(OLLAMA_URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def check(self):
        try:
            with urlopen(OLLAMA_URL + "/api/tags", timeout=10) as resp:
                tags = {m["name"] for m in json.load(resp)["models"]}
        except OSError:
            raise SystemExit("Ollama не отвечает: запустите приложение Ollama или ollama serve") from None
        if self.tag not in tags:
            raise SystemExit(f"В Ollama нет модели {self.tag}. Установлены: {', '.join(sorted(tags))}")

    def unload(self):
        try:
            self._post("/api/generate", {"model": self.tag, "keep_alive": 0})
        except OSError:
            pass

    def chat(self, system, user, max_tokens=96, temperature=0, schema=None):
        options = {"temperature": temperature, "seed": 42, "num_ctx": self.num_ctx, "num_predict": max_tokens}
        payload = {"model": self.tag, "stream": False, "keep_alive": "30m", "options": options,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None:
            payload["format"] = schema
        path = _cache_path("ollama", payload)
        if READ_CACHE and path.exists():
            return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
        start = time.perf_counter()
        try:
            resp = self._post("/api/chat", payload)
        except (OSError, ValueError) as exc:
            return {"raw": None, "seconds": time.perf_counter() - start, "input_tokens": 0, "output_tokens": 0,
                    "error": type(exc).__name__, "cached": False}
        rec = {"raw": resp.get("message", {}).get("content"), "seconds": time.perf_counter() - start,
               "input_tokens": resp.get("prompt_eval_count", 0), "output_tokens": resp.get("eval_count", 0),
               "error": None}
        if rec["input_tokens"] + max_tokens >= self.num_ctx:
            rec["error"] = "context_overflow"
        if rec["error"] is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return {**rec, "cached": False}


def _is_offer(value):
    """Evidence value: a quote (v8) or {"quote": ..., "status": ...} (v9), where only status "offer" counts."""
    if isinstance(value, dict):
        quote = value.get("quote")
        return value.get("status") == "offer" and isinstance(quote, str) and bool(quote.strip())
    return isinstance(value, str) and bool(value.strip())


def parse_labels(content, allowed):
    """Tolerant parse of {"labels": [...]}; returns sorted labels within allowed or None."""
    if not isinstance(content, str):
        return None
    text = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        found = re.search(r"\{[^{}]*\"labels\"[^{}]*\}", text)
        outer = text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""
        if '"evidence"' in outer:
            text = outer
        elif found:
            text = found.group(0)
        else:
            return None
    try:
        value = json.loads(text)
        if "labels" in value:
            labels = value["labels"]
        else:
            evidence = value["evidence"]
            labels = sorted(int(k) for k, v in evidence.items() if _is_offer(v))
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if not isinstance(labels, list) or any(type(x) is not int for x in labels):
        return None
    return sorted(set(labels) & set(allowed))


def label_schema(allowed):
    return {"type": "object", "required": ["labels"], "additionalProperties": False,
            "properties": {"labels": {"type": "array", "maxItems": len(allowed), "uniqueItems": True,
                                      "items": {"type": "integer", "enum": list(allowed)}}}}
