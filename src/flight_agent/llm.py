"""Chat model configuration: Google Gemini through the Gemini API.

Every pattern receives a LangChain ``BaseChatModel``; nothing else in the package
depends on the provider. The default client is ``ChatGoogleGenerativeAI``.
``FLIGHT_AGENT_MODEL_FACTORY`` replaces it with any callable that returns a
``BaseChatModel`` (routing, caching, a different client), without touching the
harness or the patterns.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, SecretStr

from flight_agent.harness.budget import Pricing

DEFAULT_MODEL = "gemini-3.6-flash"

ModelFactory = Callable[[str, "ModelSettings"], BaseChatModel]
"""``factory(model_id, settings) -> BaseChatModel``."""


class ModelSettings(BaseModel):
    model: str = DEFAULT_MODEL
    api_key: SecretStr | None = None
    executor_model: str | None = None
    temperature: float | None = None
    """``None`` keeps the model's default (recommended for Gemini 3 models)."""
    thinking_level: str | None = None
    """``minimal``, ``low``, ``medium`` or ``high``; ``None`` keeps the model's default."""
    max_retries: int = 6
    timeout_s: float = 120.0
    requests_per_minute: float | None = None
    pricing: Pricing | None = None
    factory: str | None = None
    """``module:function`` or ``path/to/file.py:function`` returning a chat model."""

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> ModelSettings:
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)

        def optional(name: str) -> str | None:
            value = os.environ.get(name, "").strip()
            return value or None

        factory = optional("FLIGHT_AGENT_MODEL_FACTORY")
        api_key = optional("GOOGLE_API_KEY")
        if not api_key and not factory:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY. Copy .env.example to .env and set it "
                "(or set FLIGHT_AGENT_MODEL_FACTORY to supply your own model client)."
            )
        price_in = optional("FLIGHT_AGENT_PRICE_INPUT_PER_MTOK")
        price_out = optional("FLIGHT_AGENT_PRICE_OUTPUT_PER_MTOK")
        temperature = optional("FLIGHT_AGENT_TEMPERATURE")
        rpm = optional("FLIGHT_AGENT_REQUESTS_PER_MINUTE")
        return cls(
            model=optional("FLIGHT_AGENT_MODEL") or DEFAULT_MODEL,
            api_key=SecretStr(api_key) if api_key else None,
            executor_model=optional("FLIGHT_AGENT_EXECUTOR_MODEL"),
            temperature=float(temperature) if temperature else None,
            thinking_level=optional("FLIGHT_AGENT_THINKING_LEVEL"),
            max_retries=int(optional("FLIGHT_AGENT_MAX_RETRIES") or 6),
            timeout_s=float(optional("FLIGHT_AGENT_TIMEOUT_S") or 120),
            requests_per_minute=float(rpm) if rpm else None,
            pricing=(
                Pricing(input_per_mtok=float(price_in), output_per_mtok=float(price_out))
                if price_in and price_out
                else None
            ),
            factory=factory,
        )


class Models(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    main: BaseChatModel
    executor: BaseChatModel | None = None
    name: str
    pricing: Pricing | None = None


def gemini_factory(settings: ModelSettings) -> ModelFactory:
    """The default factory. Both models share one client-side rate limiter because
    they usually share one quota."""
    limiter = (
        InMemoryRateLimiter(
            requests_per_second=settings.requests_per_minute / 60.0,
            check_every_n_seconds=0.05,
            max_bucket_size=1,
        )
        if settings.requests_per_minute
        else None
    )

    def make(model: str, settings: ModelSettings) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.api_key,
            temperature=settings.temperature,
            thinking_level=settings.thinking_level,
            max_retries=settings.max_retries,
            timeout=settings.timeout_s,
            rate_limiter=limiter,
        )

    return make


def load_factory(spec: str) -> ModelFactory:
    """Resolve ``module:function`` or ``path/to/file.py:function``."""
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise ValueError(f"invalid model factory {spec!r}; expected 'module:function'")
    if target.endswith(".py"):
        path = Path(target).resolve()
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load model factory from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, attr)


def build_models(settings: ModelSettings) -> Models:
    """Create the main model and, optionally, a separate (smaller) executor model."""
    make = load_factory(settings.factory) if settings.factory else gemini_factory(settings)
    return Models(
        main=make(settings.model, settings),
        executor=make(settings.executor_model, settings) if settings.executor_model else None,
        name=settings.model + (f" + {settings.executor_model}" if settings.executor_model else ""),
        pricing=settings.pricing,
    )
