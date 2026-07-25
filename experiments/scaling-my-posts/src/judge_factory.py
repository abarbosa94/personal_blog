"""Registry and construction helpers for judge adapters."""

from __future__ import annotations

from judge_interface import JudgeAdapter, JudgeConfiguration
from judge_providers import PROVIDER_ADAPTERS


class JudgeAdapterFactory:
    _adapters: dict[str, type[JudgeAdapter]] = {}

    @classmethod
    def register(cls, adapter: type[JudgeAdapter]) -> None:
        cls._adapters[adapter.provider] = adapter

    @classmethod
    def create(cls, provider: str, model: str | None = None) -> JudgeAdapter:
        try:
            adapter = cls._adapters[provider]
        except KeyError as error:
            available = ", ".join(cls.providers())
            raise ValueError(
                f"Unknown judge provider {provider!r}; choose one of: {available}"
            ) from error
        return adapter(model)

    @classmethod
    def providers(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._adapters))


for provider_adapter in PROVIDER_ADAPTERS:
    JudgeAdapterFactory.register(provider_adapter)


DEFAULT_JUDGE_PROVIDER = "kimi"


def create_judge_adapter(provider: str, model: str | None = None) -> JudgeAdapter:
    return JudgeAdapterFactory.create(provider, model)


def available_judge_providers() -> tuple[str, ...]:
    return JudgeAdapterFactory.providers()


def resolve_judge_configuration(
    provider: str = DEFAULT_JUDGE_PROVIDER,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> JudgeConfiguration:
    return create_judge_adapter(provider, model).configuration(
        reasoning_effort, max_completion_tokens
    )
