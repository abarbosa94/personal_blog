"""Provider-neutral contract for translation-quality judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class JudgeConfiguration:
    provider: str
    provider_name: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int

    def request_settings(self) -> dict[str, Any]:
        return {
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
        }


@dataclass(frozen=True)
class JudgeCompletion:
    result: BaseModel
    response_model: str | None
    finish_reason: str | None
    usage: dict[str, Any] | None
    reasoning_content_omitted: bool
    api_response: dict[str, Any]


class JudgeAdapter(ABC):
    """Interface implemented by every external judge provider."""

    provider: str
    provider_name: str
    default_model: str
    allowed_reasoning_efforts: tuple[str, ...]
    default_reasoning_effort: str
    default_max_completion_tokens: int

    def __init__(self, model: str | None = None) -> None:
        self.model = model or self.default_model
        self._client: Any = None

    def configuration(
        self,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> JudgeConfiguration:
        effort = reasoning_effort or self.default_reasoning_effort
        if effort not in self.allowed_reasoning_efforts:
            allowed = ", ".join(self.allowed_reasoning_efforts)
            raise ValueError(
                f"{self.provider} reasoning effort must be one of: {allowed}"
            )
        token_limit = (
            self.default_max_completion_tokens
            if max_completion_tokens is None
            else max_completion_tokens
        )
        if token_limit < 1:
            raise ValueError("max_completion_tokens must be positive")
        return JudgeConfiguration(
            provider=self.provider,
            provider_name=self.provider_name,
            model=self.model,
            reasoning_effort=effort,
            max_completion_tokens=token_limit,
        )

    @abstractmethod
    def judge(
        self,
        system_prompt: str,
        payload: dict[str, str],
        schema: type[BaseModel],
        configuration: JudgeConfiguration,
    ) -> JudgeCompletion:
        raise NotImplementedError
