"""Concrete provider adapters for the translation-quality judge."""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel

from judge_interface import JudgeAdapter, JudgeCompletion, JudgeConfiguration


def _openai_sdk() -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install the openai package before running the judge.") from error
    return OpenAI


def _inline_local_json_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline Pydantic's local definitions for a simple MFJS-compatible schema."""
    definitions = schema.get("$defs", {})

    def expand(value: Any, trail: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [expand(item, trail) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            reference = value["$ref"]
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise ValueError(f"Unsupported JSON Schema reference: {reference!r}")
            name = reference.removeprefix(prefix)
            if name not in definitions:
                raise ValueError(f"Unknown JSON Schema definition: {name!r}")
            if name in trail:
                raise ValueError(
                    f"Recursive JSON Schema definition is unsupported: {name!r}"
                )
            siblings = {key: item for key, item in value.items() if key != "$ref"}
            return expand({**definitions[name], **siblings}, (*trail, name))
        return {
            key: expand(item, trail)
            for key, item in value.items()
            if key != "$defs"
        }

    expanded = expand(schema)
    if not isinstance(expanded, dict):
        raise TypeError("The expanded JSON Schema must be an object")
    return expanded


def kimi_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__.lower(),
            "strict": True,
            "schema": _inline_local_json_schema_refs(schema.model_json_schema()),
        },
    }


def _without_reasoning_content(response: Any) -> tuple[dict[str, Any], bool]:
    """Serialize an API response while deliberately excluding hidden reasoning."""
    serialized = response.model_dump(mode="json")
    omitted = False
    for choice in serialized.get("choices", []):
        message = choice.get("message")
        if isinstance(message, dict) and "reasoning_content" in message:
            omitted = True
            message.pop("reasoning_content", None)
    return serialized, omitted


class KimiJudgeAdapter(JudgeAdapter):
    provider = "kimi"
    provider_name = "Moonshot AI"
    default_model = "kimi-k3"
    allowed_reasoning_efforts = ("low", "high", "max")
    default_reasoning_effort = "max"
    default_max_completion_tokens = 8_192
    api_base_url = "https://api.moonshot.ai/v1"
    api_key_environment_variable = "MOONSHOT_API_KEY"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_environment_variable)
            if not api_key:
                raise SystemExit(
                    f"Set {self.api_key_environment_variable} before running "
                    "the paid judge command."
                )
            self._client = _openai_sdk()(api_key=api_key, base_url=self.api_base_url)
        return self._client

    def judge(
        self,
        system_prompt: str,
        payload: dict[str, str],
        schema: type[BaseModel],
        configuration: JudgeConfiguration,
    ) -> JudgeCompletion:
        response = self._get_client().chat.completions.create(
            model=configuration.model,
            reasoning_effort=configuration.reasoning_effort,
            max_completion_tokens=configuration.max_completion_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=kimi_response_format(schema),
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(
                "Kimi returned an incomplete judgment with "
                f"finish_reason={choice.finish_reason!r}"
            )
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Kimi returned no final structured content")
        result = schema.model_validate_json(content)
        api_response, reasoning_omitted = _without_reasoning_content(response)
        return JudgeCompletion(
            result=result,
            response_model=api_response.get("model"),
            finish_reason=choice.finish_reason,
            usage=api_response.get("usage"),
            reasoning_content_omitted=reasoning_omitted,
            api_response=api_response,
        )


class OpenAIJudgeAdapter(JudgeAdapter):
    provider = "openai"
    provider_name = "OpenAI"
    default_model = "gpt-5.5-2026-04-23"
    allowed_reasoning_efforts = ("low", "medium", "high")
    default_reasoning_effort = "medium"
    default_max_completion_tokens = 2_500
    api_key_environment_variable = "OPENAI_API_KEY"

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_environment_variable)
            if not api_key:
                raise SystemExit(
                    f"Set {self.api_key_environment_variable} before running "
                    "the paid judge command."
                )
            self._client = _openai_sdk()(api_key=api_key)
        return self._client

    def judge(
        self,
        system_prompt: str,
        payload: dict[str, str],
        schema: type[BaseModel],
        configuration: JudgeConfiguration,
    ) -> JudgeCompletion:
        response = self._get_client().responses.parse(
            model=configuration.model,
            reasoning={"effort": configuration.reasoning_effort},
            max_output_tokens=configuration.max_completion_tokens,
            store=False,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text_format=schema,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("OpenAI returned no parsed result")
        api_response = response.model_dump(mode="json")
        return JudgeCompletion(
            result=result,
            response_model=api_response.get("model"),
            finish_reason=api_response.get("status"),
            usage=api_response.get("usage"),
            reasoning_content_omitted=False,
            api_response=api_response,
        )


PROVIDER_ADAPTERS: tuple[type[JudgeAdapter], ...] = (
    KimiJudgeAdapter,
    OpenAIJudgeAdapter,
)
