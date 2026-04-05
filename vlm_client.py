"""Client for LM Studio's OpenAI-compatible vision API."""

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

# Prepended to all prompts to suppress extended chain-of-thought reasoning
# which wastes output tokens in models like Gemma 4 and Qwen 3.5
_NO_THINK_PREFIX = (
    "IMPORTANT: Respond DIRECTLY with the requested output. "
    "Do NOT include any internal reasoning, thinking, or chain-of-thought. "
    "Output ONLY the final answer.\n\n"
)


def _get_client() -> OpenAI:
    return OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY)


def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_content(response) -> str:
    """
    Extract usable text from a chat completion response.
    Handles reasoning models that put output in reasoning_content
    instead of content.
    """
    msg = response.choices[0].message
    content = msg.content or ""
    if content.strip():
        return content

    # Reasoning models fallback
    reasoning = getattr(msg, "reasoning_content", None)
    if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
        reasoning = msg.model_extra.get("reasoning_content", "")
    if reasoning and reasoning.strip():
        logger.warning(
            "Model returned empty content but has reasoning_content (%d chars). "
            "Consider switching to a non-reasoning model or increasing max_tokens.",
            len(reasoning),
        )
        # Try to extract JSON from reasoning
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', reasoning)
        if json_match:
            candidate = json_match.group(1)
            # Validate it's actual JSON
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return reasoning

    logger.warning("Model returned completely empty response.")
    return ""


def list_models() -> list[str]:
    """Return model IDs available in LM Studio."""
    try:
        client = _get_client()
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        logger.warning("Could not list models: %s", e)
        return []


def ask_vision(
    prompt: str,
    image_paths: list[str | Path],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Send one or more images with a text prompt to the VLM."""
    client = _get_client()
    model = model or config.DEFAULT_MODEL

    full_prompt = _NO_THINK_PREFIX + prompt
    content: list[dict] = [{"type": "text", "text": full_prompt}]
    for img in image_paths:
        b64 = _encode_image(img)
        logger.info("Attaching image: %s (%d bytes b64)", Path(img).name, len(b64))
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    logger.info("Sending vision request to model '%s' with %d image(s)...", model, len(image_paths))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = _extract_content(response)
    logger.info("VLM response length: %d chars, starts with: %s", len(result), result[:100])
    return result


def ask_text(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Text-only completion (used for rubric generation, report writing)."""
    client = _get_client()
    model = model or config.DEFAULT_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _NO_THINK_PREFIX + prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_content(response)
