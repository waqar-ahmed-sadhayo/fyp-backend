"""Thin wrapper around the Gemini SDK, shared by every AI-backed feature
(the per-result "AI Suggestions" endpoint in ai_suggestions.py, and the
general assistant chatbot in assistant_chat.py). Kept separate from the
blueprint modules so the client setup and error normalization live in
exactly one place.

Originally built against the Anthropic API, switched to Gemini because that
was the key actually available for this deployment — the public functions
here (generate_chat_response, generate_response, contains_emergency_keywords,
AIAssistantError) kept the same shapes on purpose, so ai_suggestions.py and
assistant_chat.py needed zero changes for the swap."""

import os

from flask import current_app
from google import genai
from google.genai import errors, types

# Configurable so the model can be swapped without a code change.
# gemini-2.5-flash was retired for new users ("no longer available to new
# users" API error) — gemini-3.6-flash is the model the API itself now
# points to as the replacement.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Emergency-symptom keywords (English + common Roman Urdu spellings) — used
# to flag a response as urgent so the frontend can show a red banner
# regardless of exactly how the model phrases it. Deliberately checked
# against both the user-facing input and the model's own reply: the reply is
# what the safety banner is *about*, but scanning the input too means a user
# who typed "chest pain" still gets flagged even if the wording varies.
EMERGENCY_KEYWORDS = [
    "chest pain", "seene mein dard", "seenay mein dard",
    "shortness of breath", "difficulty breathing", "breathing issue",
    "saans ki takleef", "saans lene mein", "saans ki dushwari",
    "heavy bleeding", "severe bleeding", "zyada khoon", "khoon beh raha",
    "emergency", "emergency room", "hospital foran", "foran hospital",
    "stroke", "unconscious", "behosh", "heart attack", "dil ka daura",
]


class AIAssistantError(Exception):
    """Normalized error for callers — carries an HTTP-appropriate status
    code so blueprint handlers don't need to know about the underlying
    SDK's exception hierarchy."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_client = None


def _get_client():
    global _client
    api_key = current_app.config.get("GEMINI_API_KEY")
    if not api_key:
        raise AIAssistantError(
            "The AI assistant is not configured on this server (missing GEMINI_API_KEY).",
            status_code=503,
        )
    if _client is None:
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30_000),  # ms
        )
    return _client


def contains_emergency_keywords(*texts):
    haystack = " ".join(t or "" for t in texts).lower()
    return any(kw in haystack for kw in EMERGENCY_KEYWORDS)


def _to_gemini_contents(messages):
    # Gemini uses "model" where our internal shape (and the frontend) uses
    # "assistant" — everything else maps straight across.
    return [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]


def generate_chat_response(system_prompt, messages, max_tokens=1200):
    """Calls the Gemini API with a full conversation (list of
    {"role": "user"|"assistant", "content": str}) and returns the reply
    text. Raises AIAssistantError (with a caller-safe message + status code)
    on any failure — auth/config, rate limit, or a general API error.

    gemini-3.6-flash spends part of max_output_tokens on internal "thinking"
    before the visible reply (thinking_budget=0 is rejected outright by this
    model — a 400 INVALID_ARGUMENT — so it can't be disabled). thinking_level
    "low" keeps that overhead down, but max_tokens still has to budget for
    thinking *and* the answer together, so callers should pass something
    comfortably larger than "how long should the visible reply be" — 350
    tokens looked reasonable for a 3-4 sentence answer but left the model
    cut off mid-sentence after ~300 tokens of pure thinking."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=_to_gemini_contents(messages),
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
    except errors.ClientError as e:
        if e.code == 401 or e.code == 403:
            raise AIAssistantError("AI service authentication failed.", status_code=503) from e
        if e.code == 429:
            raise AIAssistantError("AI service is rate-limited right now — please try again shortly.", status_code=429) from e
        raise AIAssistantError("AI service rejected the request — please try again.", status_code=502) from e
    except errors.ServerError as e:
        raise AIAssistantError("AI service is temporarily unavailable — please try again.", status_code=502) from e
    except TimeoutError as e:
        raise AIAssistantError("AI service took too long to respond — please try again.", status_code=504) from e
    except Exception as e:
        raise AIAssistantError("Could not reach the AI service — please try again.", status_code=502) from e

    text = response.text
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    if finish_reason == types.FinishReason.MAX_TOKENS:
        # Cut off mid-answer (thinking ate too much of the budget) — surface
        # this as a clean error rather than silently showing a half sentence.
        raise AIAssistantError("AI response was cut off — please try again.", status_code=502)
    if not text:
        raise AIAssistantError("AI service returned an empty response — please try again.", status_code=502)
    return text


def generate_response(system_prompt, user_message, max_tokens=1024):
    """Single-turn convenience wrapper over generate_chat_response, used by
    the per-result AI Suggestions feature."""
    return generate_chat_response(
        system_prompt, [{"role": "user", "content": user_message}], max_tokens=max_tokens,
    )
