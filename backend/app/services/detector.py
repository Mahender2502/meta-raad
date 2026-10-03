"""
Zero-shot anomaly detection prompts and detector (AD-LLM style) with optional
RA-ZAD retrieval grounding.

    - build_detection_prompt() builds the AD-LLM Setting 1 ("normal only") or
      Setting 2 ("normal + anomaly") prompt. When `exemplars` is given it adds a
      block of retrieved real examples (RA-ZAD); with no exemplars it is the plain
      AD-LLM zero-shot baseline (k = 0).
    - parse_detection_response() extracts {"reason", "anomaly_score"} from the
      LLM output, tolerating code fences and surrounding prose.
    - detect() sends one prompt through whichever LLMClient is configured.

Prompt wording follows notes/AD-LLM_Reference.md (Tables A9/A10 of the paper);
"reason" is requested before "anomaly_score" on purpose (implicit chain-of-thought).
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from app.services.llm_client import LLMClient

Setting = Literal["normal_only", "normal_anomaly"]

# Free-tier APIs return 429 with a "retry in Ns" hint. Wait and retry a couple
# of times so a quick second click in the demo doesn't just show an error.
RATE_LIMIT_RETRIES = 2
RATE_LIMIT_MAX_WAIT_S = 40.0
_RETRY_IN_RE = re.compile(r"retry in ([0-9.]+)s|retryDelay['\"]?\s*:\s*['\"]([0-9.]+)s", re.IGNORECASE)

# Max characters of each retrieved chunk placed in the prompt. Keeps prompts
# small for small models and free-tier token limits.
EXEMPLAR_MAX_CHARS = 600


@dataclass
class Exemplar:
    text: str
    category: str
    is_anomaly: bool = False
    doc_id: str | None = None
    chunk_id: str | None = None


@dataclass
class DetectionResult:
    prompt: str
    model: str | None = None
    score: float | None = None
    reason: str | None = None
    raw_text: str = ""
    error: str | None = None
    # 1-based numbers of the reference examples the model cited (RA-ZAD only);
    # already validated against the examples that were actually in the prompt.
    citations: list[int] = field(default_factory=list)


def _format_exemplars(exemplars: list[Exemplar], setting: Setting) -> str:
    lines = [
        "## Reference Examples:",
        "Real examples retrieved from the dataset, most similar to the text sample "
        "first. Use them only to understand what each category looks like; the text "
        "sample may differ from them.",
    ]
    for i, ex in enumerate(exemplars, start=1):
        snippet = ex.text.strip().replace("\n", " ")
        if len(snippet) > EXEMPLAR_MAX_CHARS:
            snippet = snippet[:EXEMPLAR_MAX_CHARS].rstrip() + "..."
        label = ex.category
        if setting == "normal_anomaly":
            label += " (anomaly category)" if ex.is_anomaly else " (normal category)"
        lines.append(f'{i}. [{label}] "{snippet}"')
    return "\n".join(lines)


def build_detection_prompt(
    text: str,
    normal_categories: list[str] | tuple[str, ...],
    anomaly_category: str | None = None,
    setting: Setting = "normal_only",
    exemplars: list[Exemplar] | None = None,
) -> str:
    """Build the AD-LLM detection prompt, optionally grounded with retrieved exemplars."""
    if setting == "normal_anomaly" and not anomaly_category:
        raise ValueError("setting 'normal_anomaly' requires anomaly_category")

    normal_str = "\n".join(f"- **{c}**" for c in normal_categories)

    if setting == "normal_only":
        categories = normal_str
        anomaly_def = (
            "A text sample is considered an **anomaly** if it does **not** belong to "
            "**any of the categories** listed above."
        )
        step3 = (
            "**Step 3**. Determine which category the text sample **most closely aligns with**.\n"
            "     - If it aligns with any category, it is **not** an anomaly.\n"
            "     - If it does **not** align with any category, it is an anomaly."
        )
    else:
        categories = (
            f"### Normal Category(ies):\n{normal_str}\n"
            f"### Anomaly Category(ies):\n- {anomaly_category}"
        )
        anomaly_def = (
            "A text sample is considered an **anomaly** if it belongs to the "
            "**anomaly category(ies)** rather than **any of the normal category(ies)** "
            "listed above."
        )
        step3 = (
            "**Step 3**. Determine which category the text sample **most closely aligns with**.\n"
            "     - If it **most closely aligns with** **any of the anomaly category(ies)**, "
            "it is an **anomaly**.\n"
            "     - If it **most closely aligns with** **any of the normal category(ies)** "
            "instead, it is **not** an anomaly."
        )

    exemplar_block = f"\n{_format_exemplars(exemplars, setting)}\n" if exemplars else ""
    if exemplars:
        cite_rule = (
            f'   - "citations": The numbers (1 to {len(exemplars)}) of the Reference Examples '
            "that support your decision, as a JSON list such as [1, 3]. Use only numbers that "
            "appear in the Reference Examples; use [] if none were useful. Also mention them "
            'inline in "reason", for example "similar to example [2]".\n'
        )
        keys = 'the keys "reason", "anomaly_score" and "citations."'
    else:
        cite_rule = ""
        keys = 'the keys "reason" and "anomaly_score."'

    return f"""You are an intelligent and professional assistant that detects anomalies in text data.

## Task:
- Following the rules below, determine whether the given text sample is an anomaly. Provide a brief explanation of your reasoning and assign an anomaly confidence score between 0 and 1.

## Categories:
{categories}
{exemplar_block}
## Rules:
1. **Anomaly Definition**:
   - {anomaly_def}

2. **Scoring**:
   - Assign an anomaly confidence score between 0 and 1.
   - Use higher scores when you are highly confident in your decision.
   - Use lower scores when you are uncertain or think the text sample is **not** an anomaly.

3. **Step-by-step Reasoning** (Chain of Thought):
   - **Step 1**. Read the entire text sample carefully and understand it thoroughly.
   - **Step 2**. Analyze the text sample by comparing its content to each category listed in the "Categories" section above, considering factors such as main topics, meanings, background, sentiments, etc.
   - {step3}
   - **Step 4**. Assign an anomaly confidence score based on how confident you are that the text sample is an anomaly.

4. **Additional Notes**:
   - A text sample may relate to multiple categories, but it should be classified into the **most relevant** one in this task.
   - If you are uncertain whether the text sample **significantly aligns** with **any of the anomaly category(ies)**, assume that it does **not**, which means it is **not** an anomaly.

5. **Response Format**:
   - Provide responses in a strict **JSON** format with {keys}
   - "reason": Your brief explanation of the reasoning in one to three sentences logically.
   - "anomaly_score": Your anomaly confidence score between 0 and 1.
{cite_rule}   - Ensure the JSON output is correctly formatted, including correct placement of commas between key-value pairs.
   - Add a backslash (\\) before any double quotation marks (") within the values of JSON output for proper parsing.

Text sample:
"{text}"

Response in JSON format:"""


# Thinking models (e.g. Gemma 4) emit a reasoning block before the answer. Drop it so
# braces or numbers inside it are never mistaken for the JSON answer.
_THINK_RE = re.compile(r"<(thought|think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<(thought|think|thinking)>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    # An unclosed block means the output was cut off mid-thought: no answer present.
    return "" if _THINK_OPEN_RE.search(cleaned) else cleaned


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SCORE_RE = re.compile(r'"?anomaly_score"?\s*[:=]\s*"?([0-9]*\.?[0-9]+)')
_CITATIONS_RE = re.compile(r'"?citations"?\s*[:=]\s*\[([^\]]*)\]')
_REASON_RE = re.compile(r'"reason"\s*:\s*"(.*?)"\s*,\s*"anomaly_score"', re.DOTALL)


def parse_detection_response(text: str) -> tuple[str | None, float | None]:
    """
    Extract (reason, score) from raw LLM output. Returns (None, None) when no
    score can be recovered. The score is clamped to [0, 1].
    """
    candidate = _strip_thinking(text).strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(candidate[start : end + 1])
            score = float(data["anomaly_score"])
            reason = data.get("reason")
            return (str(reason) if reason is not None else None), max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # fall through to regex recovery for slightly malformed JSON

    score_match = _SCORE_RE.search(candidate)
    if not score_match:
        return None, None
    reason_match = _REASON_RE.search(candidate)
    return (
        reason_match.group(1).strip() if reason_match else None,
        max(0.0, min(1.0, float(score_match.group(1)))),
    )


def parse_citations(text: str, n_exemplars: int) -> list[int]:
    """Recover the cited example numbers; keep only valid, unique ones in 1..n_exemplars."""
    if n_exemplars <= 0:
        return []
    raw: list = []
    candidate = _strip_thinking(text).strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(candidate[start : end + 1]).get("citations")
            raw = value if isinstance(value, list) else []
        except (json.JSONDecodeError, AttributeError):
            raw = []
    if not raw:
        match = _CITATIONS_RE.search(candidate)
        raw = re.findall(r"\d+", match.group(1)) if match else []
    valid: list[int] = []
    for item in raw:
        try:
            n = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_exemplars and n not in valid:
            valid.append(n)
    return valid


def _retry_delay_seconds(message: str) -> float:
    match = _RETRY_IN_RE.search(message)
    delay = float(match.group(1) or match.group(2)) + 1.0 if match else 20.0
    return min(delay, RATE_LIMIT_MAX_WAIT_S)


def detect(
    llm: LLMClient,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,  # headroom for thinking models, which reason before answering
    n_exemplars: int = 0,
) -> DetectionResult:
    """Run one detection prompt. LLM/parse failures are captured, not raised."""
    attempt = 0
    while True:
        try:
            response = llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            break
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as result.error
            message = str(exc)
            rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
            if rate_limited and attempt < RATE_LIMIT_RETRIES:
                attempt += 1
                time.sleep(_retry_delay_seconds(message))
                continue
            if rate_limited:
                message = "Rate limit reached (free tier); wait a minute and try again."
            return DetectionResult(prompt=prompt, error=f"LLM call failed: {message}")

    reason, score = parse_detection_response(response.text)
    if score is None:
        return DetectionResult(
            prompt=prompt,
            model=response.model,
            raw_text=response.text,
            error="Could not parse an anomaly_score from the model output",
        )
    return DetectionResult(
        prompt=prompt,
        model=response.model,
        score=score,
        reason=reason,
        raw_text=response.text,
        citations=parse_citations(response.text, n_exemplars),
    )
