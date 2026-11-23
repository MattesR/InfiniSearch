"""
Local LLM keyword expansion using OpenRouter (OpenAI-compatible API).

Usage:
    # Run a single model (async, all requests fired at once)
    python openrouter_keywords.py run --topics topics_rag24_test.txt \
        --model google/gemma-4-31b-it:free

    # Throttle to respect rate limits (e.g. 10 concurrent for free tier)
    python openrouter_keywords.py run --topics topics_rag24_test.txt \
        --model google/gemma-4-31b-it:free --workers 3

    # Compare multiple models
    python openrouter_keywords.py compare --topics topics_rag24_test.txt \
        --models google/gemma-4-31b-it:free,meta-llama/llama-3.3-70b-instruct \
        --output-dir ./openrouter_results/

    # List popular models (curated shortlist)
    python openrouter_keywords.py list-models

Requires:
    pip install openai
    OPENROUTER_API_KEY in token_file.ini or environment
"""

import argparse
import asyncio
import json
import re
import os
import sys
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    print("Install the openai package: pip install openai")
    sys.exit(1)

try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

from batch_keywords import SYSTEM_PROMPT, load_topics
from utils import get_token
from fix_extensions import fix_file

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Curated shortlist for keyword expansion benchmarking
SUGGESTED_MODELS = [
    ("meta-llama/llama-3.3-70b-instruct",       "Meta  — strong instruction following"),
    ("meta-llama/llama-3.2-3b-instruct",         "Meta  — fast/cheap small model"),
    ("mistralai/mistral-7b-instruct",             "Mistral — efficient 7B"),
    ("mistralai/mixtral-8x7b-instruct",           "Mistral — MoE, good quality"),
    ("google/gemma-3-27b-it",                     "Google — 27B, strong on structured output"),
    ("google/gemma-3-4b-it",                      "Google — 4B, fast"),
    ("qwen/qwen-2.5-72b-instruct",                "Qwen  — 72B, excellent JSON"),
    ("qwen/qwen-2.5-7b-instruct",                 "Qwen  — 7B, fast/cheap"),
    ("anthropic/claude-sonnet-4-5",               "Anthropic — Claude for comparison"),
    ("openai/gpt-4o-mini",                        "OpenAI  — fast/cheap baseline"),
]


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        return get_token("OPENROUTER_API_KEY")
    except Exception:
        print("OPENROUTER_API_KEY not found in environment or token_file.ini")
        sys.exit(1)


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_get_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def _extract_last_json(text: str) -> str | None:
    """Extract the last complete top-level JSON object by scanning backwards."""
    last_close = text.rfind('}')
    if last_close == -1:
        return None
    depth = 0
    for i in range(last_close, -1, -1):
        if text[i] == '}':
            depth += 1
        elif text[i] == '{':
            depth -= 1
            if depth == 0:
                return text[i:last_close + 1]
    return None


def _strip_fences(text: str) -> str:
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    # Extract last complete JSON block to skip any thinking/preamble
    extracted = _extract_last_json(text)
    return extracted if extracted is not None else text


async def _expand_one(client: AsyncOpenAI, qid: str, query: str, model: str,
                      semaphore: asyncio.Semaphore,
                      provider_order: list[str] | None = None,
                      max_retries: int = 8) -> tuple[str, dict | None, str | None]:
    """
    Run keyword expansion for one query, retrying on 429s with exponential backoff.
    Returns (qid, parsed_dict_or_None, error_str_or_None).
    The semaphore limits how many requests are in-flight at once.
    """
    async with semaphore:
        wait = 5  # initial backoff in seconds
        for attempt in range(max_retries):
            try:
                extra: dict = {}
                if provider_order:
                    extra["provider"] = {
                        "order": provider_order,
                        "allow_fallbacks": True,
                    }

                # Disable thinking mode for reasoning models (e.g. Qwen3)
                # to avoid empty content with reasoning tokens in a separate field
                extra["thinking"] = {"type": "disabled"}

                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=2048,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    extra_body=extra or None,
                )
                msg = response.choices[0].message
                # Fallback to reasoning field if content is empty (thinking models)
                text = msg.content or getattr(msg, "reasoning", "") or ""
                text = _strip_fences(text)

                if not text:
                    finish_reason = response.choices[0].finish_reason
                    print(f"\n  [{qid}] empty response — finish_reason={finish_reason!r}, "
                          f"full message: {msg}")
                    return qid, None, f"empty_response finish_reason={finish_reason}"

                try:
                    parsed = json.loads(text)
                    return qid, parsed, None
                except json.JSONDecodeError:
                    return qid, {"raw": text}, "json_parse_error"

            except Exception as e:
                is_429 = "429" in str(e)
                if is_429 and attempt < max_retries - 1:
                    print(f"\n  [{qid}] rate-limited, retrying in {wait}s "
                          f"(attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 120)  # cap at 2 minutes
                else:
                    return qid, None, str(e)

        return qid, None, f"gave up after {max_retries} retries"


def _load_done(output_path: str) -> set:
    """Return set of qids already written to output_path (for resume support)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "qid" in obj:
                        done.add(obj["qid"])
                except json.JSONDecodeError:
                    pass
    return done


async def _run_async(topics_path: str, model: str, output_path: str, workers: int,
                     provider_order: list[str] | None = None):
    """Fire all requests concurrently (bounded by workers), write results as they arrive."""
    client = _make_client()
    topics = load_topics(topics_path)
    done = _load_done(output_path)

    pending = [(qid, query) for qid, query in topics if qid not in done]
    if done:
        print(f"Resuming: {len(done)} already done, {len(pending)} remaining.")
    provider_str = f" via {' > '.join(provider_order)}" if provider_order else ""
    print(f"Sending {len(pending)} requests (max {workers} in-flight){provider_str}...")

    semaphore = asyncio.Semaphore(workers)
    tasks = [_expand_one(client, qid, query, model, semaphore, provider_order)
             for qid, query in pending]

    errors = []
    with open(output_path, "a") as out_f:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model.split("/")[-1]):
            qid, result, err = await coro
            if result is not None:
                out_f.write(json.dumps({"qid": qid, **result}) + "\n")
                out_f.flush()
            if err:
                errors.append((qid, err))

    total = len(topics)
    succeeded = total - len(errors)
    print(f"\nDone. {succeeded}/{total} succeeded, {len(errors)} errors.")
    if errors:
        err_path = output_path.replace(".jsonl", "_errors.json")
        with open(err_path, "w") as f:
            json.dump([{"qid": q, "error": e} for q, e in errors], f, indent=2)
        print(f"Errors saved to {err_path}")
    print(f"Results saved to {output_path}")

    # Auto-repair malformed JSON entries
    print("Running JSON repair pass...")
    fix_file(output_path)


def run(topics_path: str, model: str, output_path: str, workers: int = 4,
        provider_order: list[str] | None = None):
    """Run keyword expansion for all topics with one model, writing to a JSONL file."""
    asyncio.run(_run_async(topics_path, model, output_path, workers, provider_order))


def compare(topics_path: str, models: list[str], output_dir: str, workers: int = 4,
            provider_order: list[str] | None = None):
    """Run all models and save each to a separate file in output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for model in models:
        safe_name = model.replace(":", "_").replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_name}.jsonl")
        print(f"\n=== Model: {model} -> {output_path} ===")
        run(topics_path, model, output_path, workers=workers, provider_order=provider_order)


def list_models():
    """Print a curated list of suggested models for keyword expansion."""
    print("Suggested models for keyword expansion benchmarking:\n")
    print(f"  {'Model':<45} Notes")
    print(f"  {'-'*44} {'-'*35}")
    for model, notes in SUGGESTED_MODELS:
        print(f"  {model:<45} {notes}")
    print("\nFull model catalogue: https://openrouter.ai/models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword expansion via OpenRouter models")
    subparsers = parser.add_subparsers(dest="command")

    sub = subparsers.add_parser("run", help="Run one model")
    sub.add_argument("--topics", required=True)
    sub.add_argument("--model", required=True, help="OpenRouter model ID, e.g. meta-llama/llama-3.3-70b-instruct")
    sub.add_argument("--output", default=None,
                     help="Output JSONL path (default: keyword_files/<model>.jsonl)")
    sub.add_argument("--workers", type=int, default=4,
                     help="Max concurrent requests (default: 4; use 2-3 for :free models)")
    sub.add_argument("--provider", default=None,
                     help="Comma-separated provider preference order, e.g. 'DeepInfra,Together'")

    sub = subparsers.add_parser("compare", help="Run multiple models and compare")
    sub.add_argument("--topics", required=True)
    sub.add_argument("--models", required=True,
                     help="Comma-separated OpenRouter model IDs")
    sub.add_argument("--output-dir", default="keyword_files")
    sub.add_argument("--workers", type=int, default=4,
                     help="Max concurrent requests (default: 4; use 2-3 for :free models)")
    sub.add_argument("--provider", default=None,
                     help="Comma-separated provider preference order, e.g. 'DeepInfra,Together'")

    subparsers.add_parser("list-models", help="Print suggested models for benchmarking")

    args = parser.parse_args()

    if args.command == "run":
        safe_name = args.model.replace(":", "_").replace("/", "_")
        output = args.output or f"keyword_files/{safe_name}.jsonl"
        Path("keyword_files").mkdir(exist_ok=True)
        providers = [p.strip() for p in args.provider.split(",")] if args.provider else None
        run(args.topics, args.model, output, workers=args.workers, provider_order=providers)

    elif args.command == "compare":
        models = [m.strip() for m in args.models.split(",")]
        providers = [p.strip() for p in args.provider.split(",")] if args.provider else None
        compare(args.topics, models, args.output_dir, workers=args.workers, provider_order=providers)

    elif args.command == "list-models":
        list_models()

    else:
        parser.print_help()
