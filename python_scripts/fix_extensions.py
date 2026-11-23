"""
Fix malformed JSON in collected keyword expansion files.

Usage:
    python fix_extensions.py
"""

import json
import re
from glob import glob


def try_parse(text):
    """Try to parse text as JSON with progressive repairs."""
    # Strip preamble
    idx = text.find('{')
    if idx > 0:
        text = text[idx:]

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 1b: maybe double-encoded (string within string)
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            return json.loads(decoded)
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: fix trailing commas (e.g. "item",\n] )
    fixed = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: fix VERBS: {"word", "word"} -> VERBS: ["word", "word"]
    # Detect VERBS block that has strings but no colons (set-like, not dict-like)
    verbs_match = re.search(r'"VERBS"\s*:\s*\{([^}]+)\}', fixed)
    if verbs_match:
        verbs_content = verbs_match.group(1)
        # If no ":" in the verbs content, it's a set-like list
        if ':' not in verbs_content:
            fixed3 = fixed[:verbs_match.start()] + '"VERBS": [' + verbs_content + ']' + fixed[verbs_match.end():]
            try:
                return json.loads(fixed3)
            except json.JSONDecodeError:
                pass

    # Attempt 4: fix VERBS bracket mismatches
    # Case A: "VERBS": [...] with key-value pairs -> "VERBS": {...}
    fixed4 = re.sub(
        r'"VERBS"\s*:\s*\[(\s*")',
        r'"VERBS": {\1',
        fixed,
    )
    fixed4 = re.sub(r'\]\s*\]\s*\}\s*$', ']\n  }\n}', fixed4.rstrip())
    try:
        return json.loads(fixed4)
    except json.JSONDecodeError:
        pass

    # Case B: VERBS has swapped brackets: [...} ] } instead of [...] } }
    fixed5 = re.sub(r'\}\s*\]\s*\}\s*$', ']\n  }\n}', fixed.rstrip())
    try:
        return json.loads(fixed5)
    except json.JSONDecodeError:
        pass

    # Case C: VERBS inner list closed with } instead of ]: "verb": [...}
    # Find [ that isn't closed by ] before the next }
    fixed6 = fixed
    # Replace last occurrence of } } } with ] } } (fix innermost)
    fixed6 = re.sub(r'\}\s*\}\s*\}\s*$', ']\n  }\n}', fixed6.rstrip())
    try:
        return json.loads(fixed6)
    except json.JSONDecodeError:
        pass

    # Attempt 4: truncate at last valid }
    depth = 0
    last_valid = -1
    for i, c in enumerate(fixed):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_valid = i
                break

    if last_valid > 0:
        try:
            return json.loads(fixed[:last_valid + 1])
        except json.JSONDecodeError:
            pass

    # Attempt 5: truncated JSON — find deepest valid point and close brackets
    # Count unclosed brackets and try to close them
    depth_brace = 0
    depth_bracket = 0
    last_complete = -1

    for i, c in enumerate(fixed):
        if c == '{':
            depth_brace += 1
        elif c == '}':
            depth_brace -= 1
        elif c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket -= 1

        # A complete value ends after ] or } or a quoted string followed by comma/bracket
        if c in (']', '}') and depth_brace >= 0 and depth_bracket >= 0:
            last_complete = i

    if last_complete > 0:
        truncated = fixed[:last_complete + 1]
        # Close any remaining open brackets
        closing = ']' * depth_bracket + '}' * depth_brace
        try:
            return json.loads(truncated + closing)
        except json.JSONDecodeError:
            pass

    return None


def fix_file(path, verbose=True, debug=False):
    """Fix malformed entries in a single JSONL file. Returns (total, fixed, still_broken)."""
    with open(path) as f:
        lines = f.readlines()

    total = len(lines)
    fixed = 0
    still_broken = 0
    output_lines = []

    for line in lines:
        obj = json.loads(line.strip())

        if "raw" in obj:
            qid = obj["qid"]
            parsed = try_parse(obj["raw"])
            if parsed is not None:
                parsed["qid"] = qid
                output_lines.append(json.dumps(parsed) + "\n")
                fixed += 1
            else:
                output_lines.append(line)
                still_broken += 1
                if debug:
                    print(f"\n  BROKEN qid={qid}:")
                    raw = obj["raw"]
                    print(f"    Length: {len(raw)}")
                    print(f"    First 200 chars: {raw[:200]}")
                    print(f"    Last 100 chars: {raw[-100:]}")
                    # Try to get the actual parse error
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError as e:
                        print(f"    JSON error: {e}")
        else:
            output_lines.append(line)

    with open(path, "w") as f:
        f.writelines(output_lines)

    if verbose:
        print(f"  {path}: {total} entries, {fixed} fixed, {still_broken} still broken")

    return total, fixed, still_broken


def fix_all(beir_dir="../beir", pattern="*_extensions.jsonl", verbose=True):
    """Fix all extension files in BEIR directories."""
    files = sorted(glob(f"{beir_dir}/**/{pattern}", recursive=True))

    if verbose:
        print(f"Found {len(files)} files")

    totals = {"total": 0, "fixed": 0, "broken": 0}

    for path in files:
        total, fixed, broken = fix_file(path, verbose=verbose)
        totals["total"] += total
        totals["fixed"] += fixed
        totals["broken"] += broken

    if verbose:
        print(f"\nSummary: {totals['total']} entries, "
              f"{totals['fixed']} fixed, {totals['broken']} still broken")

    return totals


if __name__ == "__main__":
    fix_all()