import json
from collections import defaultdict


def load_all_expansions(expansions_path: str) -> dict:
    """
    Load keyword expansions from JSON or JSONL file.

    JSON: {"qid": {...}, ...}
    JSONL: one {"qid": "...", ...} per line

    Returns:
        Dict mapping qid -> keyword data dict.
    """
    if expansions_path.endswith(".jsonl"):
        expansions = {}
        with open(expansions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                qid = entry.pop("qid")
                expansions[qid] = entry
        return expansions
    else:
        with open(expansions_path) as f:
            return json.load(f)
        

def load_topics(path: str) -> list[tuple[str, str]]:
    """
    Load topics file. Supports:
    - TSV: query_id\\tquery_text
    - JSONL: {"id": "...", "title": "..."}

    Returns:
        List of (query_id, query_text) tuples.
    """
    topics = []
    with open(path) as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith("{"):
            # JSONL format
            for line in f:
                obj = json.loads(line.strip())
                qid = str(obj.get("id", obj.get("qid", "")))
                text = obj.get("title", obj.get("text", obj.get("query", "")))
                if qid and text:
                    topics.append((qid, text))
        else:
            # TSV format
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    topics.append((parts[0], parts[1]))

    return topics


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """
    Load qrels in TREC format.
    Handles both 3-column (qid docid rel) and 4-column (qid iter docid rel).

    Returns:
        {query_id: {doc_id: relevance}}
    """
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                continue
            qrels[qid][docid] = int(rel)
    return dict(qrels)


def get_token(key: str, filename: str = "token_file.ini") -> str | None:
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip("'\"")
    except FileNotFoundError:
        print(f"Token file '{filename}' not found.")
    return None
