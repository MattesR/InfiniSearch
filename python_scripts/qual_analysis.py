"""
Qualitative analysis: cross-encoder ranking of API experiment results.

Usage:
    from qual_analysis import rank_all_results

    rank_all_results("./api_expansions_results.pkl", output_dir="./qual_analysis/")
"""

import os
import csv
import pickle
import re
from sentence_transformers import CrossEncoder


def rank_single(
    query: str,
    documents: list,
    model=None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k: int = None,
    batch_size: int = 32,
    extra_texts: list = None,
):
    """
    Rank documents by cross-encoder score for a single query.

    Args:
        extra_texts: Optional list of strings to include in ranking.

    Returns list of (score, doc_id, text, from_queries) sorted by score descending.
    """
    if model is None:
        model = CrossEncoder(model_name)

    if not documents and not extra_texts:
        return [], model

    # Deduplicate by text content, merge from_queries
    seen_texts = {}
    for d in (documents or []):
        text = d.get("text", "").strip()
        if not text:
            continue
        if text in seen_texts:
            existing = seen_texts[text]
            existing_fq = set(existing.get("from_queries", []))
            new_fq = set(d.get("from_queries", []))
            existing["from_queries"] = list(existing_fq | new_fq)
        else:
            seen_texts[text] = dict(d)

    # Add extra texts
    if extra_texts:
        for i, text in enumerate(extra_texts):
            text = text.strip()
            if text and text not in seen_texts:
                seen_texts[text] = {
                    "doc_id": "",
                    "text": text,
                    "from_queries": [],
                }

    unique_docs = list(seen_texts.values())

    doc_ids = [d.get("doc_id", "") for d in unique_docs]
    texts = [d.get("text", "") for d in unique_docs]
    from_queries = ["; ".join(d.get("from_queries", [])) for d in unique_docs]

    # Score all pairs
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    # Sort by score descending
    ranked = sorted(zip(scores, doc_ids, texts, from_queries), key=lambda x: x[0], reverse=True)

    if top_k:
        ranked = ranked[:top_k]

    return ranked, model


def rank_and_save(
    query: str,
    documents: list,
    output_dir: str = "./qual_analysis/",
    model=None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k: int = 50,
    max_text_len: int = 500,
    extra_texts: list = None,
):
    """
    Rank documents and save to CSV.

    Args:
        extra_texts: Optional list of strings to include in ranking.

    Returns (ranked, model).
    """
    os.makedirs(output_dir, exist_ok=True)

    ranked, model = rank_single(query, documents, model=model,
                                 model_name=model_name, top_k=top_k,
                                 extra_texts=extra_texts)

    if not ranked:
        print(f"  No documents to rank for: {query[:50]}")
        return ranked, model

    # Build filename from query
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '_', query)[:80].strip('_')
    csv_path = os.path.join(output_dir, f"{safe_name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", "query", "doc_id", "from_queries", "text"])
        for rank, (score, doc_id, text, fq) in enumerate(ranked, 1):
            text_clean = text[:max_text_len].replace("\n", " ").replace("\r", " ")
            writer.writerow([rank, f"{score:.4f}", query, doc_id, fq, text_clean])

    print(f"  Saved {len(ranked)} ranked docs to {csv_path}")
    return ranked, model


def rank_all_results(
    results_path: str = "./api_expansions_results.pkl",
    output_dir: str = "./qual_analysis/",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k: int = 50,
    max_text_len: int = 2500,
    extra_texts=None
):
    """
    Rank all experiments in a results pickle and save CSVs.
    """
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} experiments from {results_path}")

    model = CrossEncoder(model_name)
    print(f"Loaded model: {model_name}")

    all_ranked = {}

    for i, r in enumerate(results):
        query = r.get("query", "")
        llm_model = r.get("model", "unknown")
        documents = r.get("documents", [])

        print(f"\n[{i+1}/{len(results)}] {llm_model}: {query[:60]}... ({len(documents)} docs)")

        if not documents:
            print("  No documents, skipping")
            continue

        ranked, model = rank_and_save(
            query=query,
            documents=documents,
            output_dir=output_dir,
            model=model,
            top_k=top_k,
            max_text_len=max_text_len,
            extra_texts=extra_texts
        )
        all_ranked[query] = ranked

    print(f"\nDone. {len(all_ranked)} CSVs saved to {output_dir}")
    return all_ranked