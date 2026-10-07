"""
RAG Evaluation Script (Legal / HR Edition)
Checks retrieval accuracy against a labeled test set (eval_set.csv).
"""

import argparse
import csv
import os
import sys

from rag_chat import retrieve_chunks


def run_eval(csv_path: str, db_dir: str, collection_name: str, top_k: int = 3):
    if not os.path.isfile(csv_path):
        print(f"ERROR: Evaluation file '{csv_path}' not found.")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in reader.fieldnames or "expected_source" not in reader.fieldnames:
            print("ERROR: CSV must have columns: question, expected_source")
            sys.exit(1)
        rows = list(reader)

    if not rows:
        print("ERROR: Evaluation file is empty.")
        sys.exit(1)

    print(f"Running evaluation on {len(rows)} test questions against '{collection_name}'...\\n")

    passed = 0
    for idx, row in enumerate(rows, 1):
        q = row["question"].strip()
        expected = row["expected_source"].strip()

        try:
            chunks = retrieve_chunks(q, db_dir=db_dir, collection_name=collection_name, top_k=top_k)
            retrieved_sources = [c["source"] for c in chunks]
            top_1 = retrieved_sources[0] if retrieved_sources else "NONE"

            # Check if expected source is at top-1 or in top-k
            is_top_1 = (top_1 == expected)
            is_in_top_k = (expected in retrieved_sources)

            if is_top_1:
                passed += 1
                status = "PASS (Top-1)"
            elif is_in_top_k:
                passed += 1
                status = f"PASS (Top-{top_k})"
            else:
                status = "FAIL"

            print(f"[{idx}/{len(rows)}] {status}")
            print(f"  Question: {q}")
            print(f"  Expected: {expected}")
            print(f"  Retrieved: {top_1} (Section: {chunks[0].get('section', '?') if chunks else 'None'})")
            print()
        except Exception as e:
            print(f"[{idx}/{len(rows)}] ERROR: {e}")

    score = (passed / len(rows)) * 100
    print(f"=== Summary: {passed}/{len(rows)} passed ({score:.1f}% accuracy) ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="eval_set.csv", help="Path to evaluation CSV")
    parser.add_argument("--db_dir", default="./chroma_db", help="Chroma DB folder")
    parser.add_argument("--collection_name", default="nexora", help="Chroma collection name")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k search window")
    args = parser.parse_args()

    run_eval(args.csv, args.db_dir, args.collection_name, args.top_k)


if __name__ == "__main__":
    main()
