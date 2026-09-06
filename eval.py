"""
eval.py

A lightweight retrieval-accuracy check for the RAG pipeline.

Usage:
    python eval.py --pdf sample.pdf --testset testset.json

testset.json format:
[
    {"question": "What is the model's context window?", "expect_contains": "context window"},
    {"question": "What dataset was used for training?", "expect_contains": "dataset"}
]

For each question, retrieves the top chunk(s) and checks whether any
retrieved chunk contains the expected substring (case-insensitive).
This is a simple proxy for "did retrieval find the right passage" —
not a substitute for human-graded answer quality, but useful as a
fast regression check when you change chunking, embeddings, or k.
"""

import argparse
import json
import sys
import time

import rag_core


def run_eval(pdf_path, testset_path, k=3, score_threshold=0.0):
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    print(f"Building index for {pdf_path} ...")
    doc = rag_core.build_document(file_bytes, pdf_path)
    print(f"  {len(doc['chunks'])} chunks indexed.")

    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    hits = 0
    total_latency = 0.0

    print(f"\nRunning {len(testset)} test questions...\n")

    for i, case in enumerate(testset, start=1):
        question = case["question"]
        expected = case["expect_contains"].lower()

        start = time.time()
        retrieved = rag_core.retrieve(
            question, [doc], k=k, score_threshold=score_threshold
        )
        latency = time.time() - start
        total_latency += latency

        found = any(expected in r["text"].lower() for r in retrieved)
        hits += int(found)

        status = "PASS" if found else "FAIL"
        top_score = f"{retrieved[0]['score']:.2f}" if retrieved else "n/a"
        print(f"[{status}] ({i}/{len(testset)}) top_score={top_score} latency={latency:.2f}s :: {question}")

    accuracy = hits / len(testset) if testset else 0.0
    avg_latency = total_latency / len(testset) if testset else 0.0

    print("\n--- Summary ---")
    print(f"Retrieval hit rate : {accuracy * 100:.1f}% ({hits}/{len(testset)})")
    print(f"Avg retrieval time : {avg_latency:.3f}s")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval accuracy.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF/DOCX/TXT file to index")
    parser.add_argument("--testset", required=True, help="Path to a JSON test set")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    accuracy = run_eval(args.pdf, args.testset, k=args.k, score_threshold=args.threshold)
    sys.exit(0 if accuracy >= 0.5 else 1)