#!/usr/bin/env python3
"""Check these source snapshots through Open-MOPD's real scoring wrappers.

Run from the Open-MOPD root after sourcing env.sh:
    PYTHONPATH="$PWD" python /path/to/deps/smoke_test.py
Uses synthetic answers only; these results are not model benchmark scores.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path


def block_network(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.sendto"}:
        if args[0].family in {socket.AF_INET, socket.AF_INET6}:
            raise RuntimeError(f"Network disabled during smoke test: {event}")
    if event == "socket.getaddrinfo":
        raise RuntimeError("DNS disabled during smoke test")


# Runs in multiprocessing children too. Unix sockets used by multiprocessing
# managers remain available; outbound TCP/UDP and DNS are forbidden.
sys.addaudithook(block_network)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def main():
    import pandas as pd
    import nltk
    from evals.verifier.score_functions.code.livecodebench import score_lcb
    from evals.verifier.score_functions.instruction_following.ifbench import score_ifbench

    for resource in (
        "tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords",
        "taggers/averaged_perceptron_tagger_eng",
    ):
        nltk.data.find(resource)
    assert nltk.word_tokenize("Hello, world!") == ["Hello", ",", "world", "!"]
    assert nltk.pos_tag(["Run"])

    # An IFBench-specific constraint, with one passing and one failing answer.
    if_rows = []
    for index, completion in enumerate(("one two three", "one")):
        if_rows.append({
            "prompt": f"Write between 2 and 4 words. Example {index}.",
            "completion": completion,
            "metadata": {
                "key": index,
                "instruction_id_list": ["count:word_count_range"],
                "kwargs": [{"min_words": 2, "max_words": 4}],
            },
        })
    if_result = score_ifbench(pd.DataFrame(if_rows), None)
    assert if_result["per_row_strict"] == [1.0, 0.0], if_result

    # Cover the official 300 prompts' checker paths with a fixed dummy answer.
    # No aggregate accuracy is reported because this is a dependency check.
    fixture = Path(os.environ["OPENOPD_IFBENCH_REPO"]) / "data/IFBench_test.jsonl"
    official_rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    fixture_df = pd.DataFrame([{
        "prompt": row["prompt"], "completion": "Run to the blue house. Take two apples!",
        "metadata": row,
    } for row in official_rows])
    fixture_result = score_ifbench(fixture_df, None)
    assert fixture_result["scored_rows"] == len(official_rows)

    # Two independent LCB problems, each with a correct and an incorrect answer.
    # Testcases reside in metadata; no Hugging Face dataset is loaded.
    lcb_rows = []
    for question_id in ("offline-smoke-1", "offline-smoke-2"):
        metadata = {
            "question_title": "Add two integers", "question_content": "Read two integers and print their sum.",
            "platform": "atcoder", "question_id": question_id, "contest_id": "offline-smoke",
            "contest_date": "2025-01-01T00:00:00", "starter_code": "", "difficulty": "easy",
            "public_test_cases": json.dumps([{"input": "2 3\n", "output": "5\n", "testtype": "stdin"}]),
            "private_test_cases": json.dumps([{"input": "4 8\n", "output": "12\n", "testtype": "stdin"}]),
            "metadata": "{}",
        }
        for completion in (
            "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
            "```python\nprint(0)\n```",
        ):
            lcb_rows.append({"sample_id": question_id, "completion": completion, "metadata": metadata})
    with tempfile.TemporaryDirectory(prefix="open-mopd-offline-smoke-") as tmp:
        lcb_result = score_lcb(pd.DataFrame(lcb_rows), Path(tmp), workers=1,
                               process_workers=2, worker_concurrency=1, timeout=6)
        assert "error" not in lcb_result, lcb_result
        assert lcb_result["scored_rows"] == 2, lcb_result
        assert lcb_result["correct"] == 0.5, lcb_result
        evaluated = json.loads(Path(lcb_result["official_extra"]["eval_path"]).read_text())
        assert [item["graded_list"] for item in evaluated] == [[True, False], [True, False]], evaluated
    print(json.dumps({
        "status": "passed", "network": "TCP/UDP and DNS blocked by Python audit hook",
        "ifbench_positive_negative": [True, False], "ifbench_fixture_rows_executed": len(official_rows),
        "livecodebench_positive_negative": [[True, False], [True, False]],
        "livecodebench_outer_processes": 2, "model_benchmark_evaluation": False,
    }, indent=2))


if __name__ == "__main__":
    main()
