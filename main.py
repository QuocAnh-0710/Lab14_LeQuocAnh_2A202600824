"""
Entry point của AI Evaluation Factory.

Quy trình:
    1. Nạp Golden Dataset (data/golden_set.jsonl).
    2. Benchmark Agent V1 (baseline) và Agent V2 (optimized) — chạy Async.
    3. Tổng hợp metrics: avg_score, Hit Rate, MRR, Agreement Rate, Cohen's Kappa,
       Faithfulness/Relevancy, Cost & Token usage, Latency.
    4. Regression: so sánh V2 vs V1 (Delta).
    5. Release Gate tự động: APPROVE / BLOCK dựa trên ngưỡng Chất lượng + Chi phí
       + Hiệu năng + Độ tin cậy.
    6. Ghi reports/summary.json và reports/benchmark_results.json.

Chạy:  python main.py
"""
import asyncio
import json
import os
import sys
import time
from collections import Counter

# Đảm bảo in được tiếng Việt/emoji trên console Windows (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge, cohens_kappa
from agent.main_agent import MainAgent

# Ngưỡng cho Release Gate (có thể chỉnh theo SLA sản phẩm)
GATE = {
    "min_avg_score": 3.5,        # điểm Judge trung bình tối thiểu
    "min_hit_rate": 0.80,        # Retrieval phải đủ tốt
    "min_agreement_rate": 0.70,  # Judge phải đủ đồng thuận để tin kết quả
    "max_avg_latency": 2.0,      # giây/case
    "max_cost_regression": 0.10, # cho phép cost tăng tối đa 10% so với V1
}


def aggregate(results, version: str) -> dict:
    """Tổng hợp một lượt benchmark thành summary có cấu trúc."""
    total = len(results)

    # Retrieval chỉ tính trên case applicable (có expected_retrieval_ids)
    retr = [r["ragas"]["retrieval"] for r in results if r["ragas"]["retrieval"]["applicable"]]
    n_retr = len(retr) or 1

    # Cohen's Kappa giữa 2 Judge trên toàn dataset
    judge_models = list(results[0]["judge"]["individual_scores"].keys())[:2]
    ratings_a = [r["judge"]["individual_scores"][judge_models[0]] for r in results]
    ratings_b = [r["judge"]["individual_scores"][judge_models[1]] for r in results]

    total_cost = sum(r["cost_usd"] for r in results) + \
        sum(r["judge"]["judge_cost_usd"] for r in results)
    total_tokens = sum(r["tokens_used"] for r in results)

    passed = sum(1 for r in results if r["status"] == "pass")

    return {
        "metadata": {
            "version": version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judges": judge_models,
        },
        "metrics": {
            "avg_score": round(sum(r["judge"]["final_score"] for r in results) / total, 3),
            "pass_rate": round(passed / total, 3),
            "hit_rate": round(sum(r["hit_rate"] for r in retr) / n_retr, 3),
            "mrr": round(sum(r["mrr"] for r in retr) / n_retr, 3),
            "agreement_rate": round(sum(r["judge"]["agreement"] for r in results) / total, 3),
            "cohens_kappa": round(cohens_kappa(ratings_a, ratings_b), 3),
            "avg_faithfulness": round(sum(r["ragas"]["faithfulness"] for r in results) / total, 3),
            "avg_relevancy": round(sum(r["ragas"]["relevancy"] for r in results) / total, 3),
            "avg_latency": round(sum(r["latency"] for r in results) / total, 4),
            "conflicts_resolved": sum(1 for r in results if r["judge"]["conflict"]),
        },
        "cost": {
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "cost_per_eval_usd": round(total_cost / total, 6),
        },
    }


async def run_benchmark(agent_key: str, label: str, dataset, concurrency: int = 10):
    """agent_key ('v1'|'v2') chọn hành vi Agent; label là tên hiển thị trong report."""
    agent = MainAgent(version=agent_key)
    judge = LLMJudge()
    retrieval_eval = RetrievalEvaluator(top_k=agent.top_k)
    runner = BenchmarkRunner(agent, judge, retrieval_eval, concurrency=concurrency)

    t0 = time.perf_counter()
    results = await runner.run_all(dataset)
    wall = time.perf_counter() - t0

    summary = aggregate(results, label)
    summary["metadata"]["wall_time_sec"] = round(wall, 2)
    return results, summary


def failure_clusters(results) -> dict:
    """Phân cụm các case fail theo loại lỗi (hỗ trợ Failure Analysis)."""
    clusters = Counter()
    examples = {}
    for r in results:
        if r["status"] != "pass":
            faithful = r["ragas"]["faithfulness"]
            hit = r["ragas"]["retrieval"].get("hit_rate")
            if hit == 0.0:
                label = "Retrieval Miss (lấy sai context)"
            elif faithful < 0.3:
                label = "Hallucination (bịa, không bám context)"
            elif r["type"] in ("prompt-injection", "goal-hijacking"):
                label = "Safety / Injection failure"
            else:
                label = "Answer Quality (thiếu/sai nội dung)"
            clusters[label] += 1
            examples.setdefault(label, r["test_case"])
    return {"counts": dict(clusters), "examples": examples}


def evaluate_gate(v1: dict, v2: dict) -> dict:
    """Release Gate tự động: APPROVE nếu vượt mọi ngưỡng và không hồi quy."""
    m1, m2 = v1["metrics"], v2["metrics"]
    delta_score = round(m2["avg_score"] - m1["avg_score"], 3)
    cost_change = (v2["cost"]["cost_per_eval_usd"] - v1["cost"]["cost_per_eval_usd"]) / \
        max(v1["cost"]["cost_per_eval_usd"], 1e-9)

    checks = {
        "no_quality_regression": delta_score >= 0,
        "avg_score_above_min": m2["avg_score"] >= GATE["min_avg_score"],
        "hit_rate_above_min": m2["hit_rate"] >= GATE["min_hit_rate"],
        "agreement_above_min": m2["agreement_rate"] >= GATE["min_agreement_rate"],
        "latency_ok": m2["avg_latency"] <= GATE["max_avg_latency"],
        "cost_not_regressed": cost_change <= GATE["max_cost_regression"],
    }
    decision = "APPROVE" if all(checks.values()) else "BLOCK"
    return {
        "decision": decision,
        "delta_avg_score": delta_score,
        "cost_change_pct": round(cost_change * 100, 2),
        "checks": checks,
        "thresholds": GATE,
    }


async def main():
    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Chạy 'python data/synthetic_gen.py' trước.")
        return

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    if not dataset:
        print("❌ data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return

    print(f"🚀 Benchmark trên {len(dataset)} cases (Async)...\n")

    v1_results, v1_summary = await run_benchmark("v1", "Agent_V1_Base", dataset)
    print(f"   V1 xong trong {v1_summary['metadata']['wall_time_sec']}s")
    v2_results, v2_summary = await run_benchmark("v2", "Agent_V2_Optimized", dataset)
    print(f"   V2 xong trong {v2_summary['metadata']['wall_time_sec']}s\n")

    gate = evaluate_gate(v1_summary, v2_summary)
    clusters = failure_clusters(v2_results)

    # ----- In kết quả ------------------------------------------------------
    print("📊 --- SO SÁNH REGRESSION (V1 -> V2) ---")
    for k in ("avg_score", "hit_rate", "mrr", "agreement_rate", "cohens_kappa",
              "avg_faithfulness", "pass_rate", "avg_latency"):
        print(f"   {k:18s}: {v1_summary['metrics'][k]:>7} -> {v2_summary['metrics'][k]:>7}")
    print(f"   {'cost_per_eval':18s}: {v1_summary['cost']['cost_per_eval_usd']:>7} -> "
          f"{v2_summary['cost']['cost_per_eval_usd']:>7}")
    print(f"\n   Delta avg_score: {'+' if gate['delta_avg_score'] >= 0 else ''}"
          f"{gate['delta_avg_score']}")

    # ----- Ghi reports -----------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    final_summary = dict(v2_summary)
    final_summary["regression"] = {
        "baseline": v1_summary["metadata"]["version"],
        "candidate": v2_summary["metadata"]["version"],
        "v1_metrics": v1_summary["metrics"],
        "v2_metrics": v2_summary["metrics"],
        "gate": gate,
    }
    final_summary["failure_clusters"] = clusters

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results_v1.json", "w", encoding="utf-8") as f:
        json.dump(v1_results, f, ensure_ascii=False, indent=2)

    print(f"\n🧩 Failure clusters (V2): {clusters['counts']}")
    print(f"\n🚦 RELEASE GATE: {gate['decision']}")
    for name, ok in gate["checks"].items():
        print(f"   {'✅' if ok else '❌'} {name}")
    if gate["decision"] == "APPROVE":
        print("\n✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE / RELEASE)")
    else:
        print("\n❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE / ROLLBACK)")
    print("\n💾 Đã ghi reports/summary.json & reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
