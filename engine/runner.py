"""
BenchmarkRunner — điều phối chạy benchmark song song (Async).

Mỗi test case đi qua pipeline:
    1. Agent.query()           -> answer + retrieved_ids + tokens/cost
    2. RetrievalEvaluator      -> Hit Rate, MRR (so với expected_retrieval_ids)
    3. Faithfulness/Relevancy  -> heuristic kiểu RAGAS (overlap answer<->context)
    4. Multi-Judge consensus   -> final_score, agreement, conflict

Hiệu năng: dùng asyncio.Semaphore để chạy song song nhiều case nhưng vẫn giới
hạn số request đồng thời (tránh Rate Limit). Đo latency từng case và tổng thời
gian, đồng thời cộng dồn token & cost để báo cáo Cost-per-Eval.
"""
import asyncio
import re
import time
from typing import List, Dict


def _tokens(text: str):
    return set(re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE))


def _faithfulness(answer: str, contexts: List[str]) -> float:
    """Tỉ lệ token trong answer được 'chống lưng' bởi context (chống bịa)."""
    ans = _tokens(answer)
    if not ans:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= _tokens(c)
    return round(len(ans & ctx) / len(ans), 3)


def _relevancy(answer: str, question: str) -> float:
    """Mức độ câu trả lời liên quan tới câu hỏi (overlap token đơn giản)."""
    q, a = _tokens(question), _tokens(answer)
    if not q:
        return 0.0
    return round(len(q & a) / len(q), 3)


class BenchmarkRunner:
    def __init__(self, agent, judge, retrieval_eval, concurrency: int = 10):
        self.agent = agent
        self.judge = judge
        self.retrieval_eval = retrieval_eval
        self._sem = asyncio.Semaphore(concurrency)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self._sem:
            start = time.perf_counter()
            response = await self.agent.query(test_case["question"])
            latency = time.perf_counter() - start

        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        retrieval = self.retrieval_eval.score_one(expected_ids, retrieved_ids)

        ragas = {
            "faithfulness": _faithfulness(response["answer"], response.get("contexts", [])),
            "relevancy": _relevancy(response["answer"], test_case["question"]),
            "retrieval": retrieval,
        }

        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        return {
            "id": test_case.get("id"),
            "test_case": test_case["question"],
            "type": test_case.get("metadata", {}).get("type"),
            "difficulty": test_case.get("metadata", {}).get("difficulty"),
            "agent_response": response["answer"],
            "expected_retrieval_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "latency": round(latency, 4),
            "tokens_used": response["metadata"].get("tokens_used", 0),
            "cost_usd": response["metadata"].get("cost_usd", 0.0),
            "ragas": ragas,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = None) -> List[Dict]:
        """
        Chạy toàn bộ dataset song song. Semaphore đã giới hạn đồng thời nên có
        thể phát tất cả task một lần (giữ tham số batch_size để tương thích).
        """
        tasks = [self.run_single_test(case) for case in dataset]
        return await asyncio.gather(*tasks)
