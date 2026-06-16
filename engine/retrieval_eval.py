"""
Retrieval Evaluation — đo chất lượng tầng Retrieval của pipeline RAG.

Hai chỉ số cốt lõi:
    - Hit Rate@k: tỉ lệ case mà ÍT NHẤT một document đúng nằm trong top-k.
                  Trả lời câu hỏi "Retriever có lấy được tài liệu đúng không?".
    - MRR (Mean Reciprocal Rank): trung bình của 1/(vị trí đầu tiên trúng).
                  Trả lời câu hỏi "Tài liệu đúng có được xếp HẠNG CAO không?".

Vì sao quan trọng: nếu Retrieval đã sai (Hit Rate thấp) thì Generation chắc chắn
sẽ Hallucinate — bạn cần tách bạch lỗi Retrieval khỏi lỗi Generation.
"""
from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str],
                           top_k: int = None) -> float:
        """1.0 nếu có ít nhất 1 expected_id nằm trong top_k retrieved_ids."""
        k = top_k if top_k is not None else self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Reciprocal Rank của vị trí trúng đầu tiên (1-indexed); 0 nếu không trúng."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score_one(self, expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        """
        Chấm retrieval cho một case.

        Quy ước cho case 'out-of-context' (expected_ids rỗng): câu hỏi không có
        tài liệu đúng nào trong KB, nên Retrieval không áp dụng -> đánh dấu
        applicable=False và không tính vào trung bình Hit Rate/MRR.
        """
        if not expected_ids:
            return {"hit_rate": None, "mrr": None, "applicable": False}
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "applicable": True,
        }

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval retrieval cho cả bộ dữ liệu.

        Mỗi phần tử cần có 'expected_retrieval_ids' và 'retrieved_ids'
        (do Agent trả về). Chỉ tính trung bình trên các case applicable.
        """
        hits, mrrs = [], []
        for case in dataset:
            r = self.score_one(
                case.get("expected_retrieval_ids", []),
                case.get("retrieved_ids", []),
            )
            if r["applicable"]:
                hits.append(r["hit_rate"])
                mrrs.append(r["mrr"])
        n = len(hits) or 1
        return {
            "avg_hit_rate": sum(hits) / n,
            "avg_mrr": sum(mrrs) / n,
            "evaluated_cases": len(hits),
        }
