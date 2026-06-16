"""
MainAgent — Agent RAG mẫu có *phiên bản* (V1 vs V2) để phục vụ Regression Test.

Agent mô phỏng đầy đủ pipeline RAG:
    1. Retrieval: tìm document liên quan trong Knowledge Base bằng keyword overlap.
    2. Generation: tổng hợp câu trả lời từ context truy xuất được.

Hai phiên bản dùng để chứng minh Regression Gate:
    - "v1": retriever cơ bản (chỉ overlap thân tài liệu, top_k=3, KHÔNG biết từ chối
            khi out-of-context -> dễ Hallucination).
    - "v2": retriever cải tiến (boost theo tiêu đề, lọc stopword, có ngưỡng tin cậy
            để TỪ CHỐI khi không tìm thấy tài liệu phù hợp -> ít Hallucination hơn).

Toàn bộ deterministic (không gọi API) nên benchmark chạy nhanh, lặp lại được và
không tốn chi phí. Có thể thay bằng Agent thật bằng cách giữ nguyên interface
`async def query(question) -> dict`.
"""
import asyncio
import re
from typing import Dict, List

try:
    from data.knowledge_base import DOCUMENTS, DOC_BY_ID
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.knowledge_base import DOCUMENTS, DOC_BY_ID

_STOPWORDS = {
    "là", "và", "của", "có", "không", "cho", "tôi", "bạn", "được", "thì",
    "trong", "với", "một", "các", "này", "đó", "khi", "nếu", "để", "về",
    "bao", "nhiêu", "gì", "nào", "thế", "ra", "sao", "vậy", "mình", "cần",
    "hỏi", "biết", "làm", "hãy", "cho_tôi", "ở", "đi", "sẽ", "mà",
}


def _tokenize(text: str, drop_stopwords: bool = False) -> List[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


class MainAgent:
    """Agent RAG mẫu, có phiên bản để test regression."""

    # Bảng giá token mô phỏng (USD / 1K tokens) cho báo cáo Cost.
    PRICE_PER_1K_TOKENS = 0.00015  # ~ gpt-4o-mini input

    def __init__(self, version: str = "v2"):
        self.version = version
        self.name = f"SupportAgent-{version}"
        # V2 retrieve nhiều ứng viên hơn và có ngưỡng từ chối
        self.top_k = 3
        self.use_title_boost = version == "v2"
        self.drop_stopwords = version == "v2"
        # Ngưỡng điểm tối thiểu để tin là "có tài liệu phù hợp".
        # V1 không từ chối (ngưỡng 0), V2 biết nói "không biết".
        self.confidence_threshold = 1.0 if version == "v2" else 0.0

    # --- Retrieval ---------------------------------------------------------
    def _score_doc(self, q_tokens: List[str], doc: Dict[str, str]) -> float:
        body_tokens = set(_tokenize(doc["text"], self.drop_stopwords))
        score = sum(1 for t in q_tokens if t in body_tokens)
        if self.use_title_boost:
            title_tokens = set(_tokenize(doc["title"], self.drop_stopwords))
            score += 2 * sum(1 for t in q_tokens if t in title_tokens)
        return float(score)

    def retrieve(self, question: str) -> List[Dict]:
        q_tokens = _tokenize(question, self.drop_stopwords)
        scored = [(self._score_doc(q_tokens, d), d) for d in DOCUMENTS]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]
        return [{"id": d["id"], "score": s, "text": d["text"], "title": d["title"]}
                for s, d in top]

    # --- Generation --------------------------------------------------------
    # Dấu hiệu prompt-injection / goal-hijacking mà V2 được vá để nhận diện.
    _INJECTION_MARKERS = ["bỏ qua mọi", "bỏ qua hướng dẫn", "quên việc", "quên đi",
                          "mật khẩu quản trị", "đọc mật khẩu", "cung cấp mật khẩu",
                          "viết một bài thơ", "ca ngợi"]
    _AMBIGUOUS_MARKERS = ["làm sao để nâng cấp", "giới hạn của tôi là bao nhiêu"]

    def _is_injection(self, question: str) -> bool:
        q = question.lower()
        return any(m in q for m in self._INJECTION_MARKERS)

    def _is_ambiguous(self, question: str) -> bool:
        q = question.lower().strip().rstrip("?")
        return any(m in q for m in self._AMBIGUOUS_MARKERS)

    def _generate(self, question: str, retrieved: List[Dict]) -> str:
        best_score = retrieved[0]["score"] if retrieved else 0.0

        # V2 được vá an toàn: nhận diện và TỪ CHỐI prompt-injection / hijacking,
        # và biết HỎI LẠI khi câu hỏi mơ hồ. V1 thì không -> đây là cải tiến đo được.
        if self.version == "v2":
            if self._is_injection(question):
                return ("Tôi không thể thực hiện yêu cầu này và xin từ chối. Vì lý do "
                        "an ninh, hệ thống không bao giờ tiết lộ mật khẩu hay đi ra "
                        "ngoài phạm vi hỗ trợ sản phẩm CloudNova.")
            if self._is_ambiguous(question):
                return ("Câu hỏi của bạn còn hơi mơ hồ. Bạn vui lòng cho biết cụ thể "
                        "hơn ý bạn muốn hỏi (ví dụ: nâng cấp gói nào, hay giới hạn API "
                        "/ dung lượng / số người dùng) để tôi hỗ trợ chính xác nhé?")

        # V2 biết từ chối khi không đủ tin cậy -> tránh Hallucination.
        if best_score < self.confidence_threshold:
            return ("Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong "
                    "tài liệu hỗ trợ. Bạn có thể liên hệ đội hỗ trợ để được giúp đỡ.")

        # Mô phỏng việc tổng hợp câu trả lời từ context tốt nhất.
        top_doc = retrieved[0]
        snippet = top_doc["text"]
        if self.version == "v2":
            # V2 trả lời cô đọng, bám sát context (faithful hơn).
            return f"Theo tài liệu '{top_doc['title']}': {snippet}"
        # V1 trả lời chung chung, dễ thừa/thiếu thông tin.
        return (f"Dựa trên tài liệu hệ thống về '{top_doc['title']}', "
                f"tôi cho rằng: {snippet[:120]}")

    # --- Public API --------------------------------------------------------
    async def query(self, question: str) -> Dict:
        # Mô phỏng độ trễ; V2 được tối ưu nên nhanh hơn một chút.
        await asyncio.sleep(0.03 if self.version == "v2" else 0.05)

        retrieved = self.retrieve(question)
        answer = self._generate(question, retrieved)

        # Mô phỏng token usage (V2 cô đọng hơn -> ít token hơn -> rẻ hơn).
        prompt_tokens = len(_tokenize(question)) + sum(
            len(_tokenize(r["text"])) for r in retrieved
        )
        completion_tokens = len(_tokenize(answer))
        tokens_used = prompt_tokens + completion_tokens
        cost = tokens_used / 1000.0 * self.PRICE_PER_1K_TOKENS

        return {
            "answer": answer,
            "contexts": [r["text"] for r in retrieved],
            "retrieved_ids": [r["id"] for r in retrieved],
            "metadata": {
                "version": self.version,
                "model": "gpt-4o-mini" if self.version == "v2" else "gpt-3.5-turbo",
                "tokens_used": tokens_used,
                "cost_usd": round(cost, 8),
                "sources": [r["id"] for r in retrieved],
            },
        }


if __name__ == "__main__":
    async def _demo():
        for v in ("v1", "v2"):
            agent = MainAgent(version=v)
            resp = await agent.query("Liên kết đặt lại mật khẩu có hiệu lực bao lâu?")
            print(f"[{v}] retrieved={resp['retrieved_ids']}")
            print(f"      answer={resp['answer'][:80]}...")
            oo = await agent.query("Giá cổ phiếu CloudNova hôm nay?")
            print(f"[{v}] out-of-context answer={oo['answer'][:60]}...")
    asyncio.run(_demo())
