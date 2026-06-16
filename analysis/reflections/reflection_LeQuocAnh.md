# Reflection cá nhân — Lê Quốc Anh (2A202600824)

> Mẫu báo cáo cá nhân (Individual Report). Mỗi thành viên tạo 1 file
> `reflection_[Tên_SV].md` trong thư mục này.

## 1. Đóng góp kỹ thuật (Engineering Contribution)
- Module đã làm: _(ví dụ: Multi-Judge Consensus Engine `engine/llm_judge.py`,
  Async Runner `engine/runner.py`, …)_
- Commit tiêu biểu: _(dán hash/commit message)_
- Phần phức tạp nhất tự xử lý: _(ví dụ: logic xử lý xung đột giữa 2 Judge + tie-breaker)_

## 2. Chiều sâu kỹ thuật (Technical Depth)
Giải thích bằng lời của bạn:
- **MRR (Mean Reciprocal Rank):** trung bình của 1/(vị trí trúng đầu tiên). Khác Hit Rate
  ở chỗ MRR quan tâm tài liệu đúng được xếp **hạng** bao nhiêu, không chỉ "có trong top-k".
  Trong lab này, nhiều case có Hit Rate=1 nhưng MRR=0.5 → tài liệu đúng nằm ở hạng 2,
  khiến Generation (chỉ đọc top-1) trả lời sai.
- **Cohen's Kappa:** đo độ đồng thuận giữa 2 Judge **sau khi loại trừ** phần đồng thuận
  do may rủi: `kappa = (Po - Pe)/(1 - Pe)`. Kappa ~0.47 (moderate) trong khi Agreement thô
  98% → cho thấy hai Judge thực ra chỉ "tình cờ" giống nhau nhiều; cần consensus.
- **Position Bias:** Judge LLM có xu hướng thiên vị câu trả lời ở một vị trí nhất định khi
  so sánh cặp. Ta phát hiện bằng cách hoán đổi A↔B; nếu kết luận thắng/thua đổi chiều → có bias.
- **Trade-off Chi phí ↔ Chất lượng:** _(viết hiểu biết của bạn — ví dụ cascade judging)_

## 3. Giải quyết vấn đề (Problem Solving)
- Vấn đề gặp phải: _(ví dụ: V1 và V2 ban đầu cho điểm giống hệt nhau)_
- Cách chẩn đoán & khắc phục: _(ví dụ: phát hiện truyền nhầm nhãn version vào MainAgent)_
- Bài học rút ra: _(…)_
