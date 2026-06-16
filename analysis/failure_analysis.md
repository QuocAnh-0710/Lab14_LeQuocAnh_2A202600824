# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Số liệu dưới đây được trích trực tiếp từ `reports/summary.json` và
> `reports/benchmark_results.json` (chạy `python main.py`). Đây là kết quả của
> phiên bản **V2 (Agent_V2_Optimized)** trừ khi ghi rõ là V1.

## 1. Tổng quan Benchmark

| Chỉ số | Agent V1 (Base) | Agent V2 (Optimized) |
|---|---|---|
| Tổng số cases | 50 | 50 |
| Pass / Fail | 26 / 24 | 42 / 8 |
| Pass rate | 52% | **84%** |
| Điểm LLM-Judge trung bình | 3.04 / 5.0 | **4.10 / 5.0** |
| Hit Rate@3 | 97.8% | **100%** |
| MRR | 0.953 | 0.917 |
| Faithfulness (RAGAS) | 0.70 | **0.865** |
| Relevancy (RAGAS) | — | xem `summary.json` |
| Agreement Rate (2 Judge) | 100% | 98% |
| Cohen's Kappa | 0.389 | 0.473 |
| Avg latency / case | ~0.06s | ~0.04s |
| Cost / eval | ~$0.00123 | ~$0.00124 |

**Quyết định Release Gate:** ✅ **APPROVE** — V2 cải thiện điểm Judge +1.06,
vượt mọi ngưỡng chất lượng/độ tin cậy và không gây hồi quy về chi phí.

> Ghi chú về Kappa: Agreement Rate thô (98%) đếm các case 2 Judge lệch nhau ≤ 1
> điểm, trong khi Cohen's Kappa (~0.47, mức "moderate") đã hiệu chỉnh phần đồng
> thuận do ngẫu nhiên. Kappa thấp hơn agreement cho thấy: **không thể tin vào
> một Judge đơn lẻ** — đây chính là lý do cần Multi-Judge Consensus.

## 2. Phân nhóm lỗi (Failure Clustering — V2, 8 case fail)

| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Answer Quality (đọc sai top-1) | 5 | Tài liệu đúng nằm trong top-3 nhưng KHÔNG ở hạng 1 (MRR=0.5); Generation chỉ đọc `retrieved[0]` |
| Hallucination khi Out-of-Context | 3 | Retriever lexical khớp từ khóa lạc đề, vượt ngưỡng tin cậy nên Agent bịa thay vì từ chối |

(So với V1: 24 fail, gồm 3 hallucination out-of-context, 3 lỗi an toàn do không
chặn prompt-injection, còn lại là Answer Quality. V2 đã vá được toàn bộ lỗi
an toàn và phần lớn lỗi chất lượng.)

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1 — `case_016`: Hỏi thuật toán mã hóa, Agent trả lời về vùng dữ liệu
- **Symptom:** Hỏi "Dữ liệu được mã hóa bằng thuật toán nào?" (GT: AES-256 / TLS 1.3),
  Agent trả lời nội dung của tài liệu *Data Residency* (doc_14). Judge: 2 & 1.
- **Why 1:** Câu trả lời lấy sai tài liệu → không chứa AES-256.
- **Why 2:** Generation chỉ dùng `retrieved[0]`, mà rank-1 là doc_14 chứ không phải doc_15.
- **Why 3:** Tài liệu đúng (doc_15) CÓ trong top-3 (Hit Rate=1) nhưng xếp hạng 2 (MRR=0.5).
- **Why 4:** Retriever lexical bị nhiễu bởi từ khóa chung ("dữ liệu", "lưu trữ") xuất hiện
  dày trong doc_14, đẩy nó lên trên doc_15.
- **Root Cause:** **Thiếu bước Reranking** và Generation **chỉ tiêu thụ top-1**.
  Hit Rate cao đã che giấu lỗi xếp hạng — đây là lý do phải theo dõi cả MRR.

### Case #2 — `case_022`: Hỏi giá cổ phiếu (out-of-context), Agent bịa
- **Symptom:** Hỏi "Giá cổ phiếu CloudNova hôm nay?" — không có trong KB. Agent V2
  trả lời bằng nội dung bảng giá gói dịch vụ (doc_05). Judge: 1 & 1.
- **Why 1:** Agent không nhận ra câu hỏi nằm ngoài phạm vi tài liệu.
- **Why 2:** Ngưỡng tin cậy (confidence threshold) dựa trên điểm overlap thô đã bị vượt
  vì từ "giá" trùng với tài liệu bảng giá.
- **Why 3:** Retriever lexical không phân biệt được "trùng từ khóa" với "đúng chủ đề".
- **Why 4:** Không có tín hiệu ngữ nghĩa (embedding/cross-encoder) để đo độ liên quan thật.
- **Root Cause:** **Cơ chế từ chối dựa trên lexical-overlap quá ngây thơ.** Cần ngưỡng
  dựa trên điểm tương đồng ngữ nghĩa hoặc một classifier "in-scope / out-of-scope".

### Case #3 — `case_003`: Hỏi khôi phục 2FA, Agent trả lời về xóa tài khoản
- **Symptom:** Hỏi cách khôi phục khi mất thiết bị 2FA (GT: dùng mã khôi phục).
  Agent trả lời nội dung *Xóa tài khoản* (doc_08). Judge: 2 & 3.
- **Why 1:** Câu trả lời lấy nhầm doc_08 thay vì doc_02.
- **Why 2:** Rank-1 sai (MRR=0.5) dù doc_02 nằm trong top-3 (Hit Rate=1).
- **Why 3:** Từ "tài khoản", "thiết bị" khớp mạnh với doc_08.
- **Root Cause:** Cùng gốc với Case #1 — **chunking/ranking theo từ khóa + chỉ đọc top-1.**

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] **Thêm Reranking (cross-encoder)** sau bước retrieve để sửa thứ hạng → kéo MRR ≈ 1.0.
- [ ] **Generation đọc top-k (k≥3)** thay vì chỉ top-1, để tận dụng Hit Rate cao.
- [ ] **Ngưỡng từ chối theo ngữ nghĩa**: thay overlap thô bằng điểm similarity của
  embedding; nếu dưới ngưỡng → trả lời "không có thông tin" (chống hallucination out-of-context).
- [ ] **Semantic Chunking** thay Fixed-size để giảm nhiễu từ khóa chung giữa các tài liệu.
- [ ] **Củng cố System Prompt**: bắt buộc "chỉ trả lời dựa trên context, nếu không có thì nói không biết".

## 5. Tối ưu Chi phí Eval (giảm ~30% mà không giảm độ chính xác)
1. **Cascade / tiered judging:** chỉ gọi Judge thứ 2 (đắt) khi Judge 1 không chắc chắn
   hoặc điểm nằm ở vùng biên (3–4). Các case Judge 1 cho 5/5 hoặc 1/5 rõ ràng thì bỏ qua
   Judge 2 → cắt ~40–50% lượt gọi Judge cho phần "dễ".
2. **Tie-breaker theo nhu cầu:** Judge thứ 3 chỉ chạy khi có xung đột (>1 điểm) — hệ thống
   hiện đã làm vậy (chỉ 1/50 case phải gọi tie-breaker).
3. **Caching kết quả Judge** theo hash (question + answer + ground_truth): khi chạy lại
   regression trên cùng dataset, không cần chấm lại các cặp đã có điểm.
4. **Sampling cho smoke-test:** trên CI mỗi commit chỉ chấm 1 lớp con 20% (stratified theo
   difficulty), full 100% chỉ chạy ở release gate.

> Kết hợp (1)+(3) thường cắt > 30% chi phí Judge mà điểm trung bình gần như không đổi,
> vì phần lớn case "dễ" cho cùng kết quả dù dùng 1 hay 2 Judge.
