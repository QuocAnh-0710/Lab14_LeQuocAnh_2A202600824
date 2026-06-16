"""
Knowledge Base (Corpus) cho hệ thống RAG mẫu.

Đây là "nguồn sự thật" mà Agent sẽ truy xuất (retrieve) và Golden Dataset sẽ
ánh xạ Ground Truth IDs tới. Domain: Sổ tay hỗ trợ khách hàng của một SaaS giả
định tên "CloudNova".

Mỗi document có:
    - id:    Mã định danh duy nhất (dùng cho Hit Rate / MRR).
    - title: Tiêu đề ngắn.
    - text:  Nội dung tài liệu.

Việc tách KB ra một module riêng giúp cả `synthetic_gen.py` (tạo dataset) và
`agent/main_agent.py` (retrieval) dùng chung một nguồn dữ liệu nhất quán.
"""
from typing import List, Dict

DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "doc_01",
        "title": "Đặt lại mật khẩu",
        "text": (
            "Để đặt lại mật khẩu, vào trang đăng nhập và nhấn 'Quên mật khẩu'. "
            "Nhập email đăng ký, hệ thống sẽ gửi một liên kết đặt lại có hiệu lực "
            "trong 30 phút. Mật khẩu mới phải có tối thiểu 12 ký tự, gồm chữ hoa, "
            "chữ thường và ký tự đặc biệt."
        ),
    },
    {
        "id": "doc_02",
        "title": "Xác thực hai lớp (2FA)",
        "text": (
            "CloudNova hỗ trợ xác thực hai lớp qua ứng dụng TOTP (Google "
            "Authenticator, Authy) và khóa bảo mật phần cứng. Khi mất thiết bị 2FA, "
            "người dùng dùng một trong 10 mã khôi phục được cấp lúc kích hoạt 2FA."
        ),
    },
    {
        "id": "doc_03",
        "title": "Chu kỳ thanh toán & hóa đơn",
        "text": (
            "Hóa đơn được xuất vào ngày đầu tiên của mỗi chu kỳ. Khách hàng có thể "
            "chọn chu kỳ hàng tháng hoặc hàng năm. Thanh toán hàng năm được giảm "
            "20% so với tổng 12 tháng. Hóa đơn có thể tải ở mục Billing > Invoices."
        ),
    },
    {
        "id": "doc_04",
        "title": "Chính sách hoàn tiền",
        "text": (
            "Khách hàng được hoàn tiền 100% nếu hủy trong vòng 14 ngày kể từ ngày "
            "thanh toán đầu tiên. Sau 14 ngày, các gói hàng tháng không được hoàn "
            "tiền; gói hàng năm được hoàn theo tỷ lệ thời gian còn lại trừ một "
            "khoản phí xử lý 5%."
        ),
    },
    {
        "id": "doc_05",
        "title": "Các gói dịch vụ & giá",
        "text": (
            "CloudNova có 3 gói: Free (0đ, 1 người dùng, 1GB), Pro (290.000đ/tháng, "
            "10 người dùng, 100GB) và Enterprise (báo giá riêng, không giới hạn "
            "người dùng, SSO, SLA 99.9%). Có thể nâng/hạ cấp gói bất kỳ lúc nào."
        ),
    },
    {
        "id": "doc_06",
        "title": "Xuất dữ liệu & GDPR",
        "text": (
            "Người dùng có quyền xuất toàn bộ dữ liệu cá nhân ở định dạng JSON hoặc "
            "CSV tại Settings > Privacy > Export. Theo GDPR, yêu cầu xuất dữ liệu "
            "được xử lý trong tối đa 72 giờ và gửi qua liên kết tải an toàn."
        ),
    },
    {
        "id": "doc_07",
        "title": "Giới hạn tần suất API (Rate Limit)",
        "text": (
            "API CloudNova giới hạn 60 request/phút cho gói Pro và 600 request/phút "
            "cho gói Enterprise. Khi vượt giới hạn, API trả về mã lỗi HTTP 429 kèm "
            "header Retry-After cho biết số giây cần chờ trước khi thử lại."
        ),
    },
    {
        "id": "doc_08",
        "title": "Xóa tài khoản",
        "text": (
            "Việc xóa tài khoản là vĩnh viễn và không thể hoàn tác. Sau khi xác "
            "nhận, dữ liệu được giữ 30 ngày ở trạng thái 'soft delete' để khôi phục "
            "nếu cần, sau đó bị xóa hoàn toàn khỏi máy chủ."
        ),
    },
    {
        "id": "doc_09",
        "title": "Định dạng & giới hạn tải lên",
        "text": (
            "Hệ thống chấp nhận tệp PDF, DOCX, PNG, JPG và CSV. Dung lượng tối đa "
            "mỗi tệp là 50MB với gói Pro và 200MB với gói Enterprise. Tệp thực thi "
            "(.exe, .sh) bị chặn vì lý do an ninh."
        ),
    },
    {
        "id": "doc_10",
        "title": "Cam kết uptime (SLA)",
        "text": (
            "Gói Enterprise được cam kết uptime 99.9% mỗi tháng. Nếu không đạt, "
            "khách hàng được bồi thường tín dụng dịch vụ: 10% phí tháng nếu uptime "
            "dưới 99.9%, và 25% nếu dưới 99.0%."
        ),
    },
    {
        "id": "doc_11",
        "title": "Vai trò & phân quyền nhóm",
        "text": (
            "Có 3 vai trò: Owner (toàn quyền, gồm thanh toán), Admin (quản lý người "
            "dùng và cấu hình, trừ thanh toán) và Member (chỉ truy cập dự án được "
            "gán). Mỗi workspace chỉ có duy nhất một Owner tại một thời điểm."
        ),
    },
    {
        "id": "doc_12",
        "title": "Đăng nhập một lần (SSO/SAML)",
        "text": (
            "SSO qua SAML 2.0 chỉ khả dụng cho gói Enterprise. Quản trị viên cấu "
            "hình Identity Provider (Okta, Azure AD) tại Settings > Security > SSO. "
            "Khi bật SSO bắt buộc, đăng nhập bằng mật khẩu thường sẽ bị vô hiệu hóa."
        ),
    },
    {
        "id": "doc_13",
        "title": "Ứng dụng di động",
        "text": (
            "Ứng dụng CloudNova có trên iOS (yêu cầu iOS 15 trở lên) và Android "
            "(yêu cầu Android 9 trở lên). Ứng dụng di động hỗ trợ xem và chỉnh sửa "
            "cơ bản, nhưng các thao tác quản trị nâng cao chỉ có trên bản web."
        ),
    },
    {
        "id": "doc_14",
        "title": "Vùng lưu trữ dữ liệu (Data Residency)",
        "text": (
            "Khách hàng Enterprise chọn vùng lưu trữ dữ liệu: EU (Frankfurt), US "
            "(Virginia) hoặc APAC (Singapore). Vùng được chọn khi tạo workspace và "
            "không thể thay đổi sau đó nếu không di trú thủ công qua đội hỗ trợ."
        ),
    },
    {
        "id": "doc_15",
        "title": "Bảo mật & mã hóa",
        "text": (
            "Dữ liệu được mã hóa khi lưu trữ bằng AES-256 và khi truyền tải bằng "
            "TLS 1.3. CloudNova đạt chứng nhận SOC 2 Type II và ISO 27001. Nhân "
            "viên hỗ trợ không bao giờ yêu cầu khách hàng cung cấp mật khẩu."
        ),
    },
    {
        "id": "doc_16",
        "title": "Giờ hỗ trợ & kênh liên hệ",
        "text": (
            "Hỗ trợ qua email khả dụng 24/7. Live chat hoạt động từ 8h đến 20h "
            "(GMT+7) các ngày trong tuần. Gói Enterprise có thêm đường dây hotline "
            "ưu tiên và một kỹ sư hỗ trợ chuyên trách (TAM)."
        ),
    },
]

# Index tra cứu nhanh theo id
DOC_BY_ID: Dict[str, Dict[str, str]] = {d["id"]: d for d in DOCUMENTS}


def get_document(doc_id: str) -> Dict[str, str]:
    """Trả về document theo id (raise KeyError nếu không tồn tại)."""
    return DOC_BY_ID[doc_id]
