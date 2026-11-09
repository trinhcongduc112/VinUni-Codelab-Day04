import json
import os
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone

# Đường dẫn tuyệt đối → chạy từ thư mục nào cũng đúng (Trap 1: FileNotFoundError).
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw-data")

VN_TZ = timezone(timedelta(hours=7))  # Giờ Việt Nam (UTC+7, không có giờ mùa hè)
NO_PRICE_LIMIT = 999999999999
VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# Đọc file product_catalog.json, lọc theo category và max_price.
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = NO_PRICE_LIMIT) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện, sắp xếp theo giá tăng dần.
        Danh sách rỗng nếu không có sản phẩm phù hợp; [{"error": ...}] nếu lỗi dữ liệu/tham số.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]

    category = (category or "").strip().lower()
    if category not in VALID_CATEGORIES:
        return [{"error": f"Danh mục không hợp lệ: '{category}'. Chỉ hỗ trợ: {', '.join(VALID_CATEGORIES)}."}]
    try:
        max_price = NO_PRICE_LIMIT if max_price is None else int(max_price)
    except (TypeError, ValueError):
        return [{"error": f"max_price phải là số nguyên (VNĐ), nhận được: {max_price!r}."}]

    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            products = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [{"error": f"Không đọc được product catalog: {exc}"}]

    results = [
        p for p in products
        if str(p.get("category", "")).lower() == category
        and isinstance(p.get("price_vnd"), (int, float))
        and p["price_vnd"] <= max_price
    ]
    return sorted(results, key=lambda p: p["price_vnd"])


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# Tạo ticket mới và lưu (append) vào support_tickets.json.
# ---------------------------------------------------------------------------

def _next_ticket_sequence(tickets: List[Dict[str, Any]]) -> int:
    """Số thứ tự kế tiếp = số thứ tự lớn nhất hiện có + 1 → không sinh trùng ID kể cả khi có ticket bị xoá."""
    sequences = [
        int(match.group(1))
        for ticket in tickets
        if isinstance(ticket, dict)
        and (match := re.fullmatch(r"TK-\d{8}-(\d+)", str(ticket.get("ticket_id", ""))))
    ]
    return max(sequences, default=0) + 1


def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status; hoặc {"error": ...} nếu thất bại.
    """
    customer_name = (customer_name or "").strip()
    issue_description = (issue_description or "").strip()
    priority = (priority or "medium").strip().lower()
    if not customer_name or not issue_description:
        return {"error": "Thiếu thông tin bắt buộc: customer_name và issue_description không được để trống."}
    if priority not in VALID_PRIORITIES:
        return {"error": f"priority không hợp lệ: '{priority}'. Chỉ nhận: {', '.join(VALID_PRIORITIES)}."}

    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")

    # 1. Load ticket cũ trước khi ghi (Trap 2). File hỏng → báo lỗi, KHÔNG ghi đè làm mất dữ liệu.
    existing_tickets: List[Dict[str, Any]] = []
    if os.path.exists(tickets_file):
        try:
            with open(tickets_file, "r", encoding="utf-8") as f:
                existing_tickets = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"Không đọc được hệ thống ticket: {exc}"}
        if not isinstance(existing_tickets, list):
            return {"error": "Dữ liệu ticket không hợp lệ (cần là một danh sách JSON)."}

    # 2. Sinh ticket_id dạng TK-YYYYMMDD-NNN
    now = datetime.now(VN_TZ)
    ticket_id = f"TK-{now:%Y%m%d}-{_next_ticket_sequence(existing_tickets):03d}"

    # 3. Append ticket mới
    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": now.isoformat(),
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    # 4. Lưu file: ghi ra file tạm rồi thay thế → file gốc không bị hỏng nếu chương trình dừng giữa chừng
    tmp_file = tickets_file + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(existing_tickets, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, tickets_file)
    except OSError as exc:
        return {"error": f"Không lưu được ticket: {exc}"}

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": new_ticket["created_at"],
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# (định dạng function declaration: name, description, parameters — dùng được cho Gemini/OpenAI)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói du lịch Vinpearl) theo danh mục "
            "và giá tối đa. Trả về danh sách sản phẩm thật (tên, giá VNĐ, mô tả, tính năng, tình trạng), "
            "sắp xếp theo giá tăng dần. BẮT BUỘC gọi tool này trước khi nêu bất kỳ thông tin sản phẩm/giá nào."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' cho xe điện VinFast, 'du_lich' cho gói du lịch/nghỉ dưỡng Vinpearl.",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": (
                        "Giá tối đa tính bằng VNĐ (số nguyên). Ví dụ: 600 triệu → 600000000; "
                        "1,2 tỷ → 1200000000. Bỏ trống nếu khách hàng không nêu ngân sách."
                    )
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ trong hệ thống chăm sóc khách hàng khi khách báo lỗi, sự cố, khiếu nại "
            "hoặc muốn ghi nhận phản hồi. Trả về ticket_id thật — KHÔNG BAO GIỜ tự đặt ticket_id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên khách hàng đúng như khách cung cấp (ví dụ: 'Lê Minh Khoa'). Không được tự đặt."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn, khách quan vấn đề cần hỗ trợ (ví dụ: 'Xe VF 8 bị lỗi hệ thống ADAS')."
                },
                "priority": {
                    "type": "string",
                    "description": (
                        "Mức độ ưu tiên: 'high' nếu khách nói nghiêm trọng/khẩn cấp/cần gấp; "
                        "'low' nếu không gấp; mặc định 'medium'."
                    ),
                    "enum": ["low", "medium", "high"]
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
