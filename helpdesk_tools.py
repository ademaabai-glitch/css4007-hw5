"""Три инструмента помощника. Каждый работает и сам по себе, без модели."""
import json
import os
import uuid
from pathlib import Path

DATA_DIR = Path(os.environ.get("HELPDESK_DATA_DIR", "data"))
TICKETS_DIR = Path(os.environ.get("HELPDESK_TICKETS_DIR", "outputs/tickets"))

# Названия сервисов, которые принимает check_service_status
SERVICE_NAMES = {
    "wifi": "wifi", "wi-fi": "wifi",
    "account_recovery": "account_recovery", "password": "account_recovery",
}


def _load(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _topic_from_text(text: str):
    """Определяет тему (wifi/password) по алиасам из helpdesk.json."""
    text = text.lower().strip()
    for topic, aliases in _load("helpdesk.json")["topic_aliases"].items():
        if text == topic or any(a in text for a in aliases):
            return topic
    return None


# ---------- Инструмент 1: поиск инструкций (retrieval из справочника) ----------
def find_instructions(topic: str) -> dict:
    guides = _load("helpdesk.json")["instructions"]
    key = _topic_from_text(topic)
    if key is None or key not in guides:
        return {"found": False, "topic": topic, "available_topics": list(guides)}
    g = guides[key]
    return {"found": True, "topic": key, "instruction_id": g["instruction_id"],
            "title": g["title"], "steps": g["steps"]}


# ---------- Инструмент 2: статус сервиса (по выбранному снимку) ----------
def check_service_status(service: str, profile: str = "normal") -> dict:
    key = SERVICE_NAMES.get(service.lower().strip())
    profiles = _load("service_status.json")["profiles"]
    if key is None or profile not in profiles or key not in profiles[profile]:
        return {"service": service, "status_profile": profile, "status": "unknown",
                "message": "No status information is available for this service."}
    entry = profiles[profile][key]
    return {"service": key, "status_profile": profile,
            "status": entry["status"], "message": entry["message"]}


# ---------- Инструмент 3: черновик тикета (только локальный файл) ----------
def prepare_ticket(category, summary, steps_tried, user_id, thread_id,
                   service_status=None, status_profile=None,
                   reference_ids=None, tickets_dir=None) -> dict:
    missing = []
    if not summary or not summary.strip():
        missing.append("problem description")
    steps = [s.strip() for s in (steps_tried or []) if s and s.strip()]
    if not steps:
        missing.append("steps already tried")
    if missing:   # ничего не выдумываем: возвращаем, чего не хватает
        return {"created": False, "missing": missing,
                "question": "What problem are you having, and what have you already tried?"}

    ticket = _load("ticket_template.json")["ticket"]   # поля берём из шаблона
    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    ticket.update({
        "ticket_id": ticket_id, "status": "draft",
        "user_id": user_id, "thread_id": thread_id,
        "category": category, "summary": summary.strip(),
        "steps_tried": steps,
        "service_status": service_status or "not checked",
        "status_profile": status_profile,
        "reference_ids": reference_ids or [],
    })
    out = Path(tickets_dir or TICKETS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticket_id}.json"
    path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"created": True, "ticket_id": ticket_id, "filename": str(path), "ticket": ticket}
