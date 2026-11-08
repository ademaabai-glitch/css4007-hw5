"""Часть 1: граф без LLM (оригинальный и с clarify_problem)."""
import json
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

HELPDESK = json.loads((Path(os.environ.get("HELPDESK_DATA_DIR", "data")) / "helpdesk.json")
                      .read_text(encoding="utf-8"))
ALIASES = HELPDESK["topic_aliases"]


class HelpdeskState(TypedDict):
    message: str
    category: str
    reply: str


def mentions(text, topic):
    text = text.lower()
    return any(a in text for a in ALIASES[topic])


def format_instructions(topic):
    g = HELPDESK["instructions"][topic]
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(g["steps"], 1))
    return f"{g['title']} [{g['instruction_id']}]\n{steps}"


def read_request(state):
    w, p = mentions(state["message"], "wifi"), mentions(state["message"], "password")
    return {"category": "both" if w and p else "wifi" if w else "password" if p else "other"}


def wifi_help(state):
    return {"reply": format_instructions("wifi")}


def password_help(state):
    return {"reply": format_instructions("password")}


def other_help(state):
    return {"reply": "I can help with Wi-Fi or password problems. "
                     "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state):
    return {"reply": "Your message mentions both Wi-Fi and a password. "
                     "Which problem do you want to solve first: Wi-Fi or password?"}


def route_original(state):
    if state["category"] in ("wifi", "both"):
        return "wifi_help"
    return "password_help" if state["category"] == "password" else "other_help"


def route_with_clarification(state):
    if state["category"] == "both":
        return "clarify_problem"
    return route_original(state)


def build_graph(with_clarification=False):
    g = StateGraph(HelpdeskState)
    for name, fn in [("read_request", read_request), ("wifi_help", wifi_help),
                     ("password_help", password_help), ("other_help", other_help)]:
        g.add_node(name, fn)
    g.add_edge(START, "read_request")
    targets = {"wifi_help": "wifi_help", "password_help": "password_help", "other_help": "other_help"}
    if with_clarification:
        g.add_node("clarify_problem", clarify_problem)
        targets["clarify_problem"] = "clarify_problem"
        g.add_edge("clarify_problem", END)
    g.add_conditional_edges("read_request",
                            route_with_clarification if with_clarification else route_original, targets)
    for n in ("wifi_help", "password_help", "other_help"):
        g.add_edge(n, END)
    return g.compile()
