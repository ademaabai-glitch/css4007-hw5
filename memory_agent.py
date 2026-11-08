"""Часть 3: агент с памятью разговора (checkpointer) и памятью пользователя (SQLite)."""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

import agent as base
from agent import AgentState, MAX_TOOL_ROUNDS, SYSTEM_PROMPT, _text, make_llm

MEMORY_PROMPT = """
Memory rules:
- The conversation so far is in the message history. Use it for 'I already tried that' and
  'what did we discuss' questions, and report only what was actually said.
- 'User memory' below is kept by the program for THIS user only. If the user asks you to remember
  a reply language, call save_user_preference with the language name. Save nothing else; never
  save passwords, codes or personal data.
- If a reply language is saved, write ALL your replies in that language (tool text may be
  translated; keep identifiers such as GUIDE-WIFI).
- If the user asks what they told you before and neither the history nor user memory contains it,
  say plainly that nothing has been recorded. Do not guess.
- Do not call check_service_status or prepare_ticket unless the user asks about status or a ticket."""

SOURCES = {"find_instructions": "reference file data/helpdesk.json",
           "check_service_status": "service snapshot (service_status.json)",
           "prepare_ticket": "local ticket file",
           "save_user_preference": "user memory (SQLite, write)"}


def make_tools(ctx: dict, memory):
    tools = base.make_tools(ctx)

    @tool("save_user_preference")
    def save_user_preference(language: str) -> dict:
        """Save the user's preferred reply language (for example 'Kazakh'). Only the language is stored."""
        language = (language or "").strip()[:40]
        if not language:
            return {"saved": False, "error": "empty language"}
        memory.set(ctx["user_id"], "reply_language", language)   # user_id задаёт ПРОГРАММА
        return {"saved": True, "key": "reply_language", "value": language}

    return tools + [save_user_preference]


def build_agent(ctx: dict, memory, checkpointer, llm=None):
    tools = make_tools(ctx, memory)
    tool_map = {t.name: t for t in tools}
    llm = llm or make_llm(tools)

    def agent(state: AgentState) -> dict:
        prefs = memory.get_all(ctx["user_id"])            # чтение памяти пользователя
        block = ("User memory (saved preferences for this user): " + json.dumps(prefs, ensure_ascii=False)
                 if prefs else "User memory: no preferences are saved for this user.")
        system = SYSTEM_PROMPT + "\n" + MEMORY_PROMPT + "\n\n" + block
        response = llm.invoke([SystemMessage(system)] + state["messages"])
        return {"messages": [response]}

    def run_tools(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            try:
                out = tool_map[call["name"]].invoke(call["args"])
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            results.append(ToolMessage(content=json.dumps(out, ensure_ascii=False),
                                       tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def limit_reached(state: AgentState) -> dict:
        skipped = [ToolMessage(content=json.dumps({"error": "tool loop limit reached"}),
                               tool_call_id=c["id"], name=c["name"])
                   for c in state["messages"][-1].tool_calls]
        final = AIMessage(content="I reached the tool-use limit for this request, so I stopped.")
        return {"messages": skipped + [final], "limit_reached": True}

    def route_after_model(state: AgentState) -> str:
        if not getattr(state["messages"][-1], "tool_calls", None):
            return END
        if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
            return "limit_reached"
        return "run_tools"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("run_tools", run_tools)
    g.add_node("limit_reached", limit_reached)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_model,
                            {"run_tools": "run_tools", "limit_reached": "limit_reached", END: END})
    g.add_edge("run_tools", "agent")
    g.add_edge("limit_reached", END)
    return g.compile(checkpointer=checkpointer)


def run_turn(graph, ctx: dict, memory, text: str) -> dict:
    config = {"configurable": {"thread_id": ctx["thread_id"]}}

    # Что программа прочитала из памяти ДО ответа модели
    snap = graph.get_state(config)
    prior = (snap.values or {}).get("messages", []) if snap else []
    trace = [
        {"event": "conversation_memory_read",
         "source": f"checkpointer (SQLite), thread_id={ctx['thread_id']}",
         "prior_messages": len(prior)},
        {"event": "user_memory_read",
         "source": f"user memory (SQLite), user_id={ctx['user_id']}",
         "preferences": memory.get_all(ctx["user_id"])},
    ]

    inputs = {"messages": [HumanMessage(text)], "tool_rounds": 0, "limit_reached": False}
    final, limit, tickets = "", False, []
    for chunk in graph.stream(inputs, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "agent":
                msg = update["messages"][-1]
                if msg.tool_calls:
                    trace.append({"node": "agent", "event": "model_tool_request",
                                  "calls": [{"tool": c["name"], "args": c["args"]} for c in msg.tool_calls]})
                else:
                    final = _text(msg.content)
                    trace.append({"node": "agent", "event": "model_final_answer", "content": final})
            elif node == "run_tools":
                for m in update["messages"]:
                    res = json.loads(m.content)
                    if m.name == "prepare_ticket" and res.get("created"):
                        tickets.append(res["filename"])
                    trace.append({"node": "run_tools", "event": "tool_result", "tool": m.name,
                                  "source": SOURCES.get(m.name), "result": res})
            elif node == "limit_reached":
                limit, final = True, _text(update["messages"][-1].content)
                trace.append({"node": "limit_reached", "event": "loop_limit_hit", "content": final})
    return {"reply": final, "trace": trace, "tickets": tickets, "limit_reached": limit}
