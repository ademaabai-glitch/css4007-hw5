"""Граф Части 2: модель просит инструмент, программа его выполняет, результат возвращается модели."""
import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import helpdesk_tools as ht

MAX_TOOL_ROUNDS = 5   # ЛИМИТ цикла: проверяется в route_after_model

SYSTEM_PROMPT = """You are a university IT helpdesk assistant (practice version, fictional data).
You help with Wi-Fi problems and password recovery only.
Tools:
- find_instructions: look up Wi-Fi or password instructions.
- check_service_status: check if 'wifi' or 'account_recovery' is operational, in outage, or unknown.
- prepare_ticket: save a LOCAL ticket draft. Nothing is ever sent to a real helpdesk.
Rules:
1. Use find_instructions when the user asks for instructions; never invent steps.
2. Use check_service_status when the user asks about an outage or whether a service is down.
3. If the user asks to prepare a ticket only IF the service is operating: check the status first,
   then act on the result. If the status is 'outage', do NOT prepare the ticket; say there is an
   outage and that no repair time is given. If the status is 'unknown', do NOT prepare the ticket;
   say you cannot confirm the status and ask whether the user wants a draft explaining that the
   status could not be checked.
4. Call prepare_ticket only when you know BOTH the problem description AND the steps the user
   already tried. If either is missing, ask the user. Never invent details. Use only steps the
   user actually stated. Never request or store passwords or recovery codes.
5. After preparing a draft, give its filename and say that nothing was sent.
6. Do not invent repair times or claim real university outages."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int        # сколько раз за этот ход выполнялись инструменты
    limit_reached: bool


def _text(content) -> str:
    """Достаёт текст, даже если модель вернула список блоков."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def make_tools(ctx: dict):
    """ctx = {'profile', 'user_id', 'thread_id'} задаёт ПРОГРАММА, а не модель."""

    @tool("find_instructions")
    def find_instructions_tool(topic: str) -> dict:
        """Look up local helpdesk instructions. topic must be 'wifi' or 'password'."""
        return ht.find_instructions(topic)

    @tool("check_service_status")
    def check_service_status_tool(service: str) -> dict:
        """Check a service status. service must be 'wifi' or 'account_recovery'.
        Returns status: operational, outage or unknown."""
        return ht.check_service_status(service, ctx["profile"])

    @tool("prepare_ticket")
    def prepare_ticket_tool(category: str, summary: str, steps_tried: list[str],
                            service_status: str = "not checked") -> dict:
        """Prepare a LOCAL ticket draft. category: 'wifi' or 'password'. summary: the problem as
        the user described it. steps_tried: ONLY steps the user said they tried.
        service_status: 'operational', 'outage', 'unknown' or 'not checked'."""
        ref = ht.find_instructions(category)
        refs = [ref["instruction_id"]] if ref.get("found") else []
        return ht.prepare_ticket(category, summary, steps_tried, ctx["user_id"], ctx["thread_id"],
                                 service_status, ctx["profile"], refs)

    return [find_instructions_tool, check_service_status_tool, prepare_ticket_tool]


def make_llm(tools):
    from langchain_openai import ChatOpenAI
    temp = os.environ.get("LLM_TEMPERATURE", "0")          # "none" = не передавать temperature
    kwargs = {} if temp.lower() == "none" else {"temperature": float(temp)}
    return ChatOpenAI(model=os.environ["MODEL_NAME"], **kwargs).bind_tools(tools)


def build_agent(ctx: dict, llm=None, checkpointer=None):
    tools = make_tools(ctx)
    tool_map = {t.name: t for t in tools}
    llm = llm or make_llm(tools)

    def agent(state: AgentState) -> dict:
        """МОДЕЛЬ запрашивает инструмент или даёт ответ."""
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def run_tools(state: AgentState) -> dict:
        """НАША ПРОГРАММА выполняет запрошенные инструменты."""
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
        """Сработал лимит: закрываем висящие запросы и отвечаем без новых инструментов."""
        skipped = [ToolMessage(content=json.dumps({"error": "tool loop limit reached"}),
                               tool_call_id=c["id"], name=c["name"])
                   for c in state["messages"][-1].tool_calls]
        final = AIMessage(content="I reached the tool-use limit for this request, "
                                  "so I stopped. Please try again or rephrase.")
        return {"messages": skipped + [final], "limit_reached": True}

    def route_after_model(state: AgentState) -> str:
        """УСЛОВНОЕ РЕБРО: инструмент, конец или лимит."""
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
    g.add_edge("run_tools", "agent")      # результат инструмента возвращается модели
    g.add_edge("limit_reached", END)
    return g.compile(checkpointer=checkpointer)


SOURCES = {"find_instructions": "reference file data/helpdesk.json",
           "check_service_status": "service snapshot (service_status.json)",
           "prepare_ticket": "local ticket file"}


def run_turn(graph, thread_id: str, text: str) -> dict:
    """Один ход диалога: возвращает ответ и трассу (запросы модели, результаты инструментов)."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(text)], "tool_rounds": 0, "limit_reached": False}
    trace, final, limit, tickets = [], "", False, []
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
                                  "source": SOURCES.get(m.name), "result": res,
                                  "tool_rounds_after": update["tool_rounds"]})
            elif node == "limit_reached":
                limit, final = True, _text(update["messages"][-1].content)
                trace.append({"node": "limit_reached", "event": "loop_limit_hit", "content": final})
    return {"reply": final, "trace": trace, "tickets": tickets, "limit_reached": limit}
