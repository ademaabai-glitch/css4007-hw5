# HW5 — My LangGraph helpdesk

Provider and model: OpenAI, `gpt-4o-mini`, temperature 0, through `langchain-openai`. Python 3.13 (Google Colab). All data is fictional and nothing is sent to a real helpdesk.

## Part 1 — How the graph works

### My graph and state

```
START → read_request ─┬─ category = wifi or both ─→ wifi_help     → END
                      ├─ category = password ─────→ password_help → END
                      └─ category = other ────────→ other_help    → END
```

State fields:

| Field | Holds | Changed by |
|---|---|---|
| message | the user's original message | nobody (set at start) |
| category | wifi / password / both / other | read_request |
| reply | the answer for the user | wifi_help, password_help or other_help |

`both` is my own addition: `read_request` marks a message that mentions both topics. In the original graph the router sends `both` to `wifi_help`, so the routing rules from the brief are unchanged.

What each node reads and changes:

| Node | Reads | Changes |
|---|---|---|
| read_request | message | category |
| wifi_help | nothing from state; text comes from data/helpdesk.json | reply |
| password_help | same | reply |
| other_help | nothing | reply (asks "Wi-Fi or password?") |

**Update vs. conditional edge.** A node returns an update, a dictionary of the fields it changes, for example `{"category": "wifi"}`. LangGraph merges it into the state. A conditional edge (`route_original`) changes nothing in the state. It only returns the name of the next node.

**What the next node receives.** The next node receives the current full graph state, not only the previous node's update. In the S01 trace, `wifi_help` sees `message`, `category` and `reply` together, although `category` was written two steps earlier.

**START, END, compile.** `START` and `END` are built-in entry and exit points, not nodes with code. The run begins by following the edge from START, and it finishes when it reaches END, which returns the final state. `compile()` turns my description of nodes and edges into a runnable graph and checks that the edges are valid.

### Predictions before running

I wrote the predictions before running the code.

| Message | Predicted route | Actual route | What I learned |
|---|---|---|---|
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help | Alias `wi-fi` found; category = wifi |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help | Lower-casing makes the match case-insensitive |
| My laptop screen is broken. | read_request → other_help | read_request → other_help | No alias found, so the assistant only asks which topic |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help | category = both, yet the wifi rule is checked first |

All four predictions matched the actual runs (logs `outputs/S01_part1.json` … `S04_part1.json`).

**A message with both topics.** `read_request` set `category = both`. The original router sends `both` to `wifi_help`, because the Wi-Fi rule comes first and the password rule ("otherwise, if it contains password") is never reached. The user got Wi-Fi connection steps and nothing about the password. This is a design weakness, not a crash.

### My change to the graph

```
START → read_request ─┬─ wifi ─────→ wifi_help       → END
                      ├─ password ─→ password_help   → END
                      ├─ other ────→ other_help      → END
                      └─ both ─────→ clarify_problem → END
```

New trace for "My Wi-Fi password does not work." (S05):

```
START → read_request → clarify_problem → END
after read_request:   category = both, reply = ""
after clarify_problem: reply = "Your message mentions both Wi-Fi and a password.
                                Which problem do you want to solve first: Wi-Fi or password?"
```

This is a change in graph behaviour (a different node runs), not only a different sentence from the same node. The other routes are unchanged.

**What LangGraph controls and what my functions decide.** LangGraph controls the order of execution (START → node → edge → node → END), stores the state, merges each node's update into it, and calls the router at the right moment. My Python functions decide how to find the category (aliases from helpdesk.json, lower-cased), what each node answers, and which node each category goes to. There is no model in Part 1, so the same message always takes the same route.

## Part 2 — Model and tools

Provider: OpenAI. Model: `gpt-4o-mini`, temperature 0. Code: `src/helpdesk_tools.py` (tools) and `src/agent.py` (graph).

### Loop

```
START → agent (MODEL) ─┬─ no tool_calls ────────────────→ END  (final answer)
            ▲          ├─ tool_calls, tool_rounds < 5 ──→ run_tools (MY PROGRAM)
            │          │                                      │
            └──────────┘        result added as ToolMessage, tool_rounds + 1
                       └─ tool_calls, tool_rounds ≥ 5 ──→ limit_reached → END
```

**Loop limit.** `MAX_TOOL_ROUNDS = 5`. It is enforced in the conditional edge `route_after_model`, which runs after every model call. If the model asks for a tool and `tool_rounds` has already reached 5, the graph goes to `limit_reached`. That node closes the pending tool requests and ends with a short message instead of running more tools. The limit is tested with a fake model that asks for tools forever (`tests/test_agent_loop.py`). It was never reached in my real runs (`limit_reached: false` in every log).

### Direct tool calls (without the agent)

| Tool | Direct call | Result |
|---|---|---|
| find_instructions | `find_instructions("wifi")` | found = true, GUIDE-WIFI, 4 steps |
| find_instructions | `find_instructions("printer")` | found = false, lists available topics |
| check_service_status | `check_service_status("wifi", "normal" / "wifi_outage" / "status_unavailable")` | operational / outage / unknown |
| check_service_status | `check_service_status("account_recovery", "recovery_outage")` | outage |
| prepare_ticket | called with empty summary and steps | created = false, lists missing details, no file written |
| prepare_ticket | called with a problem and one step | created = true, JSON draft file in outputs/tickets/ |

The same behaviour is covered by `tests/test_tools.py`.

### One actual run with several tools (S06, snapshot `normal`, log `outputs/S06_part2.json`)

Turn 1, "Show me the Wi-Fi troubleshooting steps."
- The model requested `find_instructions(topic="wifi")`.
- The result was GUIDE-WIFI, and the final answer listed the four steps.

Turn 2, "I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem."
- The model requested `check_service_status(service="wifi")`. The result was `operational`.
- The model then requested `prepare_ticket` with steps `["Enabled Wi-Fi", "Forgot the saved network"]` and `service_status = "operational"`.
- The result was ticket `TKT-514B8A65`, saved as `outputs/tickets/TKT-514B8A65.json`.
- The final answer said the status is operational, a draft was saved, and nothing was sent.

The ticket contains only the steps the user stated.

### Same request, different snapshots

| Snapshot | Status result | What the model did |
|---|---|---|
| normal (S06) | operational | prepared the ticket |
| wifi_outage (S07) | outage | no ticket; reported the outage and that no repair time is given |
| status_unavailable (S08) | unknown | no ticket; said it cannot confirm the status and offered a draft explaining that the status could not be checked |

### The four questions

1. **How does the graph decide between running a tool and ending?** `route_after_model` looks at the last model message. If it has no `tool_calls`, the graph goes to END. If it has tool calls and the counter is below the limit, it goes to `run_tools`. If the limit is reached, it goes to `limit_reached`.
2. **Which component requests a tool and which executes it?** The model (node `agent`) only requests a tool by returning `tool_calls` with a name and arguments. My program (node `run_tools`) executes it. The user id, thread id and status snapshot are set by my program in `ctx`. The model cannot choose them.
3. **What changes in the state after a tool result?** A `ToolMessage` with the result is appended to `messages`, `tool_rounds` increases by 1, and control returns to `agent`, which now sees the result and decides whether to call another tool or answer.
4. **How is this different from Part 1?** In Part 1 keyword rules fixed the route in advance. Here the model chooses the next step, and the choice depends on tool results. The same request led to a ticket, a refusal or a question depending on the snapshot (S06 / S07 / S08). The graph only executes the model's request and limits the loop.

## Part 3 — Memory

- **Conversation memory** is the LangGraph `SqliteSaver` checkpointer. It saves the messages of a conversation in `local-storage/conversations.db`, keyed by `thread_id`.
- **User memory** is a separate SQLite table in `local-storage/user_memory.db`, keyed by `user_id` (`src/user_memory.py`). The model calls the `save_user_preference` tool when the user asks to remember a reply language. Nothing is hard-coded and `demo_users.json` is not used as a store.
- **`thread_id` vs. `user_id`.** `thread_id` identifies one conversation: the checkpointer restores its messages and state by it. `user_id` identifies a person: the preference is read and written by it. One user can have many threads, and all of them see the user's preference but not each other's history. My program supplies both ids. The model never chooses whose data is loaded.
- **Retrieval vs. memory.** Looking up Wi-Fi instructions is retrieval. The text comes from a reference file (`data/helpdesk.json`) that I prepared in advance. It is the same for every user and does not depend on the conversation. The reply language is memory. It is information learned from one particular user during a conversation, saved, and read back later for that user only.
- Every trace records what was read: `conversation_memory_read` (checkpointer, number of earlier messages), `user_memory_read` (preferences for the user id), and the `source` field of each tool result (for example `reference file data/helpdesk.json`).

| Situation | Command used | Log file | What happened |
|---|---|---|---|
| Refer to an earlier message | `python src/run_scenarios.py --ids S10` | outputs/S10_part3.json | In turn 2 the checkpointer returned 4 earlier messages. The assistant listed the steps from turn 1: enabled Wi-Fi, forgot the saved network, tried to reconnect. |
| Close, restart and resume | `python src/run_scenarios.py --ids S11` | outputs/S11_part3.json | Two separate processes (PID 15505, then 15534). The preference was saved in process 1. In process 2 the checkpointer returned 4 messages, and the assistant recalled the Wi-Fi problem and the steps. |
| Same user, new conversation | `python src/run_scenarios.py --ids S12` | outputs/S12_part3.json | New thread, new process (PID 15555 → 15576). 0 earlier messages, but user memory held `reply_language: Kazakh` and the answer was in Kazakh. |
| Different user | `python src/run_scenarios.py --ids S13` | outputs/S13_part3.json | demo-user-b had empty memory (`{}`) and the assistant said nothing had been recorded. |

I also ran the same four situations by hand as separate `python src/app.py ...` commands (see README). That output was printed in the terminal only and was not saved as a log file.

## Part 4 — Running and checking my work

- **Setup, run commands, Python version, memory location and reset:** see `README.md`.
- **Package versions:** `requirements.txt`, pinned to the versions that worked in my environment (Python 3.13).
- **Memory reset:** delete the `local-storage/` folder, or add `--reset` to `src/app.py`.
- **CI:** `.github/workflows/tests.yml` runs `python -m pytest -q` on every push. It needs no API key. Status: [fill in after checking the Actions tab: green or red].
- **Test result in Colab:** [fill in: the output of `python -m pytest -q`, for example "N passed"].

| Test file | Mistake it would catch |
|---|---|
| test_routing.py | A wrong route, case-sensitive matching, or two-topic messages not reaching `clarify_problem` in the changed graph |
| test_tools.py | Wrong instructions, a status that ignores the selected snapshot, "unknown" treated as operational, or a ticket created with missing or invented details |
| test_memory.py | Memory lost after a real process restart (separate Python processes are launched), a preference not shared across one user's threads, history leaking between threads, or one user seeing another user's data |
| test_agent_loop.py | A ticket created during an outage or unknown status, or an endless tool loop |

**Fake model.** All tests use `tests/fake_llm.py`, a clearly labelled FAKE model with scripted rules. It is not a real LLM and needs no key. The real model (`gpt-4o-mini`) was used only for the logged runs in `outputs/` (S06–S14).

## Choices and limitations

**Choices**
- **LangGraph with a conditional edge for the tool loop.** It makes the route visible and lets me enforce the iteration limit in one place.
- **SQLite for both memories.** Conversation state uses the supplied `langgraph-checkpoint-sqlite`. The user preference uses a separate small SQLite table, so the two kinds of memory stay apart. SQLite is a local file, needs no server, and survives a restart.
- **`langchain-openai` with `gpt-4o-mini`.** It supports tool calling and is available with the supplied OpenAI adapter.

**Limitations**
- **Part 1 was run in a Colab notebook, not through a script.** The logs S01–S05 come from that run. The same logic is saved in `src/part1_graph.py` and checked by `tests/test_routing.py`.
- **S06–S09 and S14 used the Part 2 agent with in-memory conversation state.** S10–S13 used the final SQLite application. The logs come from different stages of the code.
- **Ticket status bug in S09.** The model did check the Wi-Fi status (operational) but called `prepare_ticket` without `service_status`. The saved ticket `TKT-CA976C36` therefore says `"service_status": "not checked"`. In S06 the model passed `operational` correctly. The fix I would make is to remember the last status result in the program and fill it in automatically, and to tighten the prompt.
- **Extra first-turn remark in S09.** In turn 1 the model said it would "check the service status first", although the user had not asked for that. It still only asked for the missing details and created no ticket.
- **The model is not deterministic.** Another run may behave differently, especially in the multi-tool scenarios. The saved logs are single runs.
- **Replies in Kazakh were produced by the model.** I did not have the quality of the Kazakh text checked.
- **Only a reply-language preference is stored.** The tool refuses other content on purpose, so the assistant does not store personal data.

## Supplied scenarios — my actual results

| Scenario | Command or manual steps | Log file | What happened / what I changed |
|---|---|---|---|
| S01 — Simple Wi-Fi route | `run_traced(graph_original, message)` (Part 1 notebook) | outputs/S01_part1.json | read_request → wifi_help; Wi-Fi steps from GUIDE-WIFI. Matched my prediction. |
| S02 — Case-insensitive password route | same, original graph | outputs/S02_part1.json | read_request → password_help despite capital letters. Matched. |
| S03 — Unsupported request | same, original graph | outputs/S03_part1.json | read_request → other_help; asks whether the problem is Wi-Fi or a password. Matched. |
| S04 — Two topics before the graph change | same, original graph | outputs/S04_part1.json | category = both, route went to wifi_help, so the password part was ignored. This is the weakness I fixed in S05. |
| S05 — Two topics after adding clarification | `run_traced(graph_clarify, message)` | outputs/S05_part1.json | read_request → clarify_problem; asks which problem to solve first. Fix worked. |
| S06 — Instructions then local ticket | `run_scenario('S06')` (Part 2 notebook, snapshot normal) | outputs/S06_part2.json | Turn 1: find_instructions. Turn 2: check_service_status (operational), then prepare_ticket with the user's steps; ticket TKT-514B8A65 saved locally. |
| S07 — The same request during a Wi-Fi outage | `run_scenario('S07')`, snapshot wifi_outage | outputs/S07_part2.json | Status = outage. No ticket, no repair time invented. |
| S08 — Status source unavailable | `run_scenario('S08')`, snapshot status_unavailable | outputs/S08_part2.json | Status = unknown. No ticket; offered a draft explaining that the status could not be checked. |
| S09 — Missing ticket details | `run_scenario('S09')`, snapshot normal | outputs/S09_part2.json | Turn 1 asked for the problem and steps tried. Turn 2 created ticket TKT-CA976C36 with the user's details. Defect: ticket says service_status "not checked" although the status had been checked (see Limitations). |
| S10 — Use the current conversation | `python src/run_scenarios.py --ids S10` | outputs/S10_part3.json | Turn 2 read 4 earlier messages and listed the steps already tried. |
| S11 — Persist and recall after restart | `python src/run_scenarios.py --ids S11` | outputs/S11_part3.json | Different PIDs (15505, 15534). The conversation and preference survived the restart. |
| S12 — Preference across threads without replay | `python src/run_scenarios.py --ids S12` | outputs/S12_part3.json | New thread had 0 earlier messages but recalled Kazakh from user memory and answered in Kazakh. |
| S13 — Do not share memory between users | `python src/run_scenarios.py --ids S13` | outputs/S13_part3.json | demo-user-b had no memory; the assistant said nothing was recorded. |
| S14 — Password recovery outage | `run_scenario('S14')`, snapshot recovery_outage | outputs/S14_part2.json | The model checked `account_recovery` (not wifi), got outage, and reported it without inventing a repair time. |
