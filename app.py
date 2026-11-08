"""Терминальный помощник. Один запуск = один процесс.
Примеры:
  python src/app.py --user demo-user-a --thread t1 -m "I cannot connect to Wi-Fi."
  python src/app.py --user demo-user-a --thread t1            # интерактивно (пустая строка = выход)
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.checkpoint.sqlite import SqliteSaver
import memory_agent
from user_memory import UserMemory


def storage_paths(storage_dir):
    d = Path(storage_dir)
    return d / "conversations.db", d / "user_memory.db"


def run_events(events, profile, storage_dir, verbose=True):
    conv_db, mem_db = storage_paths(storage_dir)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    memory = UserMemory(mem_db)
    conn = sqlite3.connect(conv_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    results = []
    try:
        for ev in events:
            ctx = {"profile": profile, "user_id": ev["user_id"], "thread_id": ev["thread_id"]}
            graph = memory_agent.build_agent(ctx, memory, checkpointer)
            out = memory_agent.run_turn(graph, ctx, memory, ev["message"])
            out.update({"user_id": ev["user_id"], "thread_id": ev["thread_id"], "message": ev["message"]})
            results.append(out)
            if verbose:
                print(f"\n[{ev['user_id']} / {ev['thread_id']}] ПОЛЬЗОВАТЕЛЬ: {ev['message']}")
                for e in out["trace"]:
                    if e["event"] == "conversation_memory_read":
                        print(f"  ПАМЯТЬ РАЗГОВОРА: прочитано сообщений = {e['prior_messages']}  ({e['source']})")
                    elif e["event"] == "user_memory_read":
                        print(f"  ПАМЯТЬ ПОЛЬЗОВАТЕЛЯ: {e['preferences']}  ({e['source']})")
                    elif e["event"] == "model_tool_request":
                        print("  МОДЕЛЬ просит:", [(c["tool"], c["args"]) for c in e["calls"]])
                    elif e["event"] == "tool_result":
                        print(f"  РЕЗУЛЬТАТ {e['tool']} [{e['source']}]:",
                              json.dumps(e["result"], ensure_ascii=False)[:200])
                print("ОТВЕТ:", out["reply"])
    finally:
        conn.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="user_id (задаёт программа, не модель)")
    ap.add_argument("--thread", help="thread_id разговора")
    ap.add_argument("--profile", default="normal",
                    choices=["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
    ap.add_argument("--storage", default=os.environ.get("HELPDESK_STORAGE_DIR", "local-storage"))
    ap.add_argument("-m", "--message", action="append", default=[])
    ap.add_argument("--events-file", help="JSON-список событий {user_id, thread_id, message}")
    ap.add_argument("--json-out", help="сохранить результат запуска в JSON")
    ap.add_argument("--reset", action="store_true", help="стереть память перед запуском")
    a = ap.parse_args()

    if a.reset:
        shutil.rmtree(a.storage, ignore_errors=True)
        print("Память очищена:", a.storage)

    if a.events_file:
        events = json.load(open(a.events_file, encoding="utf-8"))
    else:
        if not (a.user and a.thread):
            ap.error("нужны --user и --thread (или --events-file)")
        events = [{"user_id": a.user, "thread_id": a.thread, "message": m} for m in a.message]
        if not events:                                     # интерактивный режим
            while True:
                try:
                    m = input("Вы: ").strip()
                except EOFError:
                    break
                if not m:
                    break
                run_events([{"user_id": a.user, "thread_id": a.thread, "message": m}],
                           a.profile, a.storage)
            return

    results = run_events(events, a.profile, a.storage)
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"pid": os.getpid(), "profile": a.profile, "storage": a.storage,
                   "turns": results}, open(a.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
