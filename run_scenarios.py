"""Запуск сценариев из data/scenarios.json. restart_process = новый процесс Python."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_scenario(sid, storage):
    sc = {s["id"]: s for s in json.load(open(ROOT / "data/scenarios.json", encoding="utf-8"))["scenarios"]}[sid]
    if sc.get("reset_before"):
        shutil.rmtree(storage, ignore_errors=True)            # сброс памяти перед 1-м событием

    segments = [[]]
    for ev in sc["events"]:
        if ev["action"] == "restart_process":
            segments.append([])                               # граница процессов
        else:
            segments[-1].append({k: ev[k] for k in ("user_id", "thread_id", "message")})

    processes, commands = [], []
    for i, seg in enumerate(segments, 1):
        tmp = Path(tempfile.mkdtemp())
        ev_file, out_file = tmp / "events.json", tmp / "out.json"
        json.dump(seg, open(ev_file, "w", encoding="utf-8"), ensure_ascii=False)
        cmd = [sys.executable, "src/app.py", "--profile", sc["status_profile"], "--storage", storage,
               "--events-file", str(ev_file), "--json-out", str(out_file)]
        commands.append(f"python src/app.py --profile {sc['status_profile']} --storage {storage} "
                        f"--events-file <process-{i}-events.json>")
        print(f"\n##### {sid}: процесс {i} из {len(segments)} #####")
        subprocess.run(cmd, check=True, cwd=ROOT, env=os.environ.copy())
        processes.append(json.load(open(out_file, encoding="utf-8")))

    log = {"scenario_id": sid, "title": sc["title"], "commands": commands,
           "model": os.environ.get("MODEL_NAME"), "provider": "OpenAI",
           "status_profile": sc["status_profile"], "storage": storage,
           "process_ids": [p["pid"] for p in processes],
           "restarted_process": len(segments) > 1 and len({p["pid"] for p in processes}) > 1,
           "processes": processes}
    out = ROOT / "outputs" / f"{sid}_part3.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(log, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nЛог сохранён: {out.relative_to(ROOT)}  | PID процессов: {log['process_ids']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["S10", "S11", "S12", "S13"])
    ap.add_argument("--storage", default=os.environ.get("HELPDESK_STORAGE_DIR", "local-storage"))
    a = ap.parse_args()
    for sid in a.ids:
        run_scenario(sid, a.storage)
