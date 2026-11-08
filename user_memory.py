"""Память пользователя: постоянное хранилище в SQLite, ключ = user_id."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class UserMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS user_prefs (
                user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (user_id, key))""")
            c.commit()

    def get_all(self, user_id: str) -> dict:
        with closing(sqlite3.connect(self.path)) as c:
            rows = c.execute("SELECT key, value FROM user_prefs WHERE user_id = ?",
                             (user_id,)).fetchall()
        return dict(rows)

    def set(self, user_id: str, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("""INSERT INTO user_prefs (user_id, key, value, updated_at)
                         VALUES (?, ?, ?, ?)
                         ON CONFLICT(user_id, key) DO UPDATE SET
                         value = excluded.value, updated_at = excluded.updated_at""",
                      (user_id, key, value, now))
            c.commit()
