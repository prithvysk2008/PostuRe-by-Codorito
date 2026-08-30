"""LOCAL PERSISTENCE — a single SQLite file next to the app. Still offline.

Ported verbatim from posture_app.py's sqlite layer (schema, queries and all),
replacing this backend's earlier JSON-based store so the two frontends don't
silently fork session history and so a session end is an INSERT, not a
load-mutate-truncate-rewrite of the whole file.
"""
import sqlite3
from datetime import date
from typing import Optional

from .constants import DB_FILE

SESSION_HISTORY_LIMIT = 60


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_FILE, timeout=5.0)


def init_db() -> Optional[str]:
    """Create the schema if it doesn't exist yet. Returns an error string on
    failure, or None on success — callers should treat a failure as
    non-fatal, since posture/fatigue coaching works fine even if history
    can't be saved."""
    try:
        conn = _connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_date TEXT,
                daily_streak INTEGER NOT NULL DEFAULT 0,
                best_streak INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                at TEXT NOT NULL,
                minutes REAL NOT NULL,
                avg_score REAL NOT NULL,
                spine_age INTEGER NOT NULL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO user_stats (id, last_date, daily_streak, best_streak) "
                     "VALUES (1, NULL, 0, 0)")
        conn.commit()
        conn.close()
        return None
    except Exception as exc:
        return str(exc)


def load_store() -> dict:
    """Read user stats + the most recent SESSION_HISTORY_LIMIT sessions."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT last_date, daily_streak, best_streak FROM user_stats WHERE id = 1"
        ).fetchone()
        last_date, daily_streak, best_streak = row if row else (None, 0, 0)
        cur = conn.execute(
            "SELECT at, minutes, avg_score, spine_age FROM "
            "(SELECT * FROM sessions ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (SESSION_HISTORY_LIMIT,),
        )
        sessions = [{"at": r[0], "minutes": r[1], "avg_score": r[2], "spine_age": r[3]}
                   for r in cur.fetchall()]
        conn.close()
        return {"last_date": last_date, "daily_streak": daily_streak,
                "best_streak": best_streak, "sessions": sessions}
    except Exception:
        return {"last_date": None, "daily_streak": 0, "best_streak": 0, "sessions": []}


def save_user_stats(store: dict) -> None:
    try:
        conn = _connect()
        conn.execute(
            "UPDATE user_stats SET last_date = ?, daily_streak = ?, best_streak = ? WHERE id = 1",
            (store.get("last_date"), store.get("daily_streak", 0), store.get("best_streak", 0)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def save_session(session: dict) -> None:
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO sessions (at, minutes, avg_score, spine_age) VALUES (?, ?, ?, ?)",
            (session["at"], session["minutes"], session["avg_score"], session["spine_age"]),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def register_day(store: dict) -> dict:
    today = date.today().isoformat()
    last = store.get("last_date")
    if last == today:
        return store
    if last:
        try:
            gap = (date.today() - date.fromisoformat(last)).days
        except Exception:
            gap = 99
    else:
        gap = 99
    store["daily_streak"] = store.get("daily_streak", 0) + 1 if gap == 1 else 1
    store["last_date"] = today
    store["best_streak"] = max(store.get("best_streak", 0), store["daily_streak"])
    return store
