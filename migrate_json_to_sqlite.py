"""
One-time migration: import existing posture_data.json session history into
the new posture_data.db SQLite store.

Run once, from the project directory:
    python migrate_json_to_sqlite.py

Safe to re-run: if posture_data.db already has session rows, it refuses to
import again (so you can't accidentally duplicate history) unless you pass
--force.

This script is intentionally standalone — it does not import posture_app.py,
so it doesn't need MediaPipe/OpenCV/Streamlit installed to run.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(HERE, "posture_data.json")
DB_FILE = os.path.join(HERE, "posture_data.db")


def init_schema(conn: sqlite3.Connection) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Import even if posture_data.db already has session rows "
                             "(will duplicate any session already migrated).")
    args = parser.parse_args()

    if not os.path.exists(JSON_FILE):
        print(f"No {os.path.basename(JSON_FILE)} found next to this script — nothing to migrate.")
        return 0

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            store = json.load(f)
    except Exception as exc:
        print(f"Couldn't read {JSON_FILE}: {exc}")
        return 1

    conn = sqlite3.connect(DB_FILE)
    try:
        init_schema(conn)

        existing = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if existing and not args.force:
            print(f"{os.path.basename(DB_FILE)} already has {existing} session row(s). "
                 "Re-run with --force to import on top of them anyway.")
            return 1

        conn.execute(
            "UPDATE user_stats SET last_date = ?, daily_streak = ?, best_streak = ? WHERE id = 1",
            (store.get("last_date"), store.get("daily_streak", 0), store.get("best_streak", 0)),
        )

        imported, skipped = 0, 0
        for s in store.get("sessions", []):
            try:
                conn.execute(
                    "INSERT INTO sessions (at, minutes, avg_score, spine_age) VALUES (?, ?, ?, ?)",
                    (s["at"], s["minutes"], s["avg_score"], s["spine_age"]),
                )
                imported += 1
            except KeyError as exc:
                skipped += 1
                print(f"  skipped a session missing field {exc}: {s}")

        conn.commit()
        print(f"Migrated {imported} session(s), daily_streak={store.get('daily_streak', 0)}, "
             f"best_streak={store.get('best_streak', 0)} into {os.path.basename(DB_FILE)}.")
        if skipped:
            print(f"{skipped} session(s) skipped due to missing fields — see above.")
        print(f"\n{os.path.basename(JSON_FILE)} was NOT modified or deleted — "
             "safe to remove it manually once you've confirmed the app reads history correctly.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
