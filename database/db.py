"""
database/db.py
Per-class SQLite database manager.
Each class gets its own .db file so data never mixes.
"""

import sqlite3
import os

DB_DIR = "database"

def get_db_path(class_id: str) -> str:
    os.makedirs(DB_DIR, exist_ok=True)
    return os.path.join(DB_DIR, f"{class_id}.db")

def get_db(class_id: str) -> sqlite3.Connection:
    db = sqlite3.connect(get_db_path(class_id))
    db.row_factory = sqlite3.Row
    return db

def init_db(class_id: str = "CLASS_301"):
    """Create all tables for a class database if they don't exist."""
    db = get_db(class_id)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id   TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            class_id     TEXT NOT NULL,
            face_encoding BLOB,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT NOT NULL,
            name         TEXT,
            date         TEXT DEFAULT (date('now')),
            time         TEXT DEFAULT (time('now')),
            entry_time   TEXT,
            exit_time    TEXT,
            duration_mins INTEGER DEFAULT 0,
            status       TEXT DEFAULT 'present',
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT DEFAULT CURRENT_TIMESTAMP,
            event_type   TEXT NOT NULL,
            severity     TEXT DEFAULT 'medium',
            seat         TEXT,
            student_id   TEXT,
            description  TEXT,
            resolved     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS session_logs (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp          TEXT DEFAULT CURRENT_TIMESTAMP,
            present_count      INTEGER DEFAULT 0,
            teacher_efficiency REAL DEFAULT 0,
            phone_alerts       INTEGER DEFAULT 0,
            drowsy_count       INTEGER DEFAULT 0,
            intruder_count     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS attention_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT DEFAULT CURRENT_TIMESTAMP,
            student_id   TEXT,
            seat         TEXT,
            score        REAL,
            emotion      TEXT
        );
    """)
    db.commit()
    db.close()
    print(f"[DB] Tables ready for {class_id}")
