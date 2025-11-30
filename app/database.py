from __future__ import annotations

import sqlite3
from typing import Optional, Dict, Any

from app.config import DB_PATH, DATA_DIR


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the SQLite database and create tables if needed.
    """
    DATA_DIR.mkdir(exist_ok=True)

    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, password_hash, tokens) VALUES (?, ?, ?)",
        (username, password_hash, 0),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {"id": user_id, "username": username, "tokens": 0}


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "tokens": row["tokens"],
    }


def update_tokens(username: str, tokens_delta: int) -> Optional[int]:
    """
    Add or subtract tokens for a user.
    Returns the new token balance, or None if user does not exist or result would be negative.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT tokens FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    new_tokens = row["tokens"] + tokens_delta
    if new_tokens < 0:
        conn.close()
        return None

    cur.execute(
        "UPDATE users SET tokens = ? WHERE username = ?",
        (new_tokens, username),
    )
    conn.commit()
    conn.close()
    return new_tokens


def delete_user(username: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
