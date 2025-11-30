import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


def get_db_path() -> Path:
    """
    Return the path to the SQLite database used by the FastAPI app.
    Assumes the same folder structure:
    - this file: tokens_dashboard.py
    - app DB:   ./data/app.db
    """
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "data" / "app.db"
    return db_path


def load_users_dataframe(db_path: Path) -> pd.DataFrame:
    """
    Load the 'users' table into a pandas DataFrame.
    """
    if not db_path.exists():
        return pd.DataFrame(columns=["id", "username", "tokens"])

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT id, username, tokens FROM users", conn)
    finally:
        conn.close()

    return df


def main() -> None:
    st.set_page_config(
        page_title="Users & Tokens Dashboard",
        layout="centered",
    )

    st.title("Users & Tokens Dashboard")
    st.write(
        "This dashboard reads the same SQLite database used by the FastAPI server "
        "and shows all users and their current token balance."
    )

    db_path = get_db_path()
    st.caption(f"Database path: `{db_path}`")

    df_users = load_users_dataframe(db_path)

    if df_users.empty:
        st.warning("No users found in the database yet.")
    else:
        st.subheader("Registered users")
        st.dataframe(df_users, use_container_width=True)

        st.write("---")
        st.subheader("Summary")

        total_users = len(df_users)
        total_tokens = int(df_users["tokens"].sum())

        st.metric("Total users", total_users)
        st.metric("Total tokens in system", total_tokens)


if __name__ == "__main__":
    main()

