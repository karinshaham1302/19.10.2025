from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# Path to the SQLite database used by the FastAPI app
DB_PATH = Path("data") / "app.db"


def get_connection() -> sqlite3.Connection:
    """
    Create and return a new SQLite connection.
    """
    return sqlite3.connect(DB_PATH)


def load_users(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load all users and their token balances from the database.
    """
    query = """
    SELECT
        id,
        username,
        tokens
    FROM users
    ORDER BY id;
    """
    return pd.read_sql_query(query, conn)


# ----- Streamlit page config -----
st.set_page_config(
    page_title="Tokens Dashboard",
    layout="wide",
)

st.title("Tokens Dashboard")
st.caption("Admin view of users and their token balances.")


# ----- Open DB connection and load data -----
conn = get_connection()
df_users = load_users(conn)

# ----- Summary metrics -----
st.subheader("Summary")

if df_users.empty:
    st.info("No users found in the database.")
else:
    total_users = len(df_users)
    total_tokens = int(df_users["tokens"].sum())
    zero_token_users = int((df_users["tokens"] == 0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total users", total_users)
    c2.metric("Total tokens", total_tokens)
    c3.metric("Users with 0 tokens", zero_token_users)

# ----- Users table -----
st.subheader("Users table")
st.dataframe(df_users, use_container_width=True)


# ----- Add tokens to a user -----
st.subheader("Add tokens to a user")

if df_users.empty:
    st.info("Cannot add tokens because there are no users.")
else:
    usernames = df_users["username"].tolist()

    selected_user = st.selectbox(
        "Select a username",
        options=usernames,
        index=0,
    )

    amount = st.number_input(
        "Amount of tokens to add",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
    )

    if st.button("Add tokens"):
        try:
            with conn:
                conn.execute(
                    "UPDATE users SET tokens = tokens + ? WHERE username = ?",
                    (int(amount), selected_user),
                )
            st.success(f"Added {int(amount)} tokens to '{selected_user}'.")
        except Exception as exc:
            st.error(f"Failed to add tokens: {exc}")

        # Reload data after update
        df_users = load_users(conn)
        st.dataframe(df_users, use_container_width=True)


# ----- Close connection when script finishes -----
conn.close()

