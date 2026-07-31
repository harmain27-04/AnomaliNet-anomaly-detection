import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(
    page_title="AnomaliNet Dashboard",
    layout="wide"
)

st.title("🚨 AnomaliNet Surveillance Dashboard")

DB_PATH = "database/anomaly_logs.db"

if os.path.exists(DB_PATH):

    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT *
    FROM incidents
    ORDER BY id DESC
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    st.subheader("Incident Logs")

    st.subheader("Snapshots")

    for _, row in df.iterrows():

        snapshot_path = row["snapshot"]

        if os.path.exists(snapshot_path):

            st.image(
                snapshot_path,
                caption=row["status"],
                width=400
            )

else:

    st.warning(
        "Database not found"
    )