import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(
    dbname="vision_rag",
    user="postgres",
    password="postgres",
    host="db",
    port=5432
)

register_vector(conn)

print(
    conn.execute(
        "SELECT current_database(), version()"
    ).fetchone()
)

conn.close()
