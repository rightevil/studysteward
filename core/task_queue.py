from storage.sqlite import SQLiteStore

class TaskQueue:
    def __init__(self, store: SQLiteStore):
        self.store = store

    def enqueue(self, source: str) -> int:
        c = self.store.conn.execute(
            "INSERT INTO tasks (source, status) VALUES (?, 'pending')", (source,)
        )
        self.store.conn.commit()
        return c.lastrowid

    def get_all(self) -> list[dict]:
        rows = self.store.conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, task_id: int) -> dict | None:
        row = self.store.conn.execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        return dict(row) if row else None

    def mark_running(self, task_id: int):
        self.store.conn.execute(
            "UPDATE tasks SET status='running', updated_at=datetime('now') WHERE id=?",
            (task_id,)
        )
        self.store.conn.commit()

    def mark_completed(self, task_id: int, doc_id: int):
        self.store.conn.execute(
            "UPDATE tasks SET status='completed', doc_id=?, updated_at=datetime('now') WHERE id=?",
            (doc_id, task_id)
        )
        self.store.conn.commit()

    def mark_failed(self, task_id: int, error: str):
        self.store.conn.execute(
            "UPDATE tasks SET status='failed', error=?, updated_at=datetime('now') WHERE id=?",
            (error, task_id)
        )
        self.store.conn.commit()
