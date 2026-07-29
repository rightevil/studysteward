import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_url TEXT,
    type TEXT NOT NULL,
    summary TEXT DEFAULT '',
    file_hash TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES tags(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_tags (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, tag_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS research_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    final_report TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS research_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    step_no INTEGER NOT NULL,
    thought TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    observation TEXT NOT NULL,
    duration_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_document_tags_doc ON document_tags(doc_id);
CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id);

PRAGMA foreign_keys = ON;
"""

class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_document(self, title: str, source_path: str, source_url: str | None,
                     doc_type: str, summary: str, file_hash: str) -> int:
        c = self.conn.execute(
            "INSERT INTO documents (title, source_path, source_url, type, summary, file_hash) VALUES (?,?,?,?,?,?)",
            (title, source_path, source_url, doc_type, summary, file_hash)
        )
        self.conn.commit()
        return c.lastrowid

    def get_document(self, doc_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, tag: str | None = None) -> list[dict]:
        if tag:
            rows = self.conn.execute("""
                SELECT DISTINCT d.* FROM documents d
                JOIN document_tags dt ON d.id = dt.doc_id
                JOIN tags t ON dt.tag_id = t.id
                WHERE t.name = ? OR t.name LIKE ?
                ORDER BY d.created_at DESC
            """, (tag, f"{tag}/%")).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int):
        self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self.conn.commit()

    def add_tag(self, name: str, parent_id: int | None = None) -> int:
        c = self.conn.execute(
            "INSERT INTO tags (name, parent_id) VALUES (?,?)", (name, parent_id)
        )
        self.conn.commit()
        return c.lastrowid

    def get_tag_tree(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM tags ORDER BY name").fetchall()
        tags = {r["id"]: {**dict(r), "children": []} for r in rows}
        roots = []
        for t in tags.values():
            if t["parent_id"] and t["parent_id"] in tags:
                tags[t["parent_id"]]["children"].append(t)
            else:
                roots.append(t)
        return roots

    def get_tags_for_doc(self, doc_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT t.name FROM tags t JOIN document_tags dt ON t.id = dt.tag_id WHERE dt.doc_id=?",
            (doc_id,)
        ).fetchall()
        return [r["name"] for r in rows]

    def set_document_tags(self, doc_id: int, tag_paths: list[str]):
        self.conn.execute("DELETE FROM document_tags WHERE doc_id=?", (doc_id,))
        for path in tag_paths:
            tag_id = self._ensure_tag_path(path)
            self.conn.execute(
                "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?,?)",
                (doc_id, tag_id)
            )
        self.conn.commit()

    def _ensure_tag_path(self, path: str) -> int:
        parts = path.split("/")
        parent_id = None
        tag_id = None
        for part in parts:
            if parent_id:
                row = self.conn.execute(
                    "SELECT id FROM tags WHERE name=? AND parent_id=?", (part, parent_id)
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT id FROM tags WHERE name=? AND parent_id IS NULL", (part,)
                ).fetchone()
            if row:
                tag_id = row[0]
            else:
                c = self.conn.execute(
                    "INSERT INTO tags (name, parent_id) VALUES (?,?)", (part, parent_id)
                )
                tag_id = c.lastrowid
            parent_id = tag_id
        return tag_id

    def add_chunk(self, doc_id: int, content: str, chunk_index: int, embedding_id: str) -> int:
        c = self.conn.execute(
            "INSERT INTO chunks (doc_id, content, chunk_index, embedding_id) VALUES (?,?,?,?)",
            (doc_id, content, chunk_index, embedding_id)
        )
        self.conn.commit()
        return c.lastrowid

    def add_chunks(self, doc_id: int, chunks: list[tuple[str, int, str]]):
        self.conn.executemany(
            "INSERT INTO chunks (doc_id, content, chunk_index, embedding_id) VALUES (?,?,?,?)",
            ((doc_id, content, index, embedding_id) for content, index, embedding_id in chunks),
        )
        self.conn.commit()

    def get_chunks(self, doc_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_chunks(self, doc_id: int):
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self.conn.commit()

    def document_exists_by_hash(self, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE file_hash=?", (file_hash,)
        ).fetchone()
        return row is not None

    def create_research_run(self, goal: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO research_runs (goal) VALUES (?)",
            (goal,),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_research_step(
        self,
        run_id: int,
        step_no: int,
        thought: str,
        tool: str,
        args_json: str,
        observation: str,
        duration_ms: int,
    ):
        self.conn.execute(
            """
            INSERT INTO research_steps
                (run_id, step_no, thought, tool, args_json, observation, duration_ms)
            VALUES (?,?,?,?,?,?,?)
            """,
            (run_id, step_no, thought, tool, args_json, observation, duration_ms),
        )
        self.conn.commit()

    def finish_research_run(self, run_id: int, status: str, report: str):
        self.conn.execute(
            """
            UPDATE research_runs
            SET status=?, final_report=?, completed_at=datetime('now')
            WHERE id=?
            """,
            (status, report, run_id),
        )
        self.conn.commit()

    def get_research_run(self, run_id: int | None = None) -> dict | None:
        if run_id is None:
            row = self.conn.execute(
                "SELECT * FROM research_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM research_runs WHERE id=?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_research_steps(self, run_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM research_steps WHERE run_id=? ORDER BY step_no",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
