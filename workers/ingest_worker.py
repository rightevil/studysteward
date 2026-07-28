"""
Ingest workers - run pipelines in background threads.
Emit events to the EventBus and keep task lifecycle in one place.
"""
import time

from cli.events import emit


def _progress_event(task_id: int, phase: str, progress: int | None = None):
    emit("task.update", task_id=task_id, phase=phase, progress=progress)


def ingest_file(task_id: int, config, kb, source: str):
    """Import a local file in background."""
    from core.pipeline import run_ingest
    from core.task_queue import TaskQueue

    tq = TaskQueue(kb.sqlite)
    tq.mark_running(task_id)
    emit("task.update", task_id=task_id, status="running", phase="queued", progress=None)

    try:
        doc_id = run_ingest(
            kb,
            config,
            source,
            on_progress=lambda phase, progress=None: _progress_event(task_id, phase, progress),
        )
        tq.mark_completed(task_id, doc_id)
        emit("task.done", task_id=task_id, doc_id=doc_id, source=source)
        return doc_id
    except Exception as e:
        tq.mark_failed(task_id, str(e))
        emit("task.failed", task_id=task_id, error=str(e))
        return None


def ingest_url(task_id: int, config, source: str):
    """Import a URL in background via MinerU API."""
    from core.kb_manager import KBManager
    from core.mineru_client import download_result, poll_task, submit_url
    from core.pipeline import ingest_text
    from core.task_queue import TaskQueue

    kb = KBManager(config)
    tq = TaskQueue(kb.sqlite)
    tq.mark_running(task_id)
    emit("task.update", task_id=task_id, status="running", phase="submitting", progress=None)

    try:
        tid = submit_url(source, config.mineru_token)
        emit("task.update", task_id=task_id, phase="submitted", progress=None)

        while True:
            time.sleep(2)
            status = poll_task(tid, config.mineru_token)
            emit(
                "task.update",
                task_id=task_id,
                phase=status.get("state", "processing"),
                progress=status["progress_pct"],
            )
            if status["state"] in ("done", "failed"):
                break

        if status["state"] == "done":
            emit("task.update", task_id=task_id, phase="downloading", progress=None)
            text = download_result(status["full_zip_url"])
            doc_id = ingest_text(
                kb,
                config,
                source,
                text,
                "web",
                on_progress=lambda phase, progress=None: _progress_event(task_id, phase, progress),
            )
            tq.mark_completed(task_id, doc_id)
            emit("task.done", task_id=task_id, doc_id=doc_id, source=source)
        else:
            raise Exception(status.get("error", "MinerU failed"))
    except Exception as e:
        tq.mark_failed(task_id, str(e))
        emit("task.failed", task_id=task_id, error=str(e))
