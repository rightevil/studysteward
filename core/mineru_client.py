"""
MinerU v4 Precision Extract API client.
Submit URLs/files, poll task status, download results.
"""
import io
import zipfile
import httpx

API_BASE = "https://mineru.net/api/v4"


def submit_url(url: str, token: str) -> str:
    """Submit a URL for HTML parsing. Returns task_id."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{API_BASE}/extract/task",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "model_version": "MinerU-HTML",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU submit failed: {data.get('msg', data)}")
        return data["data"]["task_id"]


def poll_task(task_id: str, token: str) -> dict:
    """Poll task status. Returns {state, progress, full_zip_url, ...}."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{API_BASE}/extract/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU poll failed: {data.get('msg', data)}")
        result = data["data"]
        progress = result.get("extract_progress", {})
        return {
            "state": result.get("state", "unknown"),
            "progress_pct": (
                progress.get("extracted_pages", 0)
                * 100
                // max(progress.get("total_pages", 1), 1)
                if progress
                else None
            ),
            "full_zip_url": result.get("full_zip_url", ""),
        }


def download_result(full_zip_url: str) -> str:
    """Download result ZIP and extract full.md. Returns markdown text."""
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(full_zip_url)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            if "full.md" in zf.namelist():
                return zf.read("full.md").decode("utf-8")
            # Fallback: find any .md file
            for name in zf.namelist():
                if name.endswith(".md"):
                    return zf.read(name).decode("utf-8")
    raise RuntimeError("No markdown file found in result ZIP")
