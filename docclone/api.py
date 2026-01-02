from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .pipeline import replicate_images

app = FastAPI(title="DocClone API")

JOBS: Dict[str, Dict[str, str]] = {}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """
    <html>
      <head><title>DocClone</title></head>
      <body>
        <h1>DocClone</h1>
        <form action="/replicate" method="post" enctype="multipart/form-data">
          <input type="file" name="files" multiple />
          <button type="submit">Replicate</button>
        </form>
      </body>
    </html>
    """


def _save_uploads(files: List[UploadFile], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for file in files:
        output_path = dest / file.filename
        with output_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    return dest


@app.post("/replicate")
async def replicate(files: List[UploadFile] = File(...)) -> FileResponse:
    tmp_path = Path(tempfile.mkdtemp(prefix="docclone_"))
    input_dir = _save_uploads(files, tmp_path / "input")
    output_pdf = tmp_path / "output.pdf"
    replicate_images(str(input_dir), str(output_pdf))
    return FileResponse(output_pdf, filename="replicated.pdf")


def _run_job(job_id: str, files: List[UploadFile]) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="docclone_"))
    input_dir = _save_uploads(files, workdir / "input")
    output_pdf = workdir / "output.pdf"
    replicate_images(str(input_dir), str(output_pdf))
    JOBS[job_id]["status"] = "completed"
    JOBS[job_id]["output"] = str(output_pdf)


@app.post("/replicate/async")
async def replicate_async(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
) -> JSONResponse:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "output": ""}
    background_tasks.add_task(_run_job, job_id, files)
    return JSONResponse({"job_id": job_id, "status": "running"})


@app.get("/job/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not_found"}, status_code=404)
    if job["status"] == "completed":
        return JSONResponse({"status": "completed", "download": f"/job/{job_id}/download"})
    return JSONResponse({"status": job["status"]})


@app.get("/job/{job_id}/download")
async def job_download(job_id: str) -> FileResponse | JSONResponse:
    job = JOBS.get(job_id)
    if not job or not job.get("output"):
        return JSONResponse({"error": "not_found"}, status_code=404)
    return FileResponse(job["output"], filename="replicated.pdf")
