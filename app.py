import os
import json
import zipfile
import queue
import time
import markdown
import shutil
from pathlib import Path
from pydantic import BaseModel


from fastapi import Form
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi import HTTPException


from job import create_job
from runner import run_job, rerun_job, modify_job
from config import BASE_WORKDIR
from job_runtime import JOB_LOG_QUEUES, DONE, ERROR



app = FastAPI(title="TraceSIR Agent Production Service")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/jobs")
async def create_job_api(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    llm_model: str = Form(...),
    llm_api_key: str = Form(...),
    llm_base_url: str = Form(...),
    report_requirement: str = Form(""),
):
    job_id, job_dir = create_job()
    JOB_LOG_QUEUES[job_id] = queue.Queue()
    
    job_config = {
        "llm": {
            "model": llm_model,
            "api_key": llm_api_key,
            "base_url": llm_base_url,
        },
        "report_requirement": report_requirement or None
    }
    with open(os.path.join(job_dir, "job_config.json"), "w") as f:
        json.dump(job_config, f, indent=2, ensure_ascii=False)

    traceinit_dir = os.path.join(job_dir, "TraceInit")

    upload_path = os.path.join(job_dir, file.filename)
    with open(upload_path, "wb") as f:
        f.write(await file.read())
    if file.filename.endswith(".zip"):
        with zipfile.ZipFile(upload_path) as z:
            z.extractall(traceinit_dir)
    else:
        os.rename(upload_path, os.path.join(traceinit_dir, file.filename))

    background_tasks.add_task(run_job, job_dir, job_id)
    return {"job_id": job_id, "state": "submitted"}


@app.get("/jobs/{job_id}")
def get_status(job_id: str):
    path = os.path.join(BASE_WORKDIR, job_id, "status.json")
    if not os.path.exists(path):
        return {"error": "job not found"}
    with open(path) as f:
        return json.load(f)


@app.get("/jobs/{job_id}/stream")
def stream_logs(job_id: str):
    if job_id not in JOB_LOG_QUEUES:
        return {"error": "job not found"}
    q = JOB_LOG_QUEUES[job_id]
    def event_generator():
        while True:
            try:
                msg = q.get(timeout=1)
                if isinstance(msg, str) and msg.startswith(ERROR):
                    yield f"data: [[ERROR]] {msg[len(ERROR)+1:]}\n\n"
                    break
                if msg == DONE:
                    yield "data: [[DONE]]\n\n"
                    break
                yield f"data: {msg}\n\n"
            except queue.Empty:
                time.sleep(0.1)
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/jobs/{job_id}/download")
def download_tracebench_tmp(job_id: str):
    job_dir = os.path.join(BASE_WORKDIR, job_id)
    trace_tmp_dir = os.path.join(job_dir, "TraceBenchTMP")
    if not os.path.exists(trace_tmp_dir):
        return {"error": "TraceBenchTMP not found"}
    zip_path = os.path.join(job_dir, "TraceBenchTMP.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(trace_tmp_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, trace_tmp_dir)
                zipf.write(full_path, arcname=f"TraceBenchTMP/{arcname}")
    return FileResponse(
        zip_path,
        filename="TraceBenchTMP.zip",
        media_type="application/zip"
    )


README_PATH = Path("templates/readme.md")

@app.get("/readme", response_class=HTMLResponse)
def readme_html():
    if not README_PATH.exists():
        return "<h3>README not found</h3>"
    md_text = README_PATH.read_text(encoding="utf-8")
    html = markdown.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "codehilite"
        ]
    )
    return f"""
    <article class="markdown-body">
      {html}
    </article>
    """


@app.post("/jobs/{job_id}/rerun")
async def rerun_job_api(
    job_id: str,
    background_tasks: BackgroundTasks,
    llm_model: str = Form(...),
    llm_api_key: str = Form(...),
    llm_base_url: str = Form(...),
    report_requirement: str = Form(""),
):
    job_dir = os.path.join(BASE_WORKDIR, job_id)

    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="job not found")

    job_config_path = os.path.join(job_dir, "job_config.json")
    if not os.path.exists(job_config_path):
        raise HTTPException(status_code=400, detail="job_config.json not found")

    with open(job_config_path, "r", encoding="utf-8") as f:
        job_config = json.load(f)

    job_config["llm"] = {
        "model": llm_model,
        "api_key": llm_api_key,
        "base_url": llm_base_url,
    }

    if report_requirement:
        job_config["report_requirement"] = report_requirement

    with open(job_config_path, "w", encoding="utf-8") as f:
        json.dump(job_config, f, indent=2, ensure_ascii=False)

    JOB_LOG_QUEUES[job_id] = queue.Queue()

    background_tasks.add_task(rerun_job, job_dir, job_id)

    return {
        "job_id": job_id,
        "state": "rerunning"
    }



@app.get("/jobs/{job_id}/conclude_report")
def get_conclude_report(job_id: str):
    report_path = os.path.join(
        BASE_WORKDIR,
        job_id,
        "TraceBenchTMP",
        "conclude_report.md"
    )
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=404,
            detail="conclude_report.md 不存在"
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return {
            "job_id": job_id,
            "content": f.read()
        }


class ReportRequirementReq(BaseModel):
    requirement: str

@app.post("/jobs/{job_id}/report_requirement")
def update_report_requirement_and_rerun(
    job_id: str,
    req: ReportRequirementReq,
    background_tasks: BackgroundTasks,
):
    job_dir = os.path.join(BASE_WORKDIR, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="job not found")

    job_config_path = os.path.join(job_dir, "job_config.json")
    if not os.path.exists(job_config_path):
        raise HTTPException(
            status_code=400,
            detail="job_config.json not found"
        )

    with open(job_config_path, "r", encoding="utf-8") as f:
        job_config = json.load(f)

    job_config["report_requirement"] = req.requirement

    with open(job_config_path, "w", encoding="utf-8") as f:
        json.dump(job_config, f, indent=2, ensure_ascii=False)

    JOB_LOG_QUEUES[job_id] = queue.Queue()

    background_tasks.add_task(modify_job, job_dir, job_id)

    return {
        "job_id": job_id,
        "state": "rerunning",
        "report_requirement": req.requirement
    }