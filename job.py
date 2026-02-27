import os
import uuid
import json
import time
from config import BASE_WORKDIR


def create_job():
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_WORKDIR, job_id)

    os.makedirs(os.path.join(job_dir, "TraceInit"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "TraceBench"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "TraceBenchTMP"), exist_ok=True)

    status = {
        "job_id": job_id,
        "state": "created",
        "created_at": time.time(),
        "updated_at": time.time()
    }

    with open(os.path.join(job_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    return job_id, job_dir


def update_status(job_dir, state, extra=None):
    path = os.path.join(job_dir, "status.json")
    with open(path) as f:
        status = json.load(f)

    status["state"] = state
    status["updated_at"] = time.time()
    if extra:
        status.update(extra)

    with open(path, "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)