import os
import json
from job import update_status
from pipeline import process_trace_folder, modify_trace_folder
from job_runtime import JOB_LOG_QUEUES, DONE, ERROR
from process import preprocess_trace_init


def run_job(job_dir, job_id):
    update_status(job_dir, "running")
    tracebench_path = os.path.join(job_dir, "TraceBench")

    with open(os.path.join(job_dir, "job_config.json")) as f:
        job_config = json.load(f)

    q = JOB_LOG_QUEUES[job_id]

    try:
        q.put("📥 解析用户输入 (TraceInit → TraceBench)")
        preprocess_trace_init(job_dir)

        q.put("🚀 开始 pipeline 处理")

        result = process_trace_folder(
            tracebench_path,
            job_id,
            job_config
        )
    except Exception as e:
        err_msg = f"❌ 任务失败: {e}"
        q.put(err_msg)
        q.put(f"{ERROR}:{e}")
        update_status(job_dir, "failed", {
            "error": str(e)
        })
        return

    q.put("✅ 任务完成")
    q.put(DONE)

    update_status(job_dir, "finished", {
        "result": result
    })


def rerun_job(job_dir, job_id):
    update_status(job_dir, "running")

    history_path = os.path.join(job_dir, "history.json")
    if os.path.exists(history_path):
        try:
            os.remove(history_path)
        except Exception as e:
            print(f"⚠️ 无法删除 history.json: {e}")

    tracebench_path = os.path.join(job_dir, "TraceBench")

    with open(os.path.join(job_dir, "job_config.json")) as f:
        job_config = json.load(f)

    q = JOB_LOG_QUEUES[job_id]

    try:
        q.put("🚀 开始 pipeline 处理")

        result = process_trace_folder(
            tracebench_path,
            job_id,
            job_config
        )
    except Exception as e:
        err_msg = f"❌ 任务失败: {e}"
        q.put(err_msg)
        q.put(f"{ERROR}:{e}")
        update_status(job_dir, "failed", {
            "error": str(e)
        })
        return

    q.put("✅ 任务完成")
    q.put(DONE)

    update_status(job_dir, "finished", {
        "result": result
    })


def modify_job(job_dir, job_id):
    update_status(job_dir, "running")
    tracebench_path = os.path.join(job_dir, "TraceBench")

    with open(os.path.join(job_dir, "job_config.json")) as f:
        job_config = json.load(f)

    q = JOB_LOG_QUEUES[job_id]

    try:
        q.put("🚀 开始 pipeline 处理")

        result = modify_trace_folder(
            tracebench_path,
            job_id,
            job_config
        )
    except Exception as e:
        err_msg = f"❌ 任务失败: {e}"
        q.put(err_msg)
        q.put(f"{ERROR}:{e}")
        update_status(job_dir, "failed", {
            "error": str(e)
        })
        return

    q.put("✅ 任务完成")
    q.put(DONE)

    update_status(job_dir, "finished", {
        "result": result
    })