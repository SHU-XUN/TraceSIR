import os
import re
import json
from datetime import datetime
from pathlib import Path


TRACE_LOG_FILE = Path("./data/trace_log.jsonl")


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_jsonl_line(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def write_txt(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(data)


def prepare_trace_table(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    thought_list = data.get("thought", [])
    action_list = data.get("action", [])
    observation_list = data.get("observation", [])

    length = min(len(thought_list), len(action_list), len(observation_list))
    table = []
    for i in range(length):
        table.append({
            "index": i,
            "thought": thought_list[i],
            "action": action_list[i],
            "observation": observation_list[i]
        })
    return table


def convert_trace_table_to_markdown(table):
    md_lines = ["| Index | Thought | Action | Observation |",
                "|-------|---------|--------|-------------|"]
    for row in table:
        index = row["index"]
        thought = str(row["thought"]).replace("|", "\\|")
        action = str(row["action"]).replace("|", "\\|")
        observation = str(row["observation"]).replace("|", "\\|")
        md_lines.append(f"| {index} | {thought} | {action} | {observation} |")
    return "\n".join(md_lines)


def log_trace_step(file_name, process, step, tool_name, args, model_thought, observation, usage, usage_tool):
    if usage_tool:
        usage_tool = [
            {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None)
            }
            for u in usage_tool
        ]
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "file": file_name,
        "process": process,
        "step": step,
        "tool": tool_name,
        "args": args,
        "model_thought": model_thought,
        "observation": observation,
        "tokens": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None
        },
        "tokens_tool": usage_tool
    }
    with TRACE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def extract_inner_json(text):
    json_block_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

    if not isinstance(text, str):
        return text
    
    match = json_block_pattern.search(text)
    if match:
        inner_json_str = match.group(1)
        cleaned = inner_json_str.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[WARN] 内层 JSON 解析失败: {e} {cleaned}")
            return text
    else:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[WARN] 内层 JSON 解析失败: {e} {text}")
            return text
    return text



def load_or_init_history(folder_path: str) -> dict:
    history_path = os.path.join(folder_path, "history.json")
    conclude_md_path = os.path.join(folder_path, "TraceBenchTMP/conclude_report.md")

    if not os.path.exists(history_path):
        if not os.path.exists(conclude_md_path):
            raise FileNotFoundError("conclude_report.md 不存在，无法初始化 V0")

        with open(conclude_md_path, "r", encoding="utf-8") as f:
            v0_report = f.read()

        history = {
            "V0": {
                "requirement": "初始报告（自动生成）",
                "report": v0_report
            }
        }

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return history

    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_next_version(history: dict) -> str:
    versions = [int(v[1:]) for v in history.keys() if v.startswith("V")]
    next_version_num = max(versions) + 1 if versions else 1
    return f"V{next_version_num}"

