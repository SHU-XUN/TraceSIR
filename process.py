from typing import List, Dict, Any, Optional
import json
import os


def parse_openai_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:

    if not messages:
        raise ValueError("messages 不能为空")

    if messages[0].get("role") == "user":
        task = messages[0].get("content")
        start_idx = 1
    else:
        task = None
        start_idx = 0

    thought: List[Optional[str]] = []
    action: List[Optional[str]] = []
    observation: List[Optional[str]] = []

    current_thought: Optional[str] = None

    i = start_idx
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        
        if role == "assistant":
            if msg.get("content") not in (None, ""):
                current_thought = msg["content"]
                i += 1
                continue
            if "tool_calls" in msg:
                tool_calls = msg["tool_calls"]
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    arg_name = tc["function"]["arguments"]
                    thought.append(current_thought)
                    action.append(tool_name + '\t' + arg_name)
                    observation.append(None)
                current_thought = None
                i += 1
                continue
            i += 1
            continue

        if role == "tool":
            for idx in range(len(observation) - 1, -1, -1):
                if observation[idx] is None:
                    observation[idx] = msg.get("content")
                    break
            i += 1
            continue

        if role == "user":
            thought.append(current_thought)
            action.append("ASK_USER")
            observation.append(msg.get("content"))
            current_thought = None
            i += 1
            continue
        i += 1

    if current_thought is not None:
        thought.append(current_thought)
        action.append(None)
        observation.append(None)

    assert len(thought) == len(action) == len(observation), \
        "thought/action/observation 长度不一致"

    def normalize(x: Optional[str]) -> str:
        return "" if x is None else x

    return {
        "task": normalize(task),
        "thought": [normalize(x) for x in thought],
        "action": [normalize(x) for x in action],
        "observation": [normalize(x) for x in observation],
        "length": len(thought)
    }


def preprocess_trace_init(job_dir: str):
    traceinit_dir = os.path.join(job_dir, "TraceInit")
    tracebench_dir = os.path.join(job_dir, "TraceBench")

    if not os.path.exists(traceinit_dir):
        raise RuntimeError("TraceInit 不存在")

    trace_idx = 1

    for root, _, files in os.walk(traceinit_dir):
        for name in sorted(files):

            if name.startswith("._") or "__MACOSX" in root:
                continue
            if not name.lower().endswith(".json"):
                continue
            src = os.path.join(root, name)
            with open(src, "r", encoding="utf-8") as f:
                try:
                    raw = load_json_safely(src)
                except Exception as e:
                    print(f"⚠️ 跳过非法文件: {name}，错误：{e}")
                    continue
            trace_id = f"TraceBench-{trace_idx}"
            try:
                normalized = normalize_input_json(raw, trace_id)
            except Exception as e:
                raise RuntimeError(f"{name} 格式转换失败: {e}")
            dst = os.path.join(tracebench_dir, f"{trace_id}.json")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=4)
            trace_idx += 1


def normalize_input_json(data: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("输入必须是 JSON object")

    if "messages" not in data or not isinstance(data["messages"], list):
        raise ValueError("缺少 messages 字段或格式错误")

    parsed = parse_openai_messages(data["messages"])

    user_task = data.get("task")
    task = user_task if user_task else parsed.get("task")

    if not task:
        raise ValueError("task 为空（用户未提供，解析也未生成）")

    result = {
        "id": trace_id,
        "oid": data.get("oid"),
        "task": task,
        "thought": parsed.get("thought", []),
        "action": parsed.get("action", []),
        "observation": parsed.get("observation", []),
        "length": parsed.get("length", 0),
        "gold_score": data.get("gold_score"),
        "gold_judge": data.get("gold_judge"),
        "other": data.get("other"),
    }

    return result


def load_json_safely(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass
    try:
        with open(path, "r", encoding="gbk") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"{path} 不是 UTF-8 / UTF-8-BOM / GBK 编码的 JSON"
    )