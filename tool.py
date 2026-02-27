import os
import re
import shutil
import json
import requests
from openai import OpenAI
import pandas as pd
from utils import *
from llm import LLMAgentAPI, LLMAgentToolAPI


theta = 100


###### StructureAgent ######


def create_storage_env(file_path):
    if "/TraceBench/" not in file_path:
        raise ValueError("file_path 中必须包含 '/TraceBench/' 才能替换")
    new_file_path = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    shutil.copy2(file_path, new_file_path)
    return "可操作文件所在的路径：" + new_file_path, False


def get_index_exceed_length(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    length = data["length"]
    thought = data["thought"]
    action = data["action"]
    observation = data["observation"]
    exceed_index = {
        "thought": [],
        "action": [],
        "observation": []
    }
    for i in range(length):
        t = thought[i]
        t_tokens = [tok for tok in t.split() if tok.strip()]
        if len(t_tokens) > theta or len(t) > theta * 10:
            exceed_index["thought"].append(i)
        a = action[i]
        a_tokens = [tok for tok in a.split() if tok.strip()]
        if len(a_tokens) > theta or len(a) > theta * 10:
            exceed_index["action"].append(i)
        o = observation[i]
        o_tokens = [tok for tok in o.split() if tok.strip()]
        if len(o_tokens) > theta or len(o) > theta * 10:
            exceed_index["observation"].append(i)
    return exceed_index, False


def if_need_generate_abstract(exceed_index):
    if_need = {
        "thought": True,
        "action": True,
        "observation": True
    }
    if len(exceed_index["thought"]) == 0:
        if_need["thought"] = False
    if len(exceed_index["action"]) == 0:
        if_need["action"] = False
    if len(exceed_index["observation"]) == 0:
        if_need["observation"] = False
    return "是否需要生成摘要的结果如下：" + str(if_need), False


def generate_abstract_thought(file_path, exceed_index_list, client):
    if "/TraceBench/" in file_path:
        file_path = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    thought_list = data["thought"]
    usages = []
    for indx in exceed_index_list:
        original_thought = thought_list[indx]
        system_prompt = "你是一名文本总结专家，擅长提炼长文本中的关键信息。"
        user_prompt = (
            "以下是智能体解决问题时的详细思路，请你在保持原意的基础上生成简洁的摘要，"
            f"提炼出核心步骤和关键观点，不超过{theta}词，不添加额外信息：\n\n"
            f"{original_thought}"
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        output, usage = client.generate(messages)
        if not output:
            output = original_thought[:theta*10]
        thought_list[indx] = output
        usages.append(usage)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已更新thought摘要至文件{file_path}", usages


def generate_abstract_action(file_path, exceed_index_list, client):
    if "/TraceBench/" in file_path:
        file_path = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    action_list = data["action"]
    usages = []
    for indx in exceed_index_list:
        original_action = action_list[indx]
        system_prompt = "你是一名文本代码总结专家，擅长提炼长文本或长代码中的关键信息。"
        user_prompt = (
            "以下是智能体解决问题时的详细行动，请你在保持原意的基础上生成简洁的摘要，"
            f"提炼出核心步骤和关键代码，不超过{theta}词，不添加额外信息：\n\n"
            f"{original_action}"
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        output, usage = client.generate(messages)
        if not output:
            output = original_action[:theta*10]
        action_list[indx] = output
        usages.append(usage)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已更新action摘要至文件{file_path}", usages


def generate_abstract_observation(file_path, exceed_index_list, client):
    if "/TraceBench/" in file_path:
        file_path = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    observation_list = data["observation"]
    usages = []
    for indx in exceed_index_list:
        original_observation = observation_list[indx]
        system_prompt = "你是一名文本代码总结专家，擅长提炼长文本或长代码中的关键信息。"
        user_prompt = (
            "以下是智能体解决问题时的详细观测结果，请你在保持原意的基础上生成简洁的摘要，"
            f"提炼出核心步骤和关键代码，不超过{theta}词，不添加额外信息：\n\n"
            f"{original_observation}"
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        output, usage = client.generate(messages)
        if not output:
            output = original_observation[:theta*10]
        observation_list[indx] = output
        usages.append(usage)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已更新observation摘要至文件{file_path}", usages


def generate_abstract_task(file_path, client):
    if "/TraceBench/" in file_path:
        file_path = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    task = data["task"]
    system_prompt = "你是一名任务总结专家，擅长提炼任务的核心需求。"
    user_prompt = (
        "以下是智能体需要解决的任务，请你从中总结出核心需求，"
        f"不超过{theta}词，不添加额外信息：\n\n"
        f"{task}"
    )
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    output, usage = client.generate(messages)
    data["task"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已更新task摘要至文件{file_path}", [usage]


###### InsightAgent ######


def score_task_completion(file_path, client):
    table_str = convert_trace_table_to_markdown(prepare_trace_table(file_path))
    # print(table_str)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gold_score = data["gold_score"]
    gold_judge_init = data["gold_judge"]
    gold_judge = ""
    if gold_score != None and gold_score < 1:
        if len(gold_judge_init) == 0:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。"
        else:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。错误信息如下：" + '\n'.join(gold_judge_init)
    elif gold_score == None:
        if len(gold_judge_init) != 0:
            gold_judge = "这个任务的评估信息如下：" + '\n'.join(gold_judge_init)
    system_prompt = (
        "你是任务评估专家，根据智能体完整的任务执行轨迹，"
        "给出总体任务完成度评分（0-100），并简述评分依据。"
        "如果任务没有被完成或者没有被完全完成，无论过程有多好，请给低分。"
        "请只输出一个JSON对象：{\"completion_score\": int, \"reason\": str}"
    )
    user_prompt = f"以下是任务和智能体执行轨迹表：\n{data['task']}\n{table_str}\n\n{gold_judge}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    output = None
    tmp = 0
    while not output and tmp <= 10:
        output, usage = client.generate(messages)
        tmp += 1
    data["score"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成总体任务完成度评分并写入文件{file_path}", [usage]


def detect_errors(file_path, client):
    table_str = convert_trace_table_to_markdown(prepare_trace_table(file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_prompt = (
        "你是错误分析专家，请分析智能体在整体完成任务过程中的错误，可分为主要核心错误和其他错误，"
        "输出JSON对象：{\"main_errors\": str, \"other_errors\": str}"
    )
    gold_score = data["gold_score"]
    gold_judge_init = data["gold_judge"]
    gold_judge = ""
    if gold_score != None and gold_score < 1:
        if len(gold_judge_init) == 0:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。"
        else:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。错误信息如下：" + '\n'.join(gold_judge_init)
    elif gold_score == None:
        if len(gold_judge_init) != 0:
            gold_judge = "这个任务的评估信息如下：" + '\n'.join(gold_judge_init)
    user_prompt = f"以下是任务和智能体执行轨迹表：\n{data['task']}\n{table_str}\n\n{gold_judge}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    output = None
    tmp = 0
    while not output and tmp <= 10:
        output, usage = client.generate(messages)
        tmp += 1
    data["error"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成错误检测并写入文件{file_path}", [usage]


def detect_advantages_disadvantages(file_path, client):
    table_str = convert_trace_table_to_markdown(prepare_trace_table(file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gold_score = data["gold_score"]
    gold_judge_init = data["gold_judge"]
    gold_judge = ""
    if gold_score != None and gold_score < 1:
        if len(gold_judge_init) == 0:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。"
        else:
            gold_judge = "请注意，根据自动评估结果，这个任务没有被完成。错误信息如下：" + '\n'.join(gold_judge_init)
    elif gold_score == None:
        if len(gold_judge_init) != 0:
            gold_judge = "这个任务的评估信息如下：" + '\n'.join(gold_judge_init)
    system_prompt = (
        "你是任务表现评估专家，请总结智能体整体执行中的优点与缺点，"
        "输出JSON对象：{\"advantages\": str, \"disadvantages\": str}"
    )
    user_prompt = f"以下是任务和智能体执行轨迹表：\n{data['task']}\n{table_str}\n\n{gold_judge}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    output = None
    tmp = 0
    while not output and tmp <= 10:
        output, usage = client.generate(messages)
        tmp += 1
    data["feature"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成优缺点检测并写入文件{file_path}", [usage]


def generate_insights(file_path, client):
    table_str = convert_trace_table_to_markdown(prepare_trace_table(file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_prompt = (
        "你是一名具备超强洞察力的专家，非常擅长全方面评估模型，请根据智能体所犯的错误和存在的优缺点，深入思考分析智能体到底为什么会存在这样的问题或特点，给出具有远见的洞察结果，尤其是针对智能体犯的主要错误，"
        "输出JSON对象：{\"insight\": str}"
    )
    user_prompt = f"以下是智能体犯的错误和存在的优缺点：\n{data['error']}\n{data['feature']}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    output = None
    tmp = 0
    while not output and tmp <= 10:
        output, usage = client.generate(messages)
        tmp += 1
    data["insight"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成洞察生成并写入文件{file_path}", [usage]


def generate_optimization_strategy(file_path, client):
    table_str = convert_trace_table_to_markdown(prepare_trace_table(file_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_prompt = (
        "你是优化策略专家，请基于智能体所犯的错误和存在的优缺点，以及根因分析后的洞察结果，生成可实施的优化建议，以及可用于优化微调的样例数据，"
        "输出JSON对象：{\"optimization_strategy\": str, \"finetune_sample\": dict}"
    )
    user_prompt = f"以下是智能体犯的错误和存在的优缺点，以及根因分析后的洞察结果：\n{data['error']}\n{data['feature']}\n{data['insight']}"
    output = None
    tmp = 0
    while not output and tmp <= 10:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        output, usage = client.generate(messages)
        tmp += 1
    data["optimization"] = output
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成优化策略生成并写入文件{file_path}", [usage]


###### ReportAgent ######


def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        outer_json = json.load(f)
    result = {}
    keys = ["score", "error", "feature", "insight"]
    for key, value in outer_json.items():
        if key in keys:
            result[key] = extract_inner_json(value)
        else:
            result[key] = value
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return f"结构化处理完毕，已写回原文件。", False


def generate_key_error(file_path, client):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_prompt = (
        "你是错误总结专家，请根据智能体在整体完成任务过程中的主要核心错误，生成四字短语进行总结，请直接返回总结后的四个字就可以，如果智能体没有主要核心错误，请直接返回‘没有错误’。"
    )
    user_prompt = f"以下是智能体的主要核心错误：\n{data['error']}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    output, usage = client.generate(messages)
    data["key_error"] = output.strip()
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return f"已完成四字核心错误生成并写入文件{file_path}", [usage]


def if_generate_conclude_report(file_path):
    folder = os.path.dirname(file_path)
    if not os.path.isdir(folder):
        return False
    json_files = [f for f in os.listdir(folder) if f.lower().endswith(".json")]
    return len(json_files) % 10 == 0 , False


def count_key_error_values(file_path):
    counts = {}
    folder_path = os.path.dirname(file_path)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".json"):
                file_full_path = os.path.join(root, filename)
                try:
                    with open(file_full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "key_error" in data:
                        val = data["key_error"]
                        if not isinstance(val, str):
                            val = json.dumps(val, ensure_ascii=False)
                        counts[val] = counts.get(val, 0) + 1
                except json.JSONDecodeError:
                    print(f"[WARN] 无法解析 JSON 文件: {file_full_path}")
                except Exception as e:
                    print(f"[ERROR] 处理文件 {file_full_path} 时出错: {e}")
    df = pd.DataFrame(sorted(counts.items(), key=lambda x: x[1], reverse=True),
                      columns=["key_error_value", "count"])
    try:
        return '关键错误统计如下：' + df.to_markdown(index=False), False
    except ImportError:
        return '关键错误统计如下：' + df.to_string(index=False), False


def count_completion_score_distribution(file_path):
    ranges = {
        "100": 0,
        "90-99": 0,
        "80-89": 0,
        "60-79": 0,
        "1-59": 0,
        "0": 0
    }
    total_count = 0
    folder_path = os.path.dirname(file_path)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(".json"):
                file_full_path = os.path.join(root, filename)
                try:
                    with open(file_full_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "score" in data and "gold_score" not in data:
                        score_val = data["score"]
                        if isinstance(score_val, dict) and "completion_score" in score_val:
                            cs = score_val["completion_score"]
                        else:
                            if isinstance(score_val, str):
                                match = re.search(r'completion_score[^0-9]*([0-9]+)', score_val, re.DOTALL)
                                cs = int(match.group(1)) if match else None
                            else:
                                cs = None
                    elif "gold_score" in data:
                        cs = data["gold_score"]*100
                    if isinstance(cs, (int, float)):
                        total_count += 1
                        if cs == 100:
                            ranges["100"] += 1
                        elif 90 <= cs <= 99:
                            ranges["90-99"] += 1
                        elif 80 <= cs <= 89:
                            ranges["80-89"] += 1
                        elif 60 <= cs <= 79:
                            ranges["60-79"] += 1
                        elif 1 <= cs < 60:
                            ranges["1-59"] += 1
                        elif cs==0:
                            ranges["0"] += 1
                except Exception as e:
                    print(f"[WARN] 处理文件 {file_full_path} 时出错: {e}")
    range_list = []
    for k, v in ranges.items():
        percent = f"{(v / total_count * 100):.2f}%" if total_count > 0 else "0%"
        range_list.append((k, v, percent))
    df = pd.DataFrame(range_list, columns=["score_range", "count", "percent"])
    return '分数区间分布如下：' + df.to_markdown(index=False), False


def generate_conclude_report(file_path, key_error, score_distribution, client, requirement=None, flag=False):
    folder_path = os.path.dirname(file_path)
    reports_data = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".json"):
            full_path = os.path.join(folder_path, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report_info = {
                    "id": data.get("id", fname),
                    "error": data.get("error", ""),
                    "feature": data.get("feature", ""),
                    "insight": data.get("insight", ""),
                    "optimization": data.get("optimization", ""),
                    "other": data.get("other", "")
                }
                reports_data.append(report_info)
            except Exception as e:
                print(f"[WARN] 无法解析 {full_path} ：{e}")
    if requirement:
        system_prompt = (
            "你是一位非常专业且有洞察力的评估专家，现在你面前有一批模型智能体的报告数据，这些数据都是从模型回复出错的数据中采样出来的部分 bad case，每条数据都包含了具体错误、缺点、洞察、改进方案。"
            "你需要进行全面的 bad case 分析，生成详细的错误报告，涵盖常见错误类型、缺点模式、根因分析洞察、优化方向等，加入包括错误分布和分数分布在内的全局统计数据表格，给出深入的分析。"
            "你的分析需要结合数据案例进行，可引用数据ID来实现。"
            "你的输出需要以 **Markdown 格式** 编写，并具有极强的可读性和分析价值，分为以下几个部分：\n"
            "1. 全局概览：对整个报告内容的客观总结和概述。\n"
            "2. 常见错误分析：结合 key_error 统计，指出最频繁的错误类型，并分析成因和影响。最重要的是，要从 error 中找到共性的错误趋势，即，**模型在什么情况下可能会犯什么样的错误**。请至少总结出**10个**在不同场景下容易出现的不同的错误趋势，并给出每个错误趋势在错误案例中的精确占比。\n"
            "3. 分数分布分析：结合 score_distribution，评价模型在不同分数段的稳定性和表现差异。但如果分数全部都是 0，就不需要进行不同分数段的分布分析了。\n"
            "3. 模型缺点模式：从 feature 中总结出典型的劣势或缺点模式。\n"
            "4. 根因分析与洞察生成：结合 insight 对模型发生的错误和存在的缺点进行深入的根因分析，给出可以惊艳读者的极具创新力和深入挖掘思考的洞察。与此同时，针对每个可以惊艳读者的极具创新力和深入挖掘思考的洞察，给出一个简单直白可读性强的解释和说明。\n"
            "5. 结论与建议：结合optimization，提出未来优化方向及趋势预测，为后续模型训练和评估提供可执行建议。\n"
            "你必须保证结构清晰、用词精准、逻辑严密，并且对每个部分都进行深入分析。"
            f"此外，在生成报告的同时，需要充分且优先满足用户提出的特殊需求：\n{requirement}"
        )
    else:
        system_prompt = (
            "你是一位非常专业且有洞察力的评估专家，现在你面前有一批模型智能体的报告数据，这些数据都是从模型回复出错的数据中采样出来的部分 bad case，每条数据都包含了具体错误、缺点、洞察、改进方案。"
            "你需要进行全面的 bad case 分析，生成详细的错误报告，涵盖常见错误类型、缺点模式、根因分析洞察、优化方向等，加入包括错误分布和分数分布在内的全局统计数据表格，给出深入的分析。"
            "你的分析需要结合数据案例进行，可引用数据ID来实现。"
            "你的输出需要以 **Markdown 格式** 编写，并具有极强的可读性和分析价值，分为以下几个部分：\n"
            "1. 全局概览：对整个报告内容的客观总结和概述。\n"
            "2. 常见错误分析：结合 key_error 统计，指出最频繁的错误类型，并分析成因和影响。最重要的是，要从 error 中找到共性的错误趋势，即，**模型在什么情况下可能会犯什么样的错误**。请至少总结出**10个**在不同场景下容易出现的不同的错误趋势，并给出每个错误趋势在错误案例中的精确占比。\n"
            "3. 分数分布分析：结合 score_distribution，评价模型在不同分数段的稳定性和表现差异。但如果分数全部都是 0，就不需要进行不同分数段的分布分析了。\n"
            "3. 模型缺点模式：从 feature 中总结出典型的劣势或缺点模式。\n"
            "4. 根因分析与洞察生成：结合 insight 对模型发生的错误和存在的缺点进行深入的根因分析，给出可以惊艳读者的极具创新力和深入挖掘思考的洞察。与此同时，针对每个可以惊艳读者的极具创新力和深入挖掘思考的洞察，给出一个简单直白可读性强的解释和说明。\n"
            "5. 结论与建议：结合optimization，提出未来优化方向及趋势预测，为后续模型训练和评估提供可执行建议。\n"
            "你必须保证结构清晰、用词精准、逻辑严密，并且对每个部分都进行深入分析。"
        )
    reports_summary_md = []
    for r in reports_data:
        reports_summary_md.append(
            f"### 模型ID: {r['id']}\n"
            f"**错误**:\n{r['error']}\n\n"
            f"**缺点**:\n{r['feature']}\n\n"
            f"**已有洞察**:\n{r['insight']}\n\n"
            f"**优化方案**:\n{r['optimization']}\n"
            f"**其他信息**:\n{r['other']}\n"
        )
    reports_summary_str = "\n".join(reports_summary_md)
    if not requirement:
        user_prompt = (
            f"以下是所有模型报告的核心信息：\n{reports_summary_str}\n\n"
            f"以下是全局 key_error 的统计结果（Markdown表格）：\n{key_error}\n\n"
            f"以下是全局 score 分布统计结果（Markdown表格）：\n{score_distribution}\n\n"
            "请你按照系统提示要求生成最终的 MARKDOWN 格式的详细总结报告。"
        )
    else:
        user_prompt = (
            f"以下是所有模型报告的核心信息：\n{reports_summary_str}\n\n"
            f"以下是全局 key_error 的统计结果（Markdown表格）：\n{key_error}\n\n"
            f"以下是全局 score 分布统计结果（Markdown表格）：\n{score_distribution}\n\n"
            "请你按照系统提示要求生成最终的 MARKDOWN 格式的详细总结报告。"
            f"此外，在生成报告的同时，需要充分且优先满足用户提出的特殊需求：\n{requirement}"
        )
    output = None
    while not output:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        output, usage = client.generate(messages)
    if flag:
        output_filepath = Path(folder_path) / "conclude_report.md"
    else:
        num_files = len([f for f in os.listdir(folder_path) if f.lower().endswith(".json")])
        output_filepath = Path(folder_path) / f"conclude_report_{num_files}.md"
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(output)
    return f"已生成总结报告: {output_filepath}", [usage]


def polish_conclude_report(report_path):
    report_path = Path(report_path)
    folder_path = report_path.parent
    if not report_path.exists():
        raise FileNotFoundError(f"报告不存在: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()
    trace_ids = sorted(set(re.findall(r"TraceBench-\d+", report_content)))
    if not trace_ids:
        return "未在报告中发现 TraceBench ID，无需润色。"
    appended_sections = []
    missing_files = []
    for trace_id in trace_ids:
        json_path = folder_path / f"{trace_id}.json"
        if not json_path.exists():
            missing_files.append(trace_id)
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
            section_md = (
                f"### {trace_id}\n\n"
                f"```json\n{json_str}\n```\n"
            )
            appended_sections.append(section_md)
        except Exception as e:
            appended_sections.append(
                f"### {trace_id}\n\n⚠️ 无法解析 JSON 文件：{e}\n"
            )
    appendix_md = (
        "\n\n---\n\n"
        "## 📎 附录：TraceBench 处理后的数据\n\n"
        + "\n".join(appended_sections)
    )
    if missing_files:
        appendix_md += (
            "\n\n---\n\n"
            "## ⚠️ 未找到对应 JSON 的 TraceBench ID\n\n"
            + "\n".join(f"- {tid}" for tid in missing_files)
            + "\n"
        )
    polished_path = report_path.with_name(
        report_path.stem + "_polished.md"
    )
    with open(polished_path, "w", encoding="utf-8") as f:
        f.write(report_content + appendix_md)
    return f"✅ 报告润色完成：{polished_path}", False


def modify_conclude_report(history, file_path, key_error, score_distribution, client, requirement, flag=False):
    folder_path = os.path.dirname(file_path)
    reports_data = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".json"):
            full_path = os.path.join(folder_path, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                report_info = {
                    "id": data.get("id", fname),
                    "error": data.get("error", ""),
                    "feature": data.get("feature", ""),
                    "insight": data.get("insight", ""),
                    "optimization": data.get("optimization", ""),
                    "other": data.get("other", "")
                }
                reports_data.append(report_info)
            except Exception as e:
                print(f"[WARN] 无法解析 {full_path} ：{e}")
    if requirement:
        system_prompt = (
            f"你是一位非常专业且有洞察力的评估专家，请根据历史报告信息和用户提出的修改需求，生成新的满足用户需求的MARKDOWN报告。\n\n历史信息：{str(history)}\n\n用户需求：{requirement}"
        )
    else:
        system_prompt = (
            "你是一位非常专业且有洞察力的评估专家，现在你面前有一批模型智能体的报告数据，这些数据都是从模型回复出错的数据中采样出来的部分 bad case，每条数据都包含了具体错误、缺点、洞察、改进方案。"
            "你需要进行全面的 bad case 分析，生成详细的错误报告，涵盖常见错误类型、缺点模式、根因分析洞察、优化方向等，加入包括错误分布和分数分布在内的全局统计数据表格，给出深入的分析。"
            "你的分析需要结合数据案例进行，可引用数据ID来实现。"
            "你的输出需要以 **Markdown 格式** 编写，并具有极强的可读性和分析价值，分为以下几个部分：\n"
            "1. 全局概览：对整个报告内容的客观总结和概述。\n"
            "2. 常见错误分析：结合 key_error 统计，指出最频繁的错误类型，并分析成因和影响。最重要的是，要从 error 中找到共性的错误趋势，即，**模型在什么情况下可能会犯什么样的错误**。请至少总结出**10个**在不同场景下容易出现的不同的错误趋势，并给出每个错误趋势在错误案例中的精确占比。\n"
            "3. 分数分布分析：结合 score_distribution，评价模型在不同分数段的稳定性和表现差异。但如果分数全部都是 0，就不需要进行不同分数段的分布分析了。\n"
            "3. 模型缺点模式：从 feature 中总结出典型的劣势或缺点模式。\n"
            "4. 根因分析与洞察生成：结合 insight 对模型发生的错误和存在的缺点进行深入的根因分析，给出可以惊艳读者的极具创新力和深入挖掘思考的洞察。与此同时，针对每个可以惊艳读者的极具创新力和深入挖掘思考的洞察，给出一个简单直白可读性强的解释和说明。\n"
            "5. 结论与建议：结合optimization，提出未来优化方向及趋势预测，为后续模型训练和评估提供可执行建议。\n"
            "你必须保证结构清晰、用词精准、逻辑严密，并且对每个部分都进行深入分析。"
        )
    reports_summary_md = []
    for r in reports_data:
        reports_summary_md.append(
            f"### 模型ID: {r['id']}\n"
            f"**错误**:\n{r['error']}\n\n"
            f"**缺点**:\n{r['feature']}\n\n"
            f"**已有洞察**:\n{r['insight']}\n\n"
            f"**优化方案**:\n{r['optimization']}\n"
            f"**其他信息**:\n{r['other']}\n"
        )
    reports_summary_str = "\n".join(reports_summary_md)
    if not requirement:
        user_prompt = (
            f"以下是所有模型报告的核心信息：\n{reports_summary_str}\n\n"
            f"以下是全局 key_error 的统计结果（Markdown表格）：\n{key_error}\n\n"
            f"以下是全局 score 分布统计结果（Markdown表格）：\n{score_distribution}\n\n"
            "请你按照系统提示要求生成最终的 MARKDOWN 格式的详细总结报告。"
        )
    else:
        user_prompt = (
            f"以下是所有模型报告的核心信息：\n{reports_summary_str}\n\n"
            f"以下是全局 key_error 的统计结果（Markdown表格）：\n{key_error}\n\n"
            f"以下是全局 score 分布统计结果（Markdown表格）：\n{score_distribution}\n\n"
            "请你按照系统提示要求生成最终的 MARKDOWN 格式的详细总结报告。"
        )
    output = None
    while not output:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        output, usage = client.generate(messages)
    if flag:
        output_filepath = Path(folder_path) / "conclude_report.md"
    else:
        num_files = len([f for f in os.listdir(folder_path) if f.lower().endswith(".json")])
        output_filepath = Path(folder_path) / f"conclude_report_{num_files}.md"
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(output)
    return f"已生成总结报告: {output_filepath}", [usage]