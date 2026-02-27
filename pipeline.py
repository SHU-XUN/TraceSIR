import os
import re
import sys
import json
import requests
from openai import OpenAI
from llm import LLMAgentAPI, LLMAgentToolAPI
from utils import *
from tool import *


def make_logger(logs: list, job_id: str):
    from job_runtime import JOB_LOG_QUEUES
    q = JOB_LOG_QUEUES[job_id]
    def _log(msg: str):
        print(msg)
        logs.append(msg)
        q.put(msg)
    return _log


def process_trace_folder(folder_path: str, job_id: str, job_config):


    SYSTEM_PROMPT_StructureAgent = """
    你是一个结构化信息处理智能体（StructureAgent），擅长对智能体的轨迹（trace）进行结构化、自动化的数据清理与摘要生成。你的任务是根据用户提供的一个 trace 的 JSON 文件路径，利用可用方法一步步进行摘要化并进行存储。用户会提供一个包含思路（thought）、行为（action）和观测（observation）等字段的 JSON 格式文件路径，你首先需要在新的存储环境中复制该文件，然后检查这三个主要字段的每一条记录，判断其中是否存在超长的内容，并记录这些超长的索引。对于被判定为超长的内容，你应当生成简洁而准确的摘要并将生成的摘要直接替换原有内容进行存储。同时，你也需要对task字段对应的任务描述进行摘要生成。当你完成任务后，你必须直接输出 `finish()`。请开始吧！
    """

    available_tools_StructureAgent = {
        "create_storage_env": create_storage_env,
        "get_index_exceed_length": get_index_exceed_length,
        "if_need_generate_abstract": if_need_generate_abstract,
        "generate_abstract_thought": lambda **kwargs:
        generate_abstract_thought(client=client_tool, **kwargs),
        "generate_abstract_action": lambda **kwargs:
        generate_abstract_action(client=client_tool, **kwargs),
        "generate_abstract_observation": lambda **kwargs:
        generate_abstract_observation(client=client_tool, **kwargs),
        "generate_abstract_task": lambda **kwargs:
        generate_abstract_task(client=client_tool, **kwargs)
    }

    tools_StructureAgent = [
        {
            "type": "function",
            "function": {
                "name": "create_storage_env",
                "description": "创建工作环境，将原始路径中的文件复制到新的路径中，并返回复制后的可操作文件所在的路径",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 所在的 JSON 文件路径"
                        },
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_index_exceed_length",
                "description": "遍历 trace 所在的 JSON 文件中所有的thought、action、observation字段对应的列表，判断列表里有哪些是超长的，记录下这些索引，并以 JSON 格式将这些不同字段对应的超长索引列表进行返回",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 所在的 JSON 文件路径"
                        },
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "if_need_generate_abstract",
                "description": "根据不同字段对应的超长索引列表进行判断是否需要生成摘要，true代表需要，false代表不需要",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exceed_index": {
                            "type": "object",
                            "description": "超长索引的字典映射，每个字段对应一个索引列表",
                            "properties": {
                                "thought": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "thought 字段超长的索引列表"
                                },
                                "action": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "action 字段超长的索引列表"
                                },
                                "observation": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "observation 字段超长的索引列表"
                                }
                            },
                            "required": ["thought", "action", "observation"]
                        }
                    },
                    "required": ["exceed_index"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_abstract_thought",
                "description": "针对超长的 thought 字段内容，生成简洁且保留核心信息的摘要，并将摘要写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 临时存储文件的 JSON 文件路径"
                        },
                        "exceed_index_list": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "需要生成摘要的 thought 字段超长内容的索引列表"
                        }
                    },
                    "required": ["file_path", "exceed_index_list"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_abstract_action",
                "description": "针对超长的 action 字段内容，生成简洁且保留核心信息的摘要，并将摘要写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 临时存储文件的 JSON 文件路径"
                        },
                        "exceed_index_list": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "需要生成摘要的 action 字段超长内容的索引列表"
                        }
                    },
                    "required": ["file_path", "exceed_index_list"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_abstract_observation",
                "description": "针对超长的 observation 字段内容，生成简洁且保留核心信息的摘要，并将摘要写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 临时存储文件的 JSON 文件路径"
                        },
                        "exceed_index_list": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "需要生成摘要的 observation 字段超长内容的索引列表"
                        }
                    },
                    "required": ["file_path", "exceed_index_list"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_abstract_task",
                "description": "针对 task 字段内容，生成简洁且保留核心信息的摘要，并将摘要写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "trace 临时存储文件的 JSON 文件路径"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]


    SYSTEM_PROMPT_InsightAgent = """
    你是一个针对智能体任务执行程度进行评估判断的洞察生成专家（InsightAgent），擅长基于智能体的完整执行轨迹进行多维度分析并进行全方面评估。你的任务是根据用户提供的一个 trace 的 JSON 文件路径，使用可用工具一步步地生成高质量的深入洞察并进行存储。具体来说，用户会提供一个包含思路（thought）、行为（action）和观测（observation）等字段的 JSON 格式文件路径，你首先会对整合成表格形式的这些数据进行对整体任务完成度打分；然后分析 trace 整体任务执行过程中的主要错误与其他错误；总结 trace 整体执行过程中的优点与缺点；然后需要基于这些错误和优缺点进行根因分析，从而生成具有深度的洞察；最后再生成可实施的优化建议和可微调样例数据。当你完成任务后，你必须直接输出 `finish()`。请开始吧！
    """

    available_tools_InsightAgent = {
        "score_task_completion": lambda **kwargs:
        score_task_completion(client=client_tool, **kwargs),
        "detect_errors": lambda **kwargs:
        detect_errors(client=client_tool, **kwargs),
        "detect_advantages_disadvantages": lambda **kwargs:
        detect_advantages_disadvantages(client=client_tool, **kwargs),
        "generate_insights": lambda **kwargs:
        generate_insights(client=client_tool, **kwargs),
        "generate_optimization_strategy": lambda **kwargs:
        generate_optimization_strategy(client=client_tool, **kwargs)
    }


    tools_InsightAgent = [
        {
            "type": "function",
            "function": {
                "name": "score_task_completion",
                "description": "基于 trace 表格对整体任务完成度进行 0~100 分评分并给出理由，然后写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "trace JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "detect_errors",
                "description": "分析 trace 整体任务执行过程中的主要错误与其他错误，然后写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "trace JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "detect_advantages_disadvantages",
                "description": "总结 trace 整体执行过程中的优点与缺点，然后写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "trace JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_insights",
                "description": "基于错误和优缺点进行根因分析，生成具有深度的洞察，然后写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "trace JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_optimization_strategy",
                "description": "基于错误、优缺点、洞察生成可实施的优化建议和可微调样例数据，然后写回文件中对应位置。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "trace JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]


    SYSTEM_PROMPT_ReportAgent = """
    你是一位擅长对智能体评估报告进行全局综合分析的专家（ReportAgent），擅长从多个报告中提炼信息并结合案例撰写高质量总结报告。你的任务是根据用户提供的一个报告 JSON 文件路径，使用可用工具一步步撰写高质量总结报告。具体来说，当用户提供一个报告 JSON 文件路径时，你需要首先对这个文件进行结构化解析，将其中的重要字段转化为可直接使用的标准字典格式。在完成结构化处理之后，你会仔细阅读其中的错误信息，基于这些主要错误生成一个精炼的四字核心错误标签，并写入到报告文件中，作为这个报告的标志性结论之一。接着，你会查看这个报告所在的文件夹，判断是否满足要生成总结报告的条件。如果这个条件没有满足，你将立刻停止任务结束工作；如果条件满足，你将继续展开全局分析。你会遍历同一目录下的所有报告文件，统计它们的 key_error 出现次数，形成一个统计表；也会对所有报告的 score.completion_score 进行分数区间统计，形成分数分布表。最重要的是，你会将这些信息整合起来，结合每个报告中的案例细节，撰写一份全面且深入的总结报告。最后，你会对总结报告进行润色。当你完成任务后，你必须直接输出 `finish()`。请开始吧！
    """


    available_tools_ReportAgent = {
        "process_json_file": process_json_file,
        "generate_key_error": lambda **kwargs:
        generate_key_error(client=client_tool, **kwargs),
        "if_generate_conclude_report": if_generate_conclude_report,
        "count_key_error_values": count_key_error_values,
        "count_completion_score_distribution": count_completion_score_distribution,
        "generate_conclude_report": lambda **kwargs:
        generate_conclude_report(client=client_tool, requirement=report_requirement, **kwargs),
        "polish_conclude_report": polish_conclude_report
    }


    tools_ReportAgent = [
        {
            "type": "function",
            "function": {
                "name": "process_json_file",
                "description": "读取报告 JSON 文件并对 score、error、feature、insight 字段进行结构化处理，转成标准字典写回文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "报告 JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_key_error",
                "description": "读取报告 JSON 文件，根据 error 字段生成四字核心错误总结并写入文件。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "报告 JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "if_generate_conclude_report",
                "description": "判断 file_path 所在文件夹下是否恰好包含预定义好个数的 JSON 报告文件，如果是则可以生成总结报告。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "任意一个报告 JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_key_error_values",
                "description": "遍历 file_path 所在文件夹下所有 JSON 文件，统计 key_error 对应的值出现次数，返回 Markdown 格式表格。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "任意一个报告 JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "count_completion_score_distribution",
                "description": "遍历 file_path 所在文件夹下所有 JSON 文件，统计 score.completion_score 的不同分数区间（100, 90-99, 80-89, 60-79, 0-59）的数量和占比，返回 Markdown 格式表格。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "任意一个报告 JSON 文件路径"}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_conclude_report",
                "description": "结合指定目录下所有报告的结构化信息、关键错误统计表、分数分布表，用 LLM 生成详细的 Markdown 总结报告并写入 conclude_report.md。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "任意一个报告 JSON 文件路径"},
                        "key_error": {"type": "string", "description": "Markdown 格式的 key_error 统计表"},
                        "score_distribution": {"type": "string", "description": "Markdown 格式的分数分布统计表"}
                    },
                    "required": ["file_path", "key_error", "score_distribution"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "polish_conclude_report",
                "description": "读取Markdown 总结报告文件路径，对其进行润色，并补充附录内容，返回 conclude_report_polished.md。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_path": {"type": "string", "description": "Markdown 总结报告文件路径"}
                    },
                    "required": ["report_path"]
                }
            }
        }
    ]

    llm_cfg = job_config["llm"]
    report_requirement = job_config["report_requirement"]

    client = LLMAgentToolAPI(
        model=llm_cfg["model"],
        apiKey=llm_cfg["api_key"],
        baseUrl=llm_cfg["base_url"],
    )

    client_tool = LLMAgentAPI(
        model=llm_cfg["model"],
        apiKey=llm_cfg["api_key"],
        baseUrl=llm_cfg["base_url"],
    )

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    total = len(json_files)
    logs = []
    log = make_logger(logs, job_id)
    results = []

    log(f"用户特殊需求： {report_requirement}")
    log(f"共发现 {total} 个 JSON 文件，将逐一处理\n" + "=" * 40)

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        log(f"开始处理文件: {file_name}\n" + "=" * 40)

        result_file = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
        if os.path.exists(result_file):
            log(f"结果已存在，跳过: {result_file}\n" + "=" * 40)
            continue


        ##### StructureAgent #####
        prompt_history = [f"文件路径: {file_path}"]
        tool_name = ''
        observation_str = ""

        for i in range(10):
            log(f"StructureAgent --- 循环 {i + 1} ---\n")
            full_prompt = "\n".join(prompt_history)
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_StructureAgent},
                {'role': 'user', 'content': full_prompt}
            ]
            message, usage = client.generate(messages, tools_StructureAgent)
            thought = message.content
            log(f"模型思考:\n{thought}\n")
            if not message.tool_calls and thought != None:
                log("模型本轮未调用工具")
                if thought.startswith("finish") or "finish(" in thought:
                    log(f"任务完成。")
                    log_trace_step(file_name, "StructureAgent", i, None, None, thought, "任务完成。", usage, False)
                    break
                prompt_history.append(thought)
                log_trace_step(file_name, "StructureAgent", i, None, None, thought, None, usage, False)
                continue
            action = message.tool_calls[0]
            tool_name = action.function.name
            args = json.loads(action.function.arguments)
            log(f"模型输出: {tool_name} {args}\n")
            prompt_history.append(tool_name + str(args))
            if tool_name == "finish" or (thought != None and thought.startswith("finish")):
                log("任务完成。")
                log_trace_step(file_name, "StructureAgent", i, tool_name, args, thought, "任务完成。", usage, False)
                break
            if tool_name in available_tools_StructureAgent:
                try:
                    observation, usage_tool = available_tools_StructureAgent[tool_name](**args)
                except:
                    observation, usage_tool = available_tools_StructureAgent[tool_name](**args)
            else:
                observation = f"错误: 未定义的工具 '{tool_name}'"
                usage_tool = False
            observation_str = f"模型观测: {observation}"
            log(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            log_trace_step(file_name, "StructureAgent", i, tool_name, args, thought, observation_str, usage, usage_tool)
        log(prompt_history)


        ##### InsightAgent #####
        prompt_history = [f"文件路径: {result_file}"]
        tool_name = ''
        observation_str = ""

        for i in range(10):
            log(f"InsightAgent --- 循环 {i + 1} ---\n")
            full_prompt = "\n".join(prompt_history)
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_InsightAgent},
                {'role': 'user', 'content': full_prompt}
            ]
            message, usage = client.generate(messages, tools_InsightAgent)
            thought = message.content
            log(f"模型思考:\n{thought}\n")
            if not message.tool_calls and thought != None:
                log("模型本轮未调用工具")
                if thought.startswith("finish") or "finish(" in thought:
                    log(f"任务完成。")
                    log_trace_step(file_name, "InsightAgent", i, None, None, thought, "任务完成。", usage, False)
                    break
                prompt_history.append(thought)
                log_trace_step(file_name, "InsightAgent", i, None, None, thought, None, usage, False)
                continue
            action = message.tool_calls[0]
            tool_name = action.function.name
            args = json.loads(action.function.arguments)
            log(f"模型输出: {tool_name} {args}\n")
            prompt_history.append(tool_name + str(args))
            if tool_name == "finish" or (thought != None and thought.startswith("finish")):
                log("任务完成。")
                log_trace_step(file_name, "InsightAgent", i, tool_name, args, thought, "任务完成。", usage, False)
                break
            if tool_name in available_tools_InsightAgent:
                observation, usage_tool = available_tools_InsightAgent[tool_name](**args)
            else:
                observation = f"错误: 未定义的工具 '{tool_name}'"
                usage_tool = False
            observation_str = f"模型观测: {observation}"
            log(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            log_trace_step(file_name, "InsightAgent", i, tool_name, args, thought, observation_str, usage, usage_tool)
        log(prompt_history)


        ##### ReportAgent #####
        prompt_history = [f"文件路径: {result_file}"]
        tool_name = ''
        observation_str = ""

        for i in range(10):
            log(f"ReportAgent --- 循环 {i + 1} ---\n")
            full_prompt = "\n".join(prompt_history)
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT_ReportAgent},
                {'role': 'user', 'content': full_prompt}
            ]
            message, usage = client.generate(messages, tools_ReportAgent)
            thought = message.content
            log(f"模型思考:\n{thought}\n")
            if not message.tool_calls and thought != None:
                log("模型本轮未调用工具")
                if thought.startswith("finish") or "finish(" in thought:
                    log(f"任务完成。")
                    log_trace_step(file_name, "ReportAgent", i, None, None, thought, "任务完成。", usage, False)
                    break
                prompt_history.append(thought)
                log_trace_step(file_name, "ReportAgent", i, None, None, thought, None, usage, False)
                continue
            action = message.tool_calls[0]
            tool_name = action.function.name
            args = json.loads(action.function.arguments)
            log(f"模型输出: {tool_name} {args}\n")
            prompt_history.append(tool_name + str(args))
            if tool_name == "finish" or (thought != None and thought.startswith("finish")):
                log("任务完成。")
                log_trace_step(file_name, "ReportAgent", i, tool_name, args, thought, "任务完成。", usage, False)
                break
            if tool_name in available_tools_ReportAgent:
                observation, usage_tool = available_tools_ReportAgent[tool_name](**args)
            else:
                observation = f"错误: 未定义的工具 '{tool_name}'"
                usage_tool = False
            observation_str = f"模型观测: {observation}"
            log(f"{observation_str}\n" + "=" * 40)
            prompt_history.append(observation_str)
            log_trace_step(file_name, "ReportAgent", i, tool_name, args, thought, observation_str, usage, usage_tool)
        log(prompt_history)


        results.append({
            "file": file_name,
            "output_path": result_file
        })

    key_error,_ = count_key_error_values(result_file)
    score_distribution,_ = count_completion_score_distribution(result_file)
    final_report = generate_conclude_report(result_file, key_error, score_distribution, client_tool, report_requirement, flag=True)
    polish_report = polish_conclude_report(Path(os.path.dirname(result_file)) / "conclude_report.md")
    
    log('\n\n###### FINAL REPORT ######\n\n')
    log(final_report[0] + '\n' + polish_report[0])

    return {
        "total_files": total,
        "logs": logs,
        "results": results
    }



def modify_trace_folder(folder_path: str, job_id: str, job_config):
    llm_cfg = job_config["llm"]
    report_requirement = job_config["report_requirement"]

    client = LLMAgentToolAPI(
        model=llm_cfg["model"],
        apiKey=llm_cfg["api_key"],
        baseUrl=llm_cfg["base_url"],
    )

    client_tool = LLMAgentAPI(
        model=llm_cfg["model"],
        apiKey=llm_cfg["api_key"],
        baseUrl=llm_cfg["base_url"],
    )

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    total = len(json_files)
    logs = []
    log = make_logger(logs, job_id)
    results = []

    log(f"用户修改需求： {report_requirement}")
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        result_file = file_path.replace("/TraceBench/", "/TraceBenchTMP/")
        results.append({
            "file": file_name,
            "output_path": result_file
        })
    key_error,_ = count_key_error_values(result_file)
    score_distribution,_ = count_completion_score_distribution(result_file)

    folder_path = folder_path.replace("/TraceBench/", "")
    folder_path = folder_path.replace("/TraceBench", "")
    history_path = os.path.join(folder_path, "history.json")
    history = load_or_init_history(folder_path)
    new_version = get_next_version(history)

    final_report = modify_conclude_report(history, result_file, key_error, score_distribution, client_tool, report_requirement, flag=True)
    polish_report = polish_conclude_report(Path(os.path.dirname(result_file)) / "conclude_report.md")

    conclude_md_path = Path(Path(os.path.dirname(result_file)) / "conclude_report.md")

    with open(conclude_md_path, "r", encoding="utf-8") as f:
            new_report = f.read()

    history[new_version] = {
        "requirement": report_requirement,
        "report": new_report
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    log('\n\n###### FINAL REPORT ######\n\n')
    log(final_report[0] + '\n' + polish_report[0])

    return {
        "total_files": total,
        "logs": logs,
        "results": results
    }