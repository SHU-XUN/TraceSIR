import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict


load_dotenv()


class LLMAgentAPI:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 600))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 1) -> str:
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            print("✅ 大语言模型响应成功:")
            answer = response.choices[0].message.content
            usage_info = response.usage
            return answer, usage_info
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None, False


class LLMAgentToolAPI:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 600))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def generate(self, messages: List[Dict[str, str]], tools, temperature: float = 1) -> str:
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temperature,
                stream=False,
            )
            print("✅ 大语言模型响应成功:")
            message = response.choices[0].message
            usage_info = response.usage
            return message, usage_info
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None, False

