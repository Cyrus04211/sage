"""LLM API 客户端封装 (OpenAI 兼容协议)。

配置来自项目根目录 .env: OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL / 代理变量。
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV)

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()   # 自动读取 OPENAI_BASE_URL / OPENAI_API_KEY / 代理
    return _client


def chat(prompt: str, system: str | None = None, model: str | None = None,
         temperature: float = 0.7, max_tokens: int = 4096) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client().chat.completions.create(
        model=model or os.environ["LLM_MODEL"],
        messages=msgs, temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content


def chat_json(prompt: str, system: str | None = None, retries: int = 3,
              **kw) -> dict:
    """要求 JSON 输出; 解析失败自动重试。"""
    err = None
    for i in range(retries):
        p = prompt if i == 0 else (
            prompt + f"\n\nPrevious output failed to parse ({err}). "
                     "Return ONLY valid JSON, no markdown fence, no commentary.")
        text = chat(p, system=system, **kw)
        try:
            return _parse_json(text)
        except Exception as e:
            err = e
    raise RuntimeError(f"LLM JSON 解析连续失败 {retries} 次: {err}")


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)
