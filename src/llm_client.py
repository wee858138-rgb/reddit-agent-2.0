# -*- coding: utf-8 -*-
"""LLM 调用封装：兼容 OpenAI / DeepSeek / 第三方中转。"""

from __future__ import annotations

from openai import OpenAI


def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """发起一次 Chat Completion，返回助手文本（已 strip）。

    签名与初代 ``app.call_llm`` 保持一致，便于最小侵入迁移。
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()
