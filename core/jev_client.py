"""Jev System One HTTP 客户端。

本客户端仅用于影子评测。没有配置 API Key 时不会发送任何网络请求。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class JevClient:
    """调用 TypeSafe System One API，并返回原始结构化答案。"""

    def __init__(
        self,
        api_key: str,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai",
        timeout_s: float = 3.0,
    ):
        self._api_key = api_key.strip()
        self.model = model.strip() or "jev-latest"
        self._client: Optional[httpx.AsyncClient] = None
        if self._api_key:
            self._client = httpx.AsyncClient(
                base_url=base_url.rstrip("/"),
                timeout=httpx.Timeout(timeout_s),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def evaluate(
        self,
        *,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Jev 未配置")

        response = await self._client.post(
            "/v1/systemone",
            json={
                "state": state,
                "model": self.model,
                "questions": questions,
            },
        )
        response.raise_for_status()
        payload = response.json()
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Jev 响应缺少 answers")
        return payload

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
