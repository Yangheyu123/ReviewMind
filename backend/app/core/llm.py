from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings
from app.core.llm_usage import latency_timer, record

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


@dataclass
class ToolCallItem:
    """单次工具调用（OpenAI 原生 tool_calls 元素）。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResponse:
    """chat_with_tools 的返回：完整 message 结构（含 tool_calls）。

    - ``content``: 模型的文本输出（可能为 None，当模型只调工具不给文本时）
    - ``tool_calls``: 原生工具调用列表；为空表示模型已给出最终文本结论
    - ``finish_reason``: stop（正常结束）/ tool_calls（要求调工具）/ length 等
    - ``reasoning_content``: 推理模型（如 deepseek-v4-flash）的思维链。
      推理模型的多轮协议要求：下一轮回灌 assistant 消息时必须带回此字段，
      否则 API 报 400（"reasoning_content must be passed back"）。
    """

    content: str | None
    tool_calls: list[ToolCallItem] = field(default_factory=list)
    finish_reason: str = ""
    reasoning_content: str | None = None
    # token 用量（prompt/completion/total），预算闸依赖；mock 或厂商未回传时为 None
    usage: dict[str, int] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        if not self.usage:
            return 0
        return int(self.usage.get("total_tokens") or (self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)))


async def aggregate_anthropic_sse(resp: httpx.Response) -> dict[str, Any]:
    """聚合 Anthropic SSE 事件流为与非流式响应等价的 message dict。

    处理 text/thinking/tool_use（input_json_delta 增量拼接）与 usage/stop_reason。
    """
    message: dict[str, Any] = {"content": [], "stop_reason": None, "usage": {}}
    blocks: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "message_start":
            m = event.get("message", {}) or {}
            message["usage"].update(m.get("usage") or {})
            message.setdefault("model", m.get("model"))
        elif etype == "content_block_start":
            idx = int(event.get("index", 0))
            block = dict(event.get("content_block") or {})
            if block.get("type") == "tool_use":
                block["input"] = {}
                block["_json"] = ""
            if block.get("type") == "text":
                block["text"] = block.get("text") or ""
            blocks[idx] = block
            order.append(idx)
        elif etype == "content_block_delta":
            idx = int(event.get("index", 0))
            delta = event.get("delta") or {}
            block = blocks.get(idx)
            if block is None:
                continue
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = block.get("text", "") + (delta.get("text") or "")
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + (delta.get("thinking") or "")
            elif dtype == "signature_delta":
                block["signature"] = block.get("signature", "") + (delta.get("signature") or "")
            elif dtype == "input_json_delta":
                block["_json"] = block.get("_json", "") + (delta.get("partial_json") or "")
        elif etype == "content_block_stop":
            idx = int(event.get("index", 0))
            block = blocks.get(idx)
            if block is not None and block.get("type") == "tool_use":
                try:
                    block["input"] = json.loads(block.pop("_json") or "{}")
                except json.JSONDecodeError:
                    block["input"] = {}
                    block.pop("_json", None)
        elif etype == "message_delta":
            message["stop_reason"] = event.get("delta", {}).get("stop_reason")
            message["usage"].update(event.get("usage") or {})
        elif etype == "message_stop":
            break
        elif etype == "error":
            raise LLMClientError(f"anthropic stream error: {event}")
    message["content"] = [blocks[i] for i in sorted(blocks)]
    return message


class LLMClientError(Exception):
    """LLM 调用失败。"""


class LLMClient:
    """OpenAI-compatible LLM 客户端，支持 mock 模式。"""

    def __init__(self) -> None:
        self._api_key = settings.llm_api_key
        self._api_base = settings.llm_api_base.rstrip("/")
        self._timeout = settings.llm_timeout_seconds or _DEFAULT_TIMEOUT
        self._mock_mode = settings.llm_mock_mode
        self._last_anthropic_latency_ms = 0

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @property
    def _is_anthropic(self) -> bool:
        """Anthropic 协议：显式配置优先，否则按 base_url 自动识别。"""
        proto = settings.llm_api_protocol.strip().lower()
        if proto in ("openai", "anthropic"):
            return proto == "anthropic"
        return "/api/anthropic" in self._api_base

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        if self._mock_mode or not self.is_configured:
            reason = "mock_mode enabled" if self._mock_mode else "no API key configured"
            logger.info("[LLM] MOCK mode active (%s), skipping real call", reason)
            return self._mock_response(messages)

        if self._is_anthropic:
            return await self._chat_anthropic(messages, model=model, temperature=temperature)

        resolved_model = model or settings.llm_model_summary
        api_url = f"{self._api_base}/chat/completions"
        logger.info(
            "[LLM] Calling API | url=%s model=%s temperature=%s timeout=%ss",
            api_url, resolved_model, temperature, self._timeout,
        )
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
            logger.info("[LLM] response_format=%s", response_format)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    api_url,
                    headers=headers,
                    json=payload,
                )
                logger.info("[LLM] Response status=%s content_length=%s", resp.status_code, len(resp.content))
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.info("[LLM] Success | model=%s input_tokens=%s output_tokens=%s",
                    data.get("model", "?"),
                    data.get("usage", {}).get("prompt_tokens", "?"),
                    data.get("usage", {}).get("completion_tokens", "?"),
                )
                return content
        except httpx.TimeoutException as exc:
            logger.error("[LLM] TIMEOUT after %ss | url=%s", self._timeout, api_url)
            raise LLMClientError(f"LLM request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.error("[LLM] HTTP_ERROR status=%s url=%s body=%s", exc.response.status_code, api_url, body)
            raise LLMClientError(f"LLM HTTP {exc.response.status_code}: {body}") from exc
        except Exception as exc:
            logger.error("[LLM] UNEXPECTED_ERROR type=%s msg=%s", type(exc).__name__, exc)
            raise LLMClientError(f"LLM call failed: {exc}") from exc

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        tool_choice: str = "auto",
    ) -> ToolCallResponse:
        """带原生工具调用的 chat（OpenAI Function Calling 协议）。

        与 ``chat()`` 的区别：
        - 透传 ``tools`` / ``tool_choice`` 字段给 API；
        - 返回完整 message 结构（``ToolCallResponse``，含 ``tool_calls``），
          而非只返回字符串——这是 Agent Loop 多轮工具调用的前提。

        现有 4 个 agent 仍用 ``chat()``，本方法仅供 Planner 等 Agent Loop 组件使用。
        """
        if self._mock_mode or not self.is_configured:
            reason = "mock_mode enabled" if self._mock_mode else "no API key configured"
            logger.info("[LLM] MOCK mode active (%s), chat_with_tools returns empty tool_calls", reason)
            return self._mock_tool_response(messages)

        if self._is_anthropic:
            return await self._chat_with_tools_anthropic(
                messages, tools=tools, model=model, temperature=temperature, tool_choice=tool_choice,
            )

        resolved_model = model or settings.llm_model_summary
        api_url = f"{self._api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "[LLM] chat_with_tools | url=%s model=%s tools=%d tool_choice=%s",
            api_url, resolved_model, len(tools), tool_choice,
        )
        _timer = latency_timer()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content")
                finish_reason = str(choice.get("finish_reason", ""))
                tool_calls = self._parse_native_tool_calls(message.get("tool_calls"))
                reasoning_content = message.get("reasoning_content")
                logger.info(
                    "[LLM] chat_with_tools OK | finish=%s tool_calls=%d content_len=%s reasoning=%s",
                    finish_reason, len(tool_calls), len(content) if content else 0,
                    bool(reasoning_content),
                )
                usage_raw = data.get("usage")
                usage = None
                if isinstance(usage_raw, dict):
                    usage = {
                        "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
                        "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
                        "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
                    }
                record(resolved_model, protocol="openai",
                       prompt_tokens=(usage or {}).get("prompt_tokens", 0),
                       completion_tokens=(usage or {}).get("completion_tokens", 0),
                       total_tokens=(usage or {}).get("total_tokens", 0),
                       latency_ms=_timer.snapshot_ms())
                return ToolCallResponse(
                    content=content, tool_calls=tool_calls, finish_reason=finish_reason,
                    reasoning_content=reasoning_content, usage=usage,
                )
        except (httpx.TimeoutException, httpx.HTTPStatusError, Exception) as exc:
            record(resolved_model, protocol="openai", latency_ms=_timer.snapshot_ms(),
                   status="error", error=str(exc)[:300])
            raise

    @staticmethod
    def _parse_native_tool_calls(raw: Any) -> list[ToolCallItem]:
        """把 OpenAI 原生 tool_calls 列表解析为 ToolCallItem。

        容错：arguments 是 JSON 字符串，解析失败则降级为空 dict。
        """
        if not isinstance(raw, list):
            return []
        items: list[ToolCallItem] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            func = entry.get("function", {})
            if not isinstance(func, dict):
                continue
            name = str(func.get("name", ""))
            if not name:
                continue
            args_raw = func.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except (json.JSONDecodeError, TypeError):
                args = {}
            items.append(ToolCallItem(id=str(entry.get("id", "")), name=name, arguments=args))
        return items

    # ------------------------------------------------------------------
    # Anthropic 兼容协议适配（GLM Coding Plan 订阅额度走 /api/anthropic）
    # ------------------------------------------------------------------

    _ANTHROPIC_MAX_TOKENS = 4096

    @staticmethod
    def _oa_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """内部 OpenAI 格式消息 → Anthropic messages 协议（system 提取、tool 消息转换）。"""
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_parts.append(str(m["content"]))
                continue
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("reasoning_content"):
                    blocks.append({"type": "thinking", "thinking": str(m["reasoning_content"]),
                                   "signature": m.get("thinking_signature", "")})
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for tc in m.get("tool_calls") or []:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = func.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args or "{}")
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = raw_args or {}
                    blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                                   "name": func.get("name", ""), "input": args})
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
                continue
            if role == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": str(m.get("content", "")),
                }]})
                continue
            out.append({"role": "user", "content": [{"type": "text", "text": str(m.get("content", ""))}]})
        # 合并相邻同角色消息（tool_result 连续时必须合并为单条 user）
        merged: list[dict[str, Any]] = []
        for m in out:
            if merged and merged[-1]["role"] == m["role"] == "user":
                merged[-1]["content"].extend(m["content"])
            else:
                merged.append(m)
        return "\n\n".join(system_parts), merged

    @staticmethod
    def _oa_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for t in tools:
            fn = t.get("function", t)
            converted.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted

    @staticmethod
    def _anthropic_finish(stop_reason: str) -> str:
        return {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(stop_reason, stop_reason or "stop")

    def _anthropic_response_to_tcr(self, data: dict[str, Any]) -> ToolCallResponse:
        text_parts: list[str] = []
        reasoning: str | None = None
        tool_calls: list[ToolCallItem] = []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                reasoning = block.get("thinking")
            elif btype == "tool_use":
                tool_calls.append(ToolCallItem(
                    id=str(block.get("id", "")), name=str(block.get("name", "")),
                    arguments=block.get("input") or {},
                ))
        usage_raw = data.get("usage") or {}
        prompt_t = int(usage_raw.get("input_tokens", 0) or 0)
        completion_t = int(usage_raw.get("output_tokens", 0) or 0)
        usage = {"prompt_tokens": prompt_t, "completion_tokens": completion_t,
                 "total_tokens": prompt_t + completion_t}
        return ToolCallResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            finish_reason=self._anthropic_finish(str(data.get("stop_reason", ""))),
            reasoning_content=reasoning,
            usage=usage,
        )

    async def _anthropic_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._api_base}/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        _timer = latency_timer()
        try:
            # 部分代理端点强制 SSE（忽略 stream:false），统一走流式并聚合为
            # 与非流式等价的 message 结构
            stream_payload = {**payload, "stream": True}
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, headers=headers, json=stream_payload) as resp:
                    resp.raise_for_status()
                    data = await aggregate_anthropic_sse(resp)
            self._last_anthropic_latency_ms = _timer.snapshot_ms()
            return data
        except Exception as exc:
            record(str(payload.get("model", "")), protocol="anthropic",
                   latency_ms=_timer.snapshot_ms(), status="error", error=str(exc)[:300])
            raise LLMClientError(f"LLM call failed: {exc}") from exc

    async def _chat_anthropic(self, messages, *, model=None, temperature=0.1) -> str:
        system, conv = self._oa_messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": model or settings.llm_model_summary,
            "max_tokens": self._ANTHROPIC_MAX_TOKENS,
            "temperature": temperature,
            "messages": conv,
        }
        if system:
            payload["system"] = system
        data = await self._anthropic_post(payload)
        tcr = self._anthropic_response_to_tcr(data)
        record(str(payload.get("model", "")), protocol="anthropic",
               prompt_tokens=tcr.usage.get("prompt_tokens", 0) if tcr.usage else 0,
               completion_tokens=tcr.usage.get("completion_tokens", 0) if tcr.usage else 0,
               total_tokens=tcr.usage.get("total_tokens", 0) if tcr.usage else 0,
               latency_ms=getattr(self, "_last_anthropic_latency_ms", 0))
        return tcr.content or ""

    async def _chat_with_tools_anthropic(
        self, messages, *, tools, model=None, temperature=0.0, tool_choice="auto",
    ) -> ToolCallResponse:
        system, conv = self._oa_messages_to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": model or settings.llm_model_review,
            "max_tokens": self._ANTHROPIC_MAX_TOKENS,
            "temperature": temperature,
            "messages": conv,
            "tools": self._oa_tools_to_anthropic(tools),
        }
        if system:
            payload["system"] = system
        if tool_choice == "required":
            payload["tool_choice"] = {"type": "any"}
        logger.info("[LLM] anthropic chat_with_tools | model=%s tools=%d", payload["model"], len(tools))
        data = await self._anthropic_post(payload)
        tcr = self._anthropic_response_to_tcr(data)
        record(str(payload.get("model", "")), protocol="anthropic",
               prompt_tokens=tcr.usage.get("prompt_tokens", 0) if tcr.usage else 0,
               completion_tokens=tcr.usage.get("completion_tokens", 0) if tcr.usage else 0,
               total_tokens=tcr.usage.get("total_tokens", 0) if tcr.usage else 0,
               latency_ms=getattr(self, "_last_anthropic_latency_ms", 0))
        logger.info("[LLM] anthropic OK | finish=%s tool_calls=%d tokens=%s",
                    tcr.finish_reason, len(tcr.tool_calls), tcr.usage)
        return tcr

    @staticmethod
    def _mock_tool_response(messages: list[dict[str, Any]]) -> ToolCallResponse:
        """mock 模式：返回无 tool_calls 的降级响应（视为模型直接给结论）。"""
        last_msg = messages[-1].get("content", "") if messages else ""
        return ToolCallResponse(
            content=f"Mock LLM（无真实工具调用）：{last_msg[:80]}",
            tool_calls=[],
            finish_reason="stop",
        )

    @staticmethod
    def _mock_response(messages: list[dict[str, str]]) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        return json.dumps({
            "summary": "Mock LLM: 分析完成，未发现重大风险。",
            "findings": [],
            "risk_level": "LOW",
            "source": "mock_llm",
            "prompt_preview": last_msg[:100],
        }, ensure_ascii=False)


llm_client = LLMClient()