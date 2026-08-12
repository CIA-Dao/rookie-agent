from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator

from my_agent.core.session.model import SessionMode


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str


class PongResult(BaseModel):
    server_version: str
    uptime_ms: int
    received_at: str


# CLI 发给 server 的请求参数结构
class AgentRunCommand(BaseModel):
    type: Literal["agent.run"] = "agent.run"
    goal: str
    workspace_root: str = ""


# 创建session
class SessionCreateCommand(BaseModel):
    type: Literal["session.create"] = "session.create"
    mode: SessionMode = "chat"
    title: str = ""
    workspace_root: str = ""
    client_type: Literal["tui", "cli", "unknown"] = "unknown"


# 创建session响应
class SessionCreateResult(BaseModel):
    session_id: str
    status: str
    title: str


# 发送session
class SessionSendMessageCommand(BaseModel):
    type: Literal["session.send_message"] = "session.send_message"
    session_id: str
    content: str


# 发送session响应
class SessionSendMessageResult(BaseModel):
    session_id: str
    run_id: str


# 获取session记录
class SessionGetHistoryCommand(BaseModel):
    type: Literal["session.get_history"] = "session.get_history"
    session_id: str


# session记录返回
class SessionGetHistoryResult(BaseModel):
    session_id: str
    messages: list[dict[str, Any]]


# 关闭session
class SessionCloseCommand(BaseModel):
    type: Literal["session.close"] = "session.close"
    session_id: str


# 关闭session响应
class SessionCloseResult(BaseModel):
    session_id: str
    status: str


class SessionHeartbeatCommand(BaseModel):
    type: Literal["session.heartbeat"] = "session.heartbeat"
    session_id: str


class SessionHeartbeatResult(BaseModel):
    session_id: str
    status: Literal["active"] = "active"


# server 返回给 CLI 的结果结构
class AgentRunResult(BaseModel):
    run_id: str


# 告诉 server：用户对某个 tool_use_id 做了什么决定  CLI -> server
class PermissionRespondCommand(BaseModel):
    type: Literal["permission.respond"] = "permission.respond"
    tool_use_id: str
    decision: str


# 告诉 CLI：这个决定已经被 server 接收 server -> CLI
class PermissionRespondResult(BaseModel):
    ok: bool = True


# 订阅哪些事件
class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str]
    scope: str = "global"
    replay_from_run: str | None = None


# 手动压缩
class SessionCompactCommand(BaseModel):
    type: Literal["session.compact"] = "session.compact"
    session_id: str
    focus: str = ""


# 手动压缩返回
class SessionCompactResult(BaseModel):
    summary_tokens: int
    saved_tokens: int


class EventSubscribeResult(BaseModel):
    subscription_id: str
    replayed_count: int = 0


Command = Annotated[
    PingCommand
    | AgentRunCommand
    | EventSubscribeCommand
    | SessionCreateCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionCloseCommand
    | SessionHeartbeatCommand
    | PermissionRespondCommand
    | SessionCompactCommand,
    Discriminator("type"),
]
