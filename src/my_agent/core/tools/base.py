"""
定义  一个工具长什么样
任何说我是一个工具的类，必须有三个属性
(name、description、input_schema)和一个方法(invoke)。
少一个就别想用。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ToolResult(执行结果)
@dataclass
class ToolResult:
    content: str  # 工具的输出内容
    is_error: bool = False  # 执行成功还是失败
    error_type: str | None = None  # 失败原因：runtime_error | timeout | schema_error


# BaseTool(抽象接口)
class BaseTool(ABC):
    name: str  # 工具名，比如 "read_file"
    description: str  # 给 LLM 看的功能描述
    input_schema: dict[str, object]  # JSON Schema 格式的参数定义

    @abstractmethod
    async def invoke(self, params: dict[str, object]) -> ToolResult: ...
