import asyncio
import warnings
from typing import Dict
from contextlib import asynccontextmanager

class ConcurrencyLimiter:
    """简单的并发限制器，修复事件循环切换导致的死锁问题"""

    def __init__(self, global_limit: int = 100):
        self._loop = None
        self._global_limit = global_limit
        # 工具级别的信号量
        self._tool_semaphores: Dict[str, asyncio.Semaphore] = {}
        # 全局信号量
        self._global_semaphore = None
        self._semaphore_waiters = 0  # 跟踪等待者数量

    def _ensure_loop(self):
        """
        确保信号量绑定当前事件循环。
        如果切换到新的事件循环（例如多次 asyncio.run），需要重建信号量。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # 创建新的事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        if self._loop is not loop:
            # 检测旧信号量是否有等待者
            if self._semaphore_waiters > 0:
                warnings.warn(
                    f"Event loop changed while {self._semaphore_waiters} tasks are waiting on semaphore. "
                    f"This may cause deadlocks. Consider finishing all async tasks before switching loops.",
                    RuntimeWarning
                )

            # 切换到新循环，重建信号量
            self._loop = loop
            self._global_semaphore = asyncio.Semaphore(self._global_limit)
            self._tool_semaphores = {}
            self._semaphore_waiters = 0

    def get_tool_semaphore(self, tool_name: str, limit: int = 10) -> asyncio.Semaphore:
        """获取或创建工具的信号量"""
        self._ensure_loop()
        if tool_name not in self._tool_semaphores:
            self._tool_semaphores[tool_name] = asyncio.Semaphore(limit)
        return self._tool_semaphores[tool_name]

    @asynccontextmanager
    async def limit(self, tool_name: str):
        """使用上下文管理器控制并发"""
        self._ensure_loop()
        sem = self.get_tool_semaphore(tool_name)

        # 增加等待者计数
        self._semaphore_waiters += 1
        try:
            async with self._global_semaphore:
                async with sem:
                    yield
        finally:
            # 减少等待者计数
            self._semaphore_waiters -= 1
