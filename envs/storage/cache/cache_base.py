from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from enum import Enum

class CacheMode(Enum):
    SINGLE = "single"  # 单机模式
    MULTI = "multi"    # 分布式模式

class EvictionPolicy(Enum):
    LRU = "lru"        # 最近最少使用
    LFU = "lfu"        # 最不经常使用
    FIFO = "fifo"      # 先进先出
    TTL = "ttl"        # 基于时间过期

class CacheBase(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示永不过期
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """删除缓存值"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    @abstractmethod
    def has(self, key: str) -> bool:
        """检查键是否存在"""
        pass

    @abstractmethod
    def get_mode(self) -> CacheMode:
        """获取缓存模式"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        pass

    @abstractmethod
    def get_eviction_policy(self) -> EvictionPolicy:
        """获取缓存淘汰策略"""
        pass

    @abstractmethod
    def set_eviction_policy(self, policy: EvictionPolicy) -> None:
        """设置缓存淘汰策略"""
        pass
