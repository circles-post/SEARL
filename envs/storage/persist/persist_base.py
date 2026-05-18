from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import json
import gzip
from datetime import datetime

class PersistBase(ABC):
    @abstractmethod
    def save(self, data: Dict[str, Any], method_name: str) -> str:
        """保存数据到磁盘
        
        Args:
            data: 要保存的数据字典
            method_name: 方法名称，用于生成文件名
            
        Returns:
            str: 保存的文件路径
        """
        pass
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """从磁盘加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据字典
        """
        pass
    
    @abstractmethod
    def merge(self, file_paths: list[str], output_path: str) -> str:
        """合并多个缓存文件
        
        Args:
            file_paths: 要合并的文件路径列表
            output_path: 输出文件路径
            
        Returns:
            str: 合并后的文件路径
        """
        pass
