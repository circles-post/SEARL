from abc import ABC, abstractmethod
from typing import Dict, Type, Any

class BaseGrader(ABC):
    """奖赏评分器基类"""
    
    @abstractmethod
    def grade(self, prediction: Any, reference: Any) -> float:
        """评分方法
        
        Args:
            prediction: 预测值
            reference: 参考值
            
        Returns:
            float: 评分结果（0-1之间）
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """评分器名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """评分器描述"""
        pass

    @property
    @abstractmethod
    def gt_required(self) -> bool:
        """是否需要gt"""
        pass

class GraderRegistry:
    """评分器注册器"""
    
    _registry: Dict[str, Type[BaseGrader]] = {}
    
    @classmethod
    def register(cls, grader_class: Type[BaseGrader]) -> Type[BaseGrader]:
        """注册评分器
        
        Args:
            grader_class: 评分器类
            
        Returns:
            注册的评分器类
        """
        # 创建一个实例来获取名称和描述
        instance = grader_class()
        name = instance.name
        cls._registry[name] = grader_class
        return grader_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseGrader]:
        """获取评分器
        
        Args:
            name: 评分器名称
            
        Returns:
            评分器类
        """
        if name not in cls._registry:
            raise KeyError(f"未找到名为 {name} 的评分器")
        return cls._registry[name]
    
    @classmethod
    def list_graders(cls) -> Dict[str, str]:
        """列出所有已注册的评分器
        
        Returns:
            Dict[str, str]: 评分器名称和描述的字典
        """
        return {
            name: grader_class().description 
            for name, grader_class in cls._registry.items()
        }

# 导入所有评分器以确保它们被注册
from .graders import *

# 导出所有需要的类和函数
__all__ = ['BaseGrader', 'GraderRegistry']
