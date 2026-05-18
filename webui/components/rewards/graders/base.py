from abc import ABC, abstractmethod
from typing import Dict, Type, Any


class BaseGrader(ABC):
    """奖赏评分器基类"""
    
    @abstractmethod
    def grade(self, extracted_answer, **kwargs) -> float:
        """评分方法
        
        Args:
            extracted_answer: 提取的答案
            **kwargs: 可选参数，可能包含：
                - data_source: 数据源
                - solution_str: 解题答案
                - ground_truth: 标准答案
                - extra_info: 额外信息
            
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
    def required_attributes(self) -> Dict[str, Any]:
        """评分器需要的属性"""
        return {}
    
    @property
    @abstractmethod
    def description(self) -> str:
        """评分器描述"""
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
        cls._registry[grader_class.name] = grader_class
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
        return {name: grader.description for name, grader in cls._registry.items()}