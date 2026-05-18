#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse
import random
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from easydict import EasyDict
import pandas as pd
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("async_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator import get_generator

# 示例查询模板列表
QUERY_TEMPLATES = [
    "如何在{}实现{}?",
    "{}的最佳实践是什么?",
    "请解释{}的工作原理",
    "如何优化{}的性能?",
    "{}和{}有什么区别?",
    "请分析{}的优缺点",
    "如何解决{}中的常见问题?",
    "{}的最新发展趋势是什么?",
    "在{}项目中如何集成{}?",
    "{}的安全最佳实践是什么?"
]

# 技术关键词
TECH_KEYWORDS = [
    "Python", "Java", "JavaScript", "Go", "Rust", "C++", "Docker", 
    "Kubernetes", "微服务", "分布式系统", "机器学习", "深度学习", "并行计算",
    "数据库优化", "Redis", "MongoDB", "MySQL", "PostgreSQL", "Kafka",
    "消息队列", "负载均衡", "高并发", "CI/CD", "DevOps", "云原生",
    "React", "Vue", "Angular", "前端框架", "后端架构", "API设计",
    "RESTful", "GraphQL", "gRPC", "WebSocket", "网络安全", "加密算法"
]

def generate_random_query():
    """生成随机查询"""
    template = random.choice(QUERY_TEMPLATES)
    if template.count("{}") == 1:
        return template.format(random.choice(TECH_KEYWORDS))
    else:
        keywords = random.sample(TECH_KEYWORDS, 2)
        return template.format(keywords[0], keywords[1])

def generate_example_data(num_samples=100):
    """生成示例数据集"""
    data = []
    for _ in range(num_samples):
        query = generate_random_query()
        data.append({
            "query": query,
            "address": random.choice(["北京", "上海", "广州", "深圳", "杭州"])
        })
    return data

class AsyncGeneratorPool:
    """异步生成器池，用于并行处理生成任务"""
    
    def __init__(self, 
                model_config: Dict[str, Any],
                concurrency_limit: int = 10,
                timeout: float = 60.0):
        """
        初始化异步生成器池
        
        Args:
            model_config: 模型配置参数
            concurrency_limit: 并发限制数量
            timeout: 请求超时时间(秒)
        """
        self.model_config = model_config
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        self.generator = get_generator('api')(config=EasyDict(model_config))
        self.semaphore = None
        
        # 系统提示模板
        self.system_prompt = """你是搜索引擎助手。用户会给你一个查询，你需要改写成更好的搜索问题。
        
请遵循以下步骤:
1. 仔细分析用户查询的核心意图
2. 考虑如何更清晰、更精确地表达这个意图
3. 生成3个不同的搜索问题，每个问题应该更具体、更精确
4. 使用简洁的语言，避免不必要的词语

最后，使用以下格式返回结果:
<think>
这里写你的分析过程
</think>

<questions>
```json
{
  "改写结果": ["问题1", "问题2", "问题3"]
}
```
</questions>
"""
        
    async def process_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个生成任务"""
        async with self.semaphore:
            start_time = time.time()
            try:
                # 准备输入
                user_message = f"请帮我改写以下搜索查询: {item['query']}"
                address_info = f"我当前在{item['address']}"
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message + "\n" + address_info}
                ]
                
                # 使用事件循环执行可能阻塞的API调用
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.generator.generate(input_data=messages, temperature=0.7)
                )
                
                return {
                    "query": item["query"],
                    "address": item["address"],
                    "response": response,
                    "time_taken": time.time() - start_time
                }
                
            except Exception as e:
                logger.error(f"处理任务出错: {str(e)}")
                return {
                    "query": item["query"],
                    "address": item["address"],
                    "response": f"ERROR: {str(e)}",
                    "time_taken": time.time() - start_time
                }
    
    async def process_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行处理多个生成任务"""
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        # 创建任务列表
        tasks = [self.process_single_item(item) for item in items]
        
        # 并行执行任务并显示进度条
        results = await async_tqdm.gather(*tasks, desc="处理中", total=len(tasks))
        
        return results

async def run_async_processing(args):
    """执行异步并行处理"""
    # 配置模型
    model_config = {
        'api_method': 'local',
        'port': args.port,
        'model_name': args.model_name
    }
    
    # 加载或生成输入数据
    if args.input_file:
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            items = df.to_dict('records')
        elif args.input_file.endswith('.json'):
            with open(args.input_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
        else:
            logger.error(f"不支持的文件格式: {args.input_file}")
            return
        logger.info(f"从文件加载了 {len(items)} 个样本")
    else:
        items = generate_example_data(args.num_samples)
        logger.info(f"生成了 {len(items)} 个随机样本")
        
    # 限制处理样本数量
    if args.limit > 0 and len(items) > args.limit:
        items = items[:args.limit]
        logger.info(f"限制处理样本数量为 {args.limit}")
    
    # 创建异步生成器池
    pool = AsyncGeneratorPool(
        model_config=model_config,
        concurrency_limit=args.concurrency,
        timeout=args.timeout
    )
    
    # 开始计时
    start_time = time.time()
    
    # 并行处理所有项目
    results = await pool.process_items(items)
    
    # 计算总耗时
    total_time = time.time() - start_time
    logger.info(f"总处理时间: {total_time:.2f}秒，平均每个样本 {total_time/len(items):.2f}秒")
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    logger.info(f"结果已保存到 {args.output}")
    
    # 计算统计信息
    success_count = df['response'].apply(lambda x: not str(x).startswith('ERROR:')).sum()
    logger.info(f"成功率: {success_count}/{len(df)} ({success_count/len(df)*100:.2f}%)")
    
    # 如果有错误，显示前几个错误
    errors = df[df['response'].apply(lambda x: str(x).startswith('ERROR:'))]
    if not errors.empty:
        logger.warning(f"检测到 {len(errors)} 个错误，显示前 3 个:")
        for i, (_, row) in enumerate(errors.iloc[:3].iterrows()):
            logger.warning(f"错误 {i+1}: {row['query']} - {row['response']}")

def main():
    parser = argparse.ArgumentParser(description='异步并行生成器测试程序')
    
    # 基本参数
    parser.add_argument('--num_samples', type=int, default=100, help='随机生成样本数量')
    parser.add_argument('--model_name', type=str, default="local", help='模型名称')
    parser.add_argument('--port', type=int, default=8080, help='API端口')
    parser.add_argument('--output', type=str, default='async_results.csv', help='输出文件路径')
    parser.add_argument('--input_file', type=str, default=None, help='输入文件路径(CSV或JSON)')
    parser.add_argument('--limit', type=int, default=0, help='处理样本的最大数量(0表示不限制)')
    
    # 并发参数
    parser.add_argument('--concurrency', type=int, default=10, help='最大并发请求数')
    parser.add_argument('--timeout', type=float, default=60.0, help='请求超时时间(秒)')
    
    args = parser.parse_args()
    
    # 运行异步主函数
    try:
        asyncio.run(run_async_processing(args))
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.exception(f"程序执行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()