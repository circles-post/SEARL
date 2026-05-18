# SEARL 本地缓存方案

## 目录
- [缓存组件设计](#缓存组件设计)
- [方案调研](#方案调研)
- [缓存淘汰策略](#缓存淘汰策略)
- [存储数据结构](#存储数据结构)
- [持久化方案](#持久化方案)
- [兜底场景](#兜底场景)
- [使用方式](#使用方式)

## 缓存组件设计

### 核心需求
- **易用性**：服务于训练过程中产生的大量Tools调用结果缓存，避免相同条件查询带来的IO和性能损耗
- **高性能**：本地缓存组件降低远程调用延时，需要本地内存访问
- **高并发**：训练过程需要支持cache多线程并发安全访问
- **持久化**：训练可能存在多轮，需要具备缓存结果序列化反序列化能力
- **内存限制**：实现合理的淘汰策略
- **功能扩展**：自带丰富API或具备开发扩展能力

## 方案调研

### 缓存组件对比分析

| 缓存组件 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| **cachebox** | • 满足本地高性能缓存需求<br>• 支持内存管理和并发访问<br>• 使用灵活，API丰富<br>• 易于集成和扩展<br>• 适合训练或高频场景 | • 受限于可用内存大小<br>• 无法持久化缓存<br>• 需要自行实现持久化<br>• 新开源组件，稳定性待验证 | 需要高性能内存缓存的场景 |
| **functools** | • Python 3.9+ 标准库自带<br>• LRU策略实现<br>• 装饰器形式使用<br>• 线程安全 | • 功能相对简单<br>• 内存限制<br>• 过期策略有限 | 简单的函数结果缓存 |
| **cachetools** | • 多种缓存策略支持<br>• 高度灵活性<br>• API简洁<br>• 支持自定义扩展 | • 仅内存存储<br>• 需额外安装<br>• 持久化需自行实现 | 需要多种缓存策略的场景 |
| **diskcache** | • 磁盘持久化<br>• 功能丰富<br>• 类字典接口<br>• 线程和进程安全 | • I/O开销较大<br>• 需额外安装<br>• 配置管理复杂 | 需要持久化的场景 |
| **joblib.Memory** | • 针对函数输出优化<br>• 透明缓存机制<br>• 磁盘持久化<br>• 大型数据友好 | • 特定场景限制<br>• 依赖Scipy生态<br>• 需额外安装 | 科学计算和机器学习场景 |

### 方案选择
基于以上分析，我们选择采用 **cachebox + 自定义序列化/反序列化** 的方案，原因如下：
1. 满足高性能和并发需求
2. 提供丰富的API支持
3. 易于扩展和定制
4. 适合训练场景

## 缓存淘汰策略

### 策略类型
1. **LRU (Least Recently Used)**
   - 基于访问时间
   - 适合大多数场景
   - 实现简单，效果稳定

2. **LFU (Least Frequently Used)**
   - 基于访问频率
   - 适合访问模式稳定的场景
   - 需要额外统计信息

3. **FIFO (First In First Out)**
   - 基于插入时间
   - 适合数据重要性随时间降低的场景
   - 实现最简单

4. **TTL (Time To Live)**
   - 基于过期时间
   - 适合数据时效性要求高的场景
   - 需要额外的时间管理

## 存储数据结构

### Key设计
```python
def generate_cache_key(method_name: str, input_params: dict) -> str:
    """
    生成缓存键
    Args:
        method_name: 方法名
        input_params: 输入参数
    Returns:
        str: 缓存键
    """
    key_str = f"{method_name}:{json.dumps(input_params, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()
```

### Value设计
```python
{
    "result": Any,  # 实际结果
    "metadata": {
        "created_at": datetime,
        "expires_at": datetime,
        "access_count": int,
        "last_access": datetime
    }
}
```

## 持久化方案

### 文件组织
- **命名格式**：`{method_hash}_{date}_{partition}.cache`
- **存储格式**：`gzip(serialize({key: value}))`
- **目录结构**：
```
cache/
├── 2024-03/
│   ├── method1_20240301_0.cache
│   └── method1_20240301_1.cache
└── 2024-04/
    └── method1_20240401_0.cache
```

### 加载机制
1. 训练启动时通过 `-loadCache` 参数指定缓存文件
2. 支持增量加载和全量加载
3. 支持多节点数据同步

### 数据一致性
- 采用定期回溯策略
- 基于文件名归并缓存数据
- 使用缓存key hash设计确保多节点数据一致

## 兜底场景

### 缓存未命中处理
1. 允许缓存未命中
2. 异步更新缓存
3. 降级处理机制

### 分布式场景
1. 单节点独立构建缓存
2. 独立持久化
3. 预留分布式扩展接口

## 使用方式

### 命令行参数
```bash
# 基础用法
python storage_test.py

# 高级配置
python storage_test.py \
    --cache single|multi \
    --persist \
    --eviction lru|lfu|fifo|ttl \
    --max-size 1000 \
    --persist-dir /path/to/cache \
    --load-cache cache_file.cache
```

### 配置说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cache` | 缓存模式：single/multi | single |
| `--persist` | 是否启用持久化 | false |
| `--eviction` | 缓存淘汰策略 | lru |
| `--max-size` | 最大缓存条目数 | 1000 |
| `--persist-dir` | 持久化目录 | ./cache |
| `--load-cache` | 加载缓存文件 | null |

### 使用示例
```python
from cache_manager import CacheManager

# 初始化缓存管理器
cache = CacheManager(
    mode="single",
    eviction="lru",
    max_size=1000,
    persist=True,
    persist_dir="./cache"
)

# 使用缓存
@cache.cached
def expensive_operation(param1, param2):
    # 实际计算逻辑
    return result
```
