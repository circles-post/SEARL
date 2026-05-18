"""
GraphRAG 增强的工具检索系统

基于现有的 centralized_qwen3_manager.py 升级到 GraphRAG，增强检索能力：
1. 多跳推理：基于依赖图进行多跳搜索
2. 社区检测：将相关工具聚类成社区并建立层次结构
3. 全局-局部搜索：结合全局摘要和局部细节检索
4. 关系增强：利用工具间的调用关系提升检索质量
5. 动态权重：根据查询类型动态调整搜索策略
6. 执行历史：考虑工具的历史成功率和使用频率

Author: Based on CentralizedQwenManager
Date: 2026-01-01
"""

import re
import json
import time
import threading
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from pathlib import Path

# ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


class GraphRAGRetrieval:
    """
    GraphRAG 增强的工具检索引擎

    核心改进：
    - 利用工具依赖图进行多跳推理
    - 社区检测和层次化组织
    - 全局摘要 + 局部详细信息的双层检索
    - 基于执行历史的动态排序
    """

    def __init__(
        self,
        embedding_model_path: str,
        device: str = 'cuda:0',
        enable_community_detection: bool = True,
        enable_history_tracking: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        初始化 GraphRAG 检索引擎

        Args:
            embedding_model_path: SentenceTransformer 模型路径
            device: 设备 ('cuda:0' 或 'cpu')
            enable_community_detection: 是否启用社区检测
            enable_history_tracking: 是否启用历史跟踪
            cache_dir: 缓存目录
        """
        # 基础组件
        self.lock = threading.RLock()
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        # 数据存储
        self.mcps: Dict[str, Dict[str, Any]] = {}  # 工具元数据
        self.tool_graph = nx.DiGraph()  # 工具依赖图
        self.tool_embeddings: Dict[str, np.ndarray] = {}  # 工具向量
        self.tfidf_matrix = None  # TF-IDF 矩阵
        self.tfidf_mcp_ids: List[str] = []  # TF-IDF 对应的工具ID

        # GraphRAG 特性
        self.enable_community_detection = enable_community_detection
        self.enable_history_tracking = enable_history_tracking
        self.communities: Dict[str, Set[str]] = {}  # 社区ID -> 工具集合
        self.community_summaries: Dict[str, str] = {}  # 社区ID -> 摘要
        self.global_summary: str = ""  # 全局摘要

        # 执行历史（用于动态排序）
        self.execution_history: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'total_time': 0.0,
            'last_used': 0.0,
            'co_occurrence': Counter()  # 与其他工具的共现次数
        })

        # 缓存
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/graphrag_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GraphRAG] Initialized with embedding_model={embedding_model_path}, device={device}")
        print(f"[GraphRAG] Community detection: {enable_community_detection}, History tracking: {enable_history_tracking}")

    # ==================== 数据加载与索引构建 ====================

    def load_tools(self, mcps: Dict[str, Dict[str, Any]], tool_graph: Optional[nx.DiGraph] = None):
        """
        加载工具数据并构建索引

        Args:
            mcps: 工具元数据字典
            tool_graph: 工具依赖图（可选）
        """
        with self.lock:
            self.mcps = mcps.copy()

            # 构建或更新工具图
            if tool_graph is not None:
                self.tool_graph = tool_graph.copy()
            else:
                # 从工具元数据推断图结构
                self._build_graph_from_metadata()

            # 构建索引
            self._build_embeddings()
            self._build_tfidf_index()

            # 社区检测
            if self.enable_community_detection:
                self._detect_communities()
                self._build_community_summaries()
                self._build_global_summary()

            print(f"[GraphRAG] Loaded {len(self.mcps)} tools, graph has {self.tool_graph.number_of_nodes()} nodes, {self.tool_graph.number_of_edges()} edges")
            if self.enable_community_detection:
                print(f"[GraphRAG] Detected {len(self.communities)} communities")

    def _build_graph_from_metadata(self):
        """从工具元数据构建依赖图"""
        self.tool_graph.clear()

        for mcp_name, mcp_data in self.mcps.items():
            step_id = mcp_data.get('step_id', mcp_name)
            self.tool_graph.add_node(step_id, mcp_name=mcp_name)

            # 从 requires 字段提取依赖
            requires = mcp_data.get('requires', [])
            if isinstance(requires, str):
                requires = [r.strip() for r in requires.split(',') if r.strip()]

            for req in requires:
                if req in self.mcps:
                    req_step_id = self.mcps[req].get('step_id', req)
                    self.tool_graph.add_edge(req_step_id, step_id)

    def _build_embeddings(self):
        """构建工具的语义向量"""
        print("[GraphRAG] Building embeddings...")

        for mcp_name, mcp_data in self.mcps.items():
            # 构建丰富的文本表示
            text_parts = []

            if mcp_data.get('description'):
                text_parts.append(f"Description: {mcp_data['description']}")
            if mcp_data.get('arguments'):
                text_parts.append(f"Arguments: {mcp_data['arguments']}")
            if mcp_data.get('returns'):
                text_parts.append(f"Returns: {mcp_data['returns']}")

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            embedding = self.embedding_model.encode(combined_text, convert_to_numpy=True)
            self.tool_embeddings[mcp_name] = embedding

        print(f"[GraphRAG] Built embeddings for {len(self.tool_embeddings)} tools")

    def _build_tfidf_index(self):
        """构建 TF-IDF 索引"""
        print("[GraphRAG] Building TF-IDF index...")

        texts = []
        self.tfidf_mcp_ids = []

        for mcp_name, mcp_data in self.mcps.items():
            text_parts = []
            if mcp_data.get('description'):
                text_parts.append(mcp_data['description'])
            if mcp_data.get('arguments'):
                text_parts.append(mcp_data['arguments'])
            if mcp_data.get('returns'):
                text_parts.append(mcp_data['returns'])

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            texts.append(combined_text)
            self.tfidf_mcp_ids.append(mcp_name)

        if texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            print(f"[GraphRAG] Built TF-IDF index for {len(texts)} tools")
        else:
            self.tfidf_matrix = None
            print("[GraphRAG] Warning: No texts to build TF-IDF index")

    # ==================== 社区检测 ====================

    def _detect_communities(self):
        """
        使用层次聚类进行社区检测

        结合图结构和语义相似度进行聚类
        """
        if not self.tool_embeddings:
            print("[GraphRAG] No embeddings available for community detection")
            return

        print("[GraphRAG] Detecting communities...")

        # 构建相似度矩阵（结合语义和结构）
        mcp_names = list(self.tool_embeddings.keys())
        n_tools = len(mcp_names)

        if n_tools < 2:
            # 只有一个工具，放入单一社区
            self.communities = {'community_0': set(mcp_names)}
            return

        # 语义相似度
        embeddings = np.array([self.tool_embeddings[name] for name in mcp_names])
        semantic_sim = cosine_similarity(embeddings)

        # 结构相似度（基于图的邻接关系）
        structural_sim = np.zeros((n_tools, n_tools))
        for i, name_i in enumerate(mcp_names):
            for j, name_j in enumerate(mcp_names):
                if i == j:
                    structural_sim[i, j] = 1.0
                else:
                    # 共同邻居数量
                    step_i = self.mcps[name_i].get('step_id', name_i)
                    step_j = self.mcps[name_j].get('step_id', name_j)

                    if self.tool_graph.has_node(step_i) and self.tool_graph.has_node(step_j):
                        neighbors_i = set(self.tool_graph.predecessors(step_i)) | set(self.tool_graph.successors(step_i))
                        neighbors_j = set(self.tool_graph.predecessors(step_j)) | set(self.tool_graph.successors(step_j))
                        common = len(neighbors_i & neighbors_j)
                        total = len(neighbors_i | neighbors_j)
                        structural_sim[i, j] = common / (total + 1e-6)

        # 组合相似度（70% 语义 + 30% 结构）
        combined_sim = 0.7 * semantic_sim + 0.3 * structural_sim

        # 转换为距离矩阵
        distance = 1 - combined_sim

        # 层次聚类
        n_clusters = min(max(2, n_tools // 5), 10)  # 动态确定聚类数
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(distance)

        # 构建社区
        self.communities = {}
        for i, label in enumerate(labels):
            community_id = f'community_{label}'
            if community_id not in self.communities:
                self.communities[community_id] = set()
            self.communities[community_id].add(mcp_names[i])

        print(f"[GraphRAG] Detected {len(self.communities)} communities:")
        for comm_id, members in self.communities.items():
            print(f"  - {comm_id}: {len(members)} tools")

    def _build_community_summaries(self):
        """为每个社区构建摘要"""
        print("[GraphRAG] Building community summaries...")

        for comm_id, members in self.communities.items():
            # 收集社区内所有工具的描述
            descriptions = []
            tool_names = []

            for mcp_name in members:
                mcp_data = self.mcps.get(mcp_name, {})
                if mcp_data.get('description'):
                    descriptions.append(mcp_data['description'])
                tool_names.append(mcp_name)

            # 构建社区摘要（简化版：列出工具名称和关键词）
            if descriptions:
                # 提取高频词作为社区主题
                all_text = " ".join(descriptions)
                words = re.findall(r'\b\w+\b', all_text.lower())
                word_counts = Counter(words)
                # 过滤停用词和短词
                stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'])
                keywords = [word for word, count in word_counts.most_common(10) if word not in stopwords and len(word) > 3]

                summary = f"Community {comm_id} contains {len(members)} tools related to: {', '.join(keywords[:5])}. "
                summary += f"Tools: {', '.join(list(tool_names)[:10])}"
                if len(tool_names) > 10:
                    summary += f" and {len(tool_names) - 10} more"
            else:
                summary = f"Community {comm_id} contains {len(members)} tools: {', '.join(list(tool_names)[:10])}"

            self.community_summaries[comm_id] = summary

        print(f"[GraphRAG] Built summaries for {len(self.community_summaries)} communities")

    def _build_global_summary(self):
        """构建全局摘要"""
        community_texts = []
        for comm_id in sorted(self.community_summaries.keys()):
            community_texts.append(self.community_summaries[comm_id])

        self.global_summary = f"Total {len(self.mcps)} tools organized into {len(self.communities)} communities. " + " ".join(community_texts)
        print(f"[GraphRAG] Built global summary ({len(self.global_summary)} chars)")

    # ==================== 检索方法 ====================

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义搜索（向后兼容）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            匹配的工具列表，每个包含 mcp_id, score, content
        """
        if not self.tool_embeddings:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        scores = []
        for mcp_name, embedding in self.tool_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            scores.append((mcp_name, similarity))

        # 排序并返回 top_k
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mcp_name, score in scores[:top_k]:
            results.append({
                'mcp_id': mcp_name,
                'score': float(score),
                'field': 'description',
                'content': self.mcps[mcp_name].get('description', '')
            })

        return results

    def text_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        TF-IDF 文本搜索（向后兼容）

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            匹配的工具列表
        """
        if self.tfidf_matrix is None or not self.tfidf_mcp_ids:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        actual_top_k = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-actual_top_k:][::-1]

        results = []
        for idx in top_indices:
            mcp_id = self.tfidf_mcp_ids[idx]
            if mcp_id in self.mcps:
                results.append({
                    'mcp_id': mcp_id,
                    'score': float(similarities[idx]),
                    'content': self.mcps[mcp_id]
                })

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.2
    ) -> List[List[Dict[str, Any]]]:
        """
        混合搜索（向后兼容接口）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            weights: 权重字典 {'text': w1, 'semantic': w2}
            score_threshold: 分数阈值

        Returns:
            匹配的工具列表（每个工具包装在列表中以兼容旧接口）
        """
        weights = weights or {'text': 0.5, 'semantic': 0.5}
        score_board = defaultdict(float)

        # 文本搜索
        text_results = self.text_search(query, top_k * 2)
        for r in text_results:
            score_board[r['mcp_id']] += weights.get('text', 0.5) * r['score']

        # 语义搜索
        semantic_results = self.semantic_search(query, top_k * 2)
        for r in semantic_results:
            score_board[r['mcp_id']] += weights.get('semantic', 0.5) * r['score']

        # 过滤和排序
        filtered_results = [(mcp_id, score) for mcp_id, score in score_board.items() if score > score_threshold]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]

        detailed_results = []
        for mcp_id, score in sorted_results:
            mcp_item = self.mcps.get(mcp_id)
            if mcp_item:
                mcp_item_with_score = mcp_item.copy()
                mcp_item_with_score['search_score'] = score
                detailed_results.append([mcp_item_with_score])

        return detailed_results

    # ==================== GraphRAG 增强检索 ====================

    def graph_enhanced_search(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
        use_community: bool = True,
        use_history: bool = True,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        GraphRAG 增强搜索（核心方法）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            max_hops: 最大跳数（多跳推理）
            use_community: 是否使用社区信息
            use_history: 是否使用执行历史
            weights: 权重配置

        Returns:
            增强的搜索结果列表
        """
        weights = weights or {
            'semantic': 0.3,
            'text': 0.2,
            'graph': 0.2,
            'community': 0.15,
            'history': 0.15
        }

        print(f"[GraphRAG] Graph-enhanced search: query='{query[:50]}...', top_k={top_k}, max_hops={max_hops}")

        # 1. 基础检索（语义 + 文本）
        semantic_results = self.semantic_search(query, top_k * 3)
        text_results = self.text_search(query, top_k * 3)

        # 2. 收集候选工具
        candidates = set()
        base_scores = defaultdict(float)

        for r in semantic_results:
            mcp_id = r['mcp_id']
            candidates.add(mcp_id)
            base_scores[mcp_id] += weights['semantic'] * r['score']

        for r in text_results:
            mcp_id = r['mcp_id']
            candidates.add(mcp_id)
            base_scores[mcp_id] += weights['text'] * r['score']

        # 3. 图增强：多跳扩展
        graph_scores = self._compute_graph_scores(candidates, max_hops=max_hops)

        # 4. 社区增强
        community_scores = {}
        if use_community and self.communities:
            community_scores = self._compute_community_scores(query, candidates)

        # 5. 历史增强
        history_scores = {}
        if use_history and self.enable_history_tracking:
            history_scores = self._compute_history_scores(candidates)

        # 6. 综合评分
        final_scores = {}
        all_candidates = candidates | set(graph_scores.keys())

        for mcp_id in all_candidates:
            score = 0.0
            score += base_scores.get(mcp_id, 0.0)
            score += weights.get('graph', 0.2) * graph_scores.get(mcp_id, 0.0)
            score += weights.get('community', 0.15) * community_scores.get(mcp_id, 0.0)
            score += weights.get('history', 0.15) * history_scores.get(mcp_id, 0.0)
            final_scores[mcp_id] = score

        # 7. 排序并返回
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        detailed_results = []
        for mcp_id, score in sorted_results:
            mcp_item = self.mcps.get(mcp_id)
            if mcp_item:
                result = mcp_item.copy()
                result['search_score'] = float(score)
                result['mcp_id'] = mcp_id
                result['mcp_name'] = mcp_id

                # 添加解释信息
                result['score_breakdown'] = {
                    'base': base_scores.get(mcp_id, 0.0),
                    'graph': graph_scores.get(mcp_id, 0.0),
                    'community': community_scores.get(mcp_id, 0.0),
                    'history': history_scores.get(mcp_id, 0.0)
                }

                detailed_results.append(result)

        print(f"[GraphRAG] Returned {len(detailed_results)} results")
        return detailed_results

    def _compute_graph_scores(self, seed_tools: Set[str], max_hops: int = 2) -> Dict[str, float]:
        """
        计算图结构分数（多跳推理）

        从种子工具出发，通过依赖图扩展，距离越近分数越高
        """
        graph_scores = {}

        # BFS 多跳扩展
        for seed in seed_tools:
            step_id = self.mcps[seed].get('step_id', seed)
            if not self.tool_graph.has_node(step_id):
                continue

            # 前向依赖（这个工具依赖谁）
            visited = {step_id: 0}
            queue = [(step_id, 0)]

            while queue:
                current, distance = queue.pop(0)
                if distance >= max_hops:
                    continue

                # 前驱（依赖的工具）
                for pred in self.tool_graph.predecessors(current):
                    if pred not in visited:
                        visited[pred] = distance + 1
                        queue.append((pred, distance + 1))

                # 后继（被依赖的工具）
                for succ in self.tool_graph.successors(current):
                    if succ not in visited:
                        visited[succ] = distance + 1
                        queue.append((succ, distance + 1))

            # 计算分数（距离越近分数越高）
            for node, dist in visited.items():
                # 找到对应的 mcp_name
                mcp_name = self.tool_graph.nodes[node].get('mcp_name', None)
                if mcp_name and mcp_name in self.mcps:
                    decay_factor = 0.5 ** dist  # 指数衰减
                    graph_scores[mcp_name] = max(graph_scores.get(mcp_name, 0.0), decay_factor)

        return graph_scores

    def _compute_community_scores(self, query: str, candidates: Set[str]) -> Dict[str, float]:
        """
        计算社区相关性分数

        如果查询与某个社区的摘要相似，则该社区内的工具获得加分
        """
        if not self.community_summaries:
            return {}

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        community_scores = {}

        # 计算查询与每个社区摘要的相似度
        for comm_id, summary in self.community_summaries.items():
            summary_embedding = self.embedding_model.encode(summary, convert_to_numpy=True)
            similarity = cosine_similarity([query_embedding], [summary_embedding])[0][0]

            # 为该社区的所有成员加分
            for mcp_name in self.communities[comm_id]:
                if mcp_name in candidates:
                    community_scores[mcp_name] = max(community_scores.get(mcp_name, 0.0), similarity)

        return community_scores

    def _compute_history_scores(self, candidates: Set[str]) -> Dict[str, float]:
        """
        计算基于执行历史的分数

        考虑成功率、使用频率、最近使用时间等因素
        """
        history_scores = {}
        current_time = time.time()

        for mcp_name in candidates:
            if mcp_name not in self.execution_history:
                history_scores[mcp_name] = 0.0
                continue

            hist = self.execution_history[mcp_name]
            total_exec = hist['success_count'] + hist['failure_count']

            if total_exec == 0:
                history_scores[mcp_name] = 0.0
                continue

            # 成功率 (0-1)
            success_rate = hist['success_count'] / total_exec

            # 使用频率 (归一化)
            frequency_score = min(1.0, total_exec / 100.0)

            # 新鲜度 (最近使用的工具加分)
            time_diff = current_time - hist['last_used']
            recency_score = np.exp(-time_diff / 86400)  # 24小时衰减

            # 综合评分
            score = 0.5 * success_rate + 0.3 * frequency_score + 0.2 * recency_score
            history_scores[mcp_name] = score

        return history_scores

    # ==================== 执行历史管理 ====================

    def record_execution(
        self,
        mcp_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        """
        记录工具执行历史

        Args:
            mcp_name: 工具名称
            success: 是否成功
            execution_time: 执行时间（秒）
            co_used_tools: 同时使用的其他工具
        """
        if not self.enable_history_tracking:
            return

        with self.lock:
            hist = self.execution_history[mcp_name]

            if success:
                hist['success_count'] += 1
            else:
                hist['failure_count'] += 1

            hist['total_time'] += execution_time
            hist['last_used'] = time.time()

            # 记录共现
            if co_used_tools:
                for other_tool in co_used_tools:
                    if other_tool != mcp_name:
                        hist['co_occurrence'][other_tool] += 1

    def get_execution_stats(self, mcp_name: str) -> Dict[str, Any]:
        """获取工具的执行统计信息"""
        if mcp_name not in self.execution_history:
            return {}

        hist = self.execution_history[mcp_name]
        total_exec = hist['success_count'] + hist['failure_count']

        return {
            'success_count': hist['success_count'],
            'failure_count': hist['failure_count'],
            'success_rate': hist['success_count'] / total_exec if total_exec > 0 else 0.0,
            'total_executions': total_exec,
            'avg_time': hist['total_time'] / total_exec if total_exec > 0 else 0.0,
            'last_used': hist['last_used'],
            'frequently_co_used': hist['co_occurrence'].most_common(5)
        }

    # ==================== 工具和缓存管理 ====================

    def update_tool(self, mcp_name: str, mcp_data: Dict[str, Any]):
        """更新单个工具"""
        with self.lock:
            self.mcps[mcp_name] = mcp_data

            # 重建该工具的索引
            text_parts = []
            if mcp_data.get('description'):
                text_parts.append(f"Description: {mcp_data['description']}")
            if mcp_data.get('arguments'):
                text_parts.append(f"Arguments: {mcp_data['arguments']}")
            if mcp_data.get('returns'):
                text_parts.append(f"Returns: {mcp_data['returns']}")

            combined_text = " ".join(text_parts) if text_parts else mcp_name
            self.tool_embeddings[mcp_name] = self.embedding_model.encode(combined_text, convert_to_numpy=True)

            # 重建 TF-IDF 索引（全量重建，较慢）
            self._build_tfidf_index()

            print(f"[GraphRAG] Updated tool: {mcp_name}")

    def remove_tool(self, mcp_name: str):
        """删除工具"""
        with self.lock:
            if mcp_name in self.mcps:
                del self.mcps[mcp_name]
            if mcp_name in self.tool_embeddings:
                del self.tool_embeddings[mcp_name]
            if mcp_name in self.execution_history:
                del self.execution_history[mcp_name]

            # 从图中删除
            step_id = None
            for node, data in self.tool_graph.nodes(data=True):
                if data.get('mcp_name') == mcp_name:
                    step_id = node
                    break
            if step_id is not None:
                self.tool_graph.remove_node(step_id)

            # 重建索引
            self._build_tfidf_index()

            print(f"[GraphRAG] Removed tool: {mcp_name}")

    def save_cache(self, cache_path: Optional[str] = None):
        """保存缓存到磁盘"""
        if cache_path is None:
            cache_path = self.cache_dir / "graphrag_cache.json"

        cache_data = {
            'execution_history': dict(self.execution_history),
            'communities': {k: list(v) for k, v in self.communities.items()},
            'community_summaries': self.community_summaries,
            'global_summary': self.global_summary
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"[GraphRAG] Cache saved to {cache_path}")

    def load_cache(self, cache_path: Optional[str] = None):
        """从磁盘加载缓存"""
        if cache_path is None:
            cache_path = self.cache_dir / "graphrag_cache.json"

        if not Path(cache_path).exists():
            print(f"[GraphRAG] Cache file not found: {cache_path}")
            return

        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # 恢复执行历史
        for mcp_name, hist in cache_data.get('execution_history', {}).items():
            self.execution_history[mcp_name] = defaultdict(lambda: 0)
            self.execution_history[mcp_name].update(hist)
            # 恢复 Counter 对象
            if 'co_occurrence' in hist:
                self.execution_history[mcp_name]['co_occurrence'] = Counter(hist['co_occurrence'])

        # 恢复社区信息
        self.communities = {k: set(v) for k, v in cache_data.get('communities', {}).items()}
        self.community_summaries = cache_data.get('community_summaries', {})
        self.global_summary = cache_data.get('global_summary', '')

        print(f"[GraphRAG] Cache loaded from {cache_path}")


# ==================== 辅助函数 ====================

def create_graphrag_retrieval(
    mcps: Dict[str, Dict[str, Any]],
    tool_graph: Optional[nx.DiGraph] = None,
    embedding_model_path: str = 'sentence-transformers/all-MiniLM-L6-v2',
    device: str = 'cuda:0',
    enable_community_detection: bool = True,
    enable_history_tracking: bool = True,
    cache_dir: Optional[str] = None
) -> GraphRAGRetrieval:
    """
    便捷函数：创建并初始化 GraphRAG 检索引擎

    Args:
        mcps: 工具元数据字典
        tool_graph: 工具依赖图（可选）
        embedding_model_path: SentenceTransformer 模型路径
        device: 设备
        enable_community_detection: 启用社区检测
        enable_history_tracking: 启用历史跟踪
        cache_dir: 缓存目录

    Returns:
        初始化好的 GraphRAGRetrieval 实例
    """
    retrieval = GraphRAGRetrieval(
        embedding_model_path=embedding_model_path,
        device=device,
        enable_community_detection=enable_community_detection,
        enable_history_tracking=enable_history_tracking,
        cache_dir=cache_dir
    )

    retrieval.load_tools(mcps, tool_graph)

    return retrieval


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("GraphRAG 检索引擎测试")
    print("=" * 60)

    # 示例数据
    sample_mcps = {
        'web_search': {
            'mcp_name': 'web_search',
            'description': 'Search the web for information using Bing API',
            'arguments': 'query: str - search query',
            'returns': 'List of search results',
            'requires': [],
            'code': 'web_search(query)',
            'step_id': 1
        },
        'extract_content': {
            'mcp_name': 'extract_content',
            'description': 'Extract main content from a web page',
            'arguments': 'url: str - web page URL',
            'returns': 'Extracted text content',
            'requires': ['web_search'],
            'code': 'extract_content(url)',
            'step_id': 2
        },
        'summarize_text': {
            'mcp_name': 'summarize_text',
            'description': 'Summarize long text into a concise summary',
            'arguments': 'text: str - text to summarize',
            'returns': 'Summary string',
            'requires': ['extract_content'],
            'code': 'summarize_text(text)',
            'step_id': 3
        },
        'python_executor': {
            'mcp_name': 'python_executor',
            'description': 'Execute Python code in a sandbox environment',
            'arguments': 'code: str - Python code to execute',
            'returns': 'Execution result',
            'requires': [],
            'code': 'python_executor(code)',
            'step_id': 4
        }
    }

    # 创建检索引擎
    print("\n1. 创建 GraphRAG 检索引擎...")
    retrieval = create_graphrag_retrieval(
        mcps=sample_mcps,
        enable_community_detection=True,
        enable_history_tracking=True,
        device='cpu'  # 测试时使用 CPU
    )

    # 测试基础检索
    print("\n2. 测试语义搜索...")
    results = retrieval.semantic_search("search for information", top_k=2)
    for r in results:
        print(f"  - {r['mcp_id']}: score={r['score']:.4f}")

    # 测试混合搜索
    print("\n3. 测试混合搜索...")
    results = retrieval.hybrid_search("find and summarize web content", top_k=3)
    for r_list in results:
        for r in r_list:
            print(f"  - {r.get('mcp_name')}: score={r.get('search_score', 0):.4f}")

    # 测试 GraphRAG 增强搜索
    print("\n4. 测试 GraphRAG 增强搜索...")
    results = retrieval.graph_enhanced_search("search and extract information from websites", top_k=3, max_hops=2)
    for r in results:
        print(f"  - {r['mcp_name']}: total_score={r['search_score']:.4f}")
        if 'score_breakdown' in r:
            print(f"    breakdown: {r['score_breakdown']}")

    # 测试执行历史记录
    print("\n5. 测试执行历史记录...")
    retrieval.record_execution('web_search', success=True, execution_time=1.5)
    retrieval.record_execution('web_search', success=True, execution_time=1.2)
    retrieval.record_execution('extract_content', success=False, execution_time=0.5)

    stats = retrieval.get_execution_stats('web_search')
    print(f"  web_search stats: {stats}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
