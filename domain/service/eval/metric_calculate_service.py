"""
指标计算领域服务
计算各项评测指标
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from domain.vo.eval.metric_value import MetricValue


class MetricCalculateService(ABC):
    """指标计算领域服务

    定义了常见的检索评测指标计算接口
    """

    @abstractmethod
    def recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算Recall@K

        召回率：正确检索出的相关文档 / 所有相关文档
        """
        pass

    @abstractmethod
    def precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算Precision@K

        精确率：正确检索出的相关文档 / 检索出的文档
        """
        pass

    @abstractmethod
    def mrr(self, relevant_docs: List[str], retrieved_docs: List[str]) -> MetricValue:
        """计算MRR（Mean Reciprocal Rank）

        平均倒数排名：第一个正确结果排名的倒数
        """
        pass

    @abstractmethod
    def ndcg(self, relevance_scores: List[float], k: int) -> MetricValue:
        """计算NDCG（Normalized Discounted Cumulative Gain）

        考虑排序位置的加权累积增益
        """
        pass

    @abstractmethod
    def hit_rate(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算命中率

        前K个结果中是否包含至少一个相关文档
        """
        pass

    @abstractmethod
    def mean_average_precision(self, all_relevant: List[List[str]], all_retrieved: List[List[str]]) -> MetricValue:
        """计算MAP（Mean Average Precision）

        多个查询的平均Precision
        """
        pass


class MetricCalculateServiceImpl(MetricCalculateService):
    """指标计算服务实现"""

    def recall_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算Recall@K"""
        if not relevant_docs:
            return MetricValue(value=0.0)

        retrieved_k = retrieved_docs[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        recall = hits / len(relevant_docs)
        return MetricValue(value=recall)

    def precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算Precision@K"""
        if k == 0:
            return MetricValue(value=0.0)

        retrieved_k = retrieved_docs[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        precision = hits / k
        return MetricValue(value=precision)

    def mrr(self, relevant_docs: List[str], retrieved_docs: List[str]) -> MetricValue:
        """计算MRR"""
        if not relevant_docs:
            return MetricValue(value=0.0)

        for idx, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                mrr = 1.0 / (idx + 1)
                return MetricValue(value=mrr)
        return MetricValue(value=0.0)

    def ndcg(self, relevance_scores: List[float], k: int) -> MetricValue:
        """计算NDCG"""
        import math

        # 取前K个
        scores = relevance_scores[:k]
        if not scores:
            return MetricValue(value=0.0)

        # 计算DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            dcg += score / math.log2(i + 2)  # i从0开始，所以+2

        # 计算IDCG（理想DCG）
        sorted_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(sorted_scores):
            idcg += score / math.log2(i + 2)

        if idcg == 0:
            return MetricValue(value=0.0)

        ndcg = dcg / idcg
        return MetricValue(value=ndcg)

    def hit_rate(self, relevant_docs: List[str], retrieved_docs: List[str], k: int) -> MetricValue:
        """计算命中率"""
        if not relevant_docs:
            return MetricValue(value=0.0)

        retrieved_k = retrieved_docs[:k]
        hit = any(doc_id in retrieved_k for doc_id in relevant_docs)
        return MetricValue(value=1.0 if hit else 0.0)

    def mean_average_precision(self, all_relevant: List[List[str]], all_retrieved: List[List[str]]) -> MetricValue:
        """计算MAP"""
        if not all_relevant:
            return MetricValue(value=0.0)

        total_ap = 0.0
        count = 0

        for relevant, retrieved in zip(all_relevant, all_retrieved):
            if not relevant:
                continue
            ap = self._average_precision(relevant, retrieved)
            total_ap += ap
            count += 1

        if count == 0:
            return MetricValue(value=0.0)

        map_value = total_ap / count
        return MetricValue(value=map_value)

    def _average_precision(self, relevant_docs: List[str], retrieved_docs: List[str]) -> float:
        """计算单个查询的Average Precision"""
        if not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        hits = 0
        sum_precision = 0.0

        for idx, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                hits += 1
                precision = hits / (idx + 1)
                sum_precision += precision

        if hits == 0:
            return 0.0

        return sum_precision / hits
