"""
文档相似度领域服务
提供纯业务逻辑的相似度计算和阈值管理
"""

from pydantic import BaseModel, field_validator
from enum import Enum


class DistanceMetricType(str, Enum):
    """距离度量类型"""
    EUCLIDEAN = "l2"  # 欧氏距离（L2）
    COSINE = "cosine"  # 余弦距离
    DOT_PRODUCT = "ip"  # 内积


class SimilarityScore(BaseModel):
    """
    相似度分数值对象
    """
    value: float

    model_config = {"frozen": True}

    @field_validator('value')
    def validate_range(cls, v):
        """验证相似度分数范围"""
        if v < 0 or v > 1:
            raise ValueError("相似度分数必须在 0 到 1 之间")
        return v

    def is_highly_relevant(self) -> bool:
        """判断是否高度相关（>= 0.8）"""
        return self.value >= 0.8

    def is_relevant(self) -> bool:
        """判断是否相关（>= 0.5）"""
        return self.value >= 0.5

    def is_low_relevance(self) -> bool:
        """判断是否低相关性（< 0.3）"""
        return self.value < 0.3


class DocumentSimilarityService:
    """
    文档相似度领域服务
    负责相似度计算的纯业务逻辑
    """

    def __init__(self, metric_type: DistanceMetricType = DistanceMetricType.EUCLIDEAN):
        self.metric_type = metric_type

    def convert_distance_to_score(self, distance: float) -> SimilarityScore:
        """
        将距离转换为相似度分数

        Args:
            distance: 距离值

        Returns:
            相似度分数值对象
        """
        if distance < 0:
            raise ValueError("距离值不能为负数")

        if self.metric_type == DistanceMetricType.EUCLIDEAN:
            # L2距离转换为相似度：使用 Sigmoid 函数
            # 距离越小，相似度越高
            score = 1 / (1 + distance)
            return SimilarityScore(value=score)

        elif self.metric_type == DistanceMetricType.COSINE:
            # 余弦距离 = 1 - 余弦相似度
            # 所以相似度 = 1 - 距离
            score = 1 - distance
            return SimilarityScore(value=score)

        elif self.metric_type == DistanceMetricType.DOT_PRODUCT:
            # 内积值本身可以作为相似度的度量
            # 通常需要归一化，这里假设已经归一化
            score = min(1, max(0, distance))
            return SimilarityScore(value=score)

        else:
            raise ValueError(f"不支持的距离类型: {self.metric_type}")

    def calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> SimilarityScore:
        """
        计算两个向量之间的相似度

        Args:
            embedding1: 向量1
            embedding2: 向量2

        Returns:
            相似度分数值对象
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("向量维度不一致")

        if self.metric_type == DistanceMetricType.EUCLIDEAN:
            # 计算欧氏距离
            distance = sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)) ** 0.5
            return self.convert_distance_to_score(distance)

        elif self.metric_type == DistanceMetricType.COSINE:
            # 计算余弦相似度
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            magnitude1 = sum(a ** 2 for a in embedding1) ** 0.5
            magnitude2 = sum(b ** 2 for b in embedding2) ** 0.5

            if magnitude1 == 0 or magnitude2 == 0:
                return SimilarityScore(value=0.0)

            cosine_similarity = dot_product / (magnitude1 * magnitude2)
            return SimilarityScore(value=cosine_similarity)

        elif self.metric_type == DistanceMetricType.DOT_PRODUCT:
            # 直接使用内积
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            return SimilarityScore(value=dot_product)

        else:
            raise ValueError(f"不支持的距离类型: {self.metric_type}")

    def is_similar_enough(self, score: SimilarityScore, threshold: float) -> bool:
        """
        判断相似度是否达到阈值

        Args:
            score: 相似度分数
            threshold: 阈值

        Returns:
            是否达到阈值
        """
        if threshold < 0 or threshold > 1:
            raise ValueError("阈值必须在 0 到 1 之间")

        return score.value >= threshold

    def get_recommended_threshold(self, precision_requirement: str = "medium") -> float:
        """
        根据精度要求获取推荐的相似度阈值

        Args:
            precision_requirement: 精度要求 (low/medium/high)

        Returns:
            推荐的相似度阈值
        """
        thresholds = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.8
        }

        if precision_requirement not in thresholds:
            raise ValueError(f"不支持的精度要求: {precision_requirement}")

        return thresholds[precision_requirement]

    def get_default_threshold(self) -> float:
        """
        获取默认的相似度阈值

        Returns:
            默认阈值
        """
        return 0.5
