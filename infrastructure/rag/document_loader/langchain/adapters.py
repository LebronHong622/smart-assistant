"""
Loader 配置适配器
解耦：将不同 loader 的特殊处理逻辑独立出来
"""
from typing import Any, Callable, Dict, Optional, Protocol
from infrastructure.core.log import app_logger


class LoaderConfigAdapter(Protocol):
    """Loader 配置适配器协议"""
    def __call__(
        self,
        file_path: Optional[str],
        loader_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理 loader 配置，返回处理后的 kwargs"""
        ...


def _create_json_metadata_func(content_key: str = "content") -> Callable[[dict, dict], dict]:
    """创建 JSON 元数据提取函数

    将除了 content_key 之外的所有字段都添加到 metadata 中
    """
    def metadata_func(record: dict, metadata: dict) -> dict:
        for key, value in record.items():
            if key != content_key:
                metadata[key] = value
        return metadata
    return metadata_func


def _json_loader_adapter(
    file_path: Optional[str],
    loader_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """JSON Loader 配置适配器

    处理 JSON 加载器的特殊配置：
    - 自动添加 metadata_func 提取所有非 content 字段
    """
    extract_all = loader_kwargs.pop("extract_all_fields", False)
    content_key = loader_kwargs.get("content_key", "content")

    if extract_all and "metadata_func" not in loader_kwargs:
        loader_kwargs["metadata_func"] = _create_json_metadata_func(content_key)
        app_logger.info(f"JSON 加载器启用自动元数据提取，content_key={content_key}")

    return loader_kwargs


# 适配器注册表
_LOADER_ADAPTERS: Dict[str, LoaderConfigAdapter] = {
    "json": _json_loader_adapter,
}
