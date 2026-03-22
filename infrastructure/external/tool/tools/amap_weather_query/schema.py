"""
高德地图天气查询工具参数 Schema
定义输入参数类型和详细描述
"""
from pydantic import BaseModel, Field


class WeatherQuery(BaseModel):
    """
    天气查询参数定义

    根据城市名称查询当前天气信息，使用高德地图 API 获取实时数据。
    """
    city_name: str = Field(
        ...,
        description="要查询天气的城市名称，支持中文城市名，例如：北京、上海、广州、深圳等",
        examples=[
            {"city_name": "北京"},
            {"city_name": "上海"},
            {"city_name": "广州"}
        ]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"city_name": "北京"},
                {"city_name": "上海"},
                {"city_name": "广州"}
            ]
        }
    }
