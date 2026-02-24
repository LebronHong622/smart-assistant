"""
工具函数的参数和返回值的定义
"""
from pydantic import BaseModel, Field

class WeatherQuery(BaseModel):
    """
    天气查询参数
    """
    city_name: str = Field(..., description="要查询的城市名称，例如：北京，上海，广州等")
    
    model_config = {
        "json_schema_extra": {
            "expamples": [
                {
                    "city_name": "北京"
                },
                {
                    "city_name": "上海"
                },
                {
                    "city_name": "广州"
                }
            ]
        }
    }
    
