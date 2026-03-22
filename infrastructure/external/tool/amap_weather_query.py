"""
高德地图天气查询工具
向后兼容：内容已移动到 tools/amap_weather_query/
"""
from .tools.amap_weather_query.tool import exec_get_weather, amap_weather_query_tool
from .tools.amap_weather_query.schema import WeatherQuery

__all__ = ["exec_getWeather", "amap_weather_query_tool", "WeatherQuery"]
