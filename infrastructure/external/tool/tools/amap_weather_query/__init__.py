"""高德地图天气查询工具"""
from .schema import WeatherQuery
from .tool import exec_get_weather, amap_weather_query_tool

__all__ = ["WeatherQuery", "exec_get_weather", "amap_weather_query_tool"]
