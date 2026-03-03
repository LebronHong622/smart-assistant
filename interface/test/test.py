from infrastructure.tool.amap_weather_query import amap_weather_query_tool
from infrastructure.tool.tool_shema import WeatherQuery

def test_amap_weather_query_tool():
    result = amap_weather_query_tool.query_weather(WeatherQuery(city_name="歙县"))
    print(result)

test_amap_weather_query_tool()
