"""
高德地图天气查询工具
"""
import pandas as pd
import requests
from typing import Dict, Any

from infrastructure.tool.tool_shema import WeatherQuery
from config.settings import settings
from infrastructure.log.log import app_logger

# 定义高德天气查询工具函数类
class AmapWeatherQuery:
    """
    高德地图天气查询工具函数类
    """
    def __init__(self):
        """
        初始化高德地图天气查询工具函数类
        """
        self.api_key = settings.api.amap_api_key
        self.api_url = settings.api.amap_api_url

        self.map_file_path = settings.api.amap_name_code_map_file_path

        app_logger.info("高德地图天气查询工具函数类初始化完成")

    def query_weather(self, query: WeatherQuery) -> Dict[str, Any]:
        """
        查询高德地图指定城市的天气信息

        :param query: 天气查询参数
        :return: 高德地图天气查询结果
        """
        city_name = query.city_name

        try:
            # 从映射文件中获取城市名称对应的城市编码
            city_code = self._get_city_code(city_name)

            # 请求高德地图天气查询API获得结果
            weather_data = self._query_amap_weather(city_code)

            # 封装结果为字符串并返回
            formatted_weather = self._parse_weather_data(weather_data)

            return {
                "success": True,
                "data": formatted_weather
            }

        except FileNotFoundError as e:
            return {
                "success": False,
                "error": str(e)
            }
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }
        except requests.RequestException as e:
            return {
                "success": False,
                "error": str(e)
           }


    def _get_city_code(self, city_name: str) -> str:
        """
        获取城市名称对应的城市编码

        :param city_name: 城市名称
        :return: 城市编码
        """
        df = pd.read_excel(self.map_file_path)

        # 获取城市名列表
        area_list = df["中文名"].tolist()

        city_code = ""

        # 如果城市名称在列表中，返回对应的城市编码
        if city_name in area_list:
            city_code = df[df["中文名"] == city_name]["adcode"].values[0]
            return city_code

        # 如果没有则加上“市、县或区”后缀在去列表中找
        for suffix in ["市", "县", "区"]:
            full_city_name = city_name + suffix
            if full_city_name in area_list:
                city_code = df[df["中文名"] == full_city_name]["adcode"].values[0]
                return city_code

        # 兜底逻辑，如果以上没有，则根据城市名称去匹配
        for name in area_list:
            if city_name in name:
                city_code = df[df["中文名"] == name]["adcode"].values[0]
                return city_code

        # 如果以上都没有匹配到，则抛出异常
        app_logger.error(f"未找到城市名称 {city_name} 对应的城市编码")
        raise ValueError(f"未找到城市名称 {city_name} 对应的城市编码")

    def _query_amap_weather(self, city_code: str):
        """
        查询高德地图指定城市编码的天气信息

        :param city_code: 城市编码
        :return: 高德地图天气查询结果
        """
        params = {
            "key": self.api_key,
            "city": city_code,
            "extensions": "base",
            "output": "JSON"
        }

        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()

            weather_info = response.json()
            if weather_info.get("status") == "1" and weather_info.get("lives"):
                return weather_info
            error_msg = weather_info.get("info", "未知错误")
            app_logger.error(f"高德地图天气查询API返回错误: {error_msg}")
            raise ValueError(f"高德地图天气查询API返回错误: {error_msg}")
        except Exception as e:
            app_logger.error(f"高德地图天气查询API请求失败: {e}")
            raise requests.RequestException(f"高德地图天气查询API请求失败: {e}")

    def _parse_weather_data(self, weather_data: Dict[str, Any]) -> str:
        """
        解析高德地图天气查询结果，提取需要的天气信息

        :param weather_data: 高德地图天气查询结果
        :return: 解析后的天气信息字符串
        """
        live_info = weather_data["lives"][0]

        try:
            temperature = live_info.get("temperature", "未知")
            weather = live_info.get("weather", "未知")
            wind_direction = live_info.get("winddirection", "未知")
            wind_power = live_info.get("windpower", "未知")
            humidity = live_info.get("humidity", "未知")
            report_time = live_info.get("reporttime", "未知")

            formatted_info = f"""
            🌡️ 温度: {temperature}°C
            🌤️ 天气: {weather}
            💨 风向: {wind_direction}风
            🌪️ 风力: {wind_power}级
            💧 湿度: {humidity}%
            🕐 更新时间: {report_time}
                        """.strip()

            return formatted_info

        except Exception as e:
            app_logger.error(f"天气信息格式化失败: {str(e)}")
            return f"天气信息格式化失败: {str(e)}"

amap_weather_query_tool = AmapWeatherQuery()

def exec_get_weather(city_name: str) -> str:
    """获取城市天气"""
    try:
        query_params = WeatherQuery(city_name=city_name)
        query_result = amap_weather_query_tool.query_weather(query_params)
        if query_result.get("success"):
            return query_result.get("data", "查询天气时出错")

        return f"获取{city_name}天气信息失败: {query_result.get('error', '未知错误')}"
    except Exception as e:
        app_logger.error(f"查询{city_name}天气时出错: {e}")
        return f"查询{city_name}天气时出错: {str(e)}"
