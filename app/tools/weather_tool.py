from __future__ import annotations

import re
from time import time

import httpx

from app.core.config import settings

_WEATHER_CACHE: dict[str, tuple[float, dict[str, object]]] = {}

_CITY_ALIASES = {
    "上海": "Shanghai",
    "北京": "Beijing",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "南京": "Nanjing",
    "苏州": "Suzhou",
    "成都": "Chengdu",
    "重庆": "Chongqing",
    "武汉": "Wuhan",
    "西安": "Xi'an",
    "天津": "Tianjin",
}

_WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "霜雾",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "较强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "短时小阵雨",
    81: "阵雨",
    82: "强阵雨",
    95: "雷雨",
    96: "雷雨伴小冰雹",
    99: "雷雨伴强冰雹",
}


def get_weather_from_question(question: str) -> dict[str, object]:
    location = _extract_location(question)
    if not location:
        return {
            "ok": False,
            "content": "请说明要查询天气的城市，比如“今天上海天气怎么样”。",
        }

    cache_key = location.lower()
    cached = _get_cached(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    geo = _geocode(location)
    if not geo:
        return {
            "ok": False,
            "content": f"没有识别到城市：{location}。请换成更明确的城市名。",
            "location": location,
        }
    weather = _fetch_weather(float(geo["latitude"]), float(geo["longitude"]))
    payload = {
        "ok": True,
        "content": _format_weather(geo, weather),
        "location": geo,
        "weather": weather,
        "provider": "open-meteo",
        "cached": False,
    }
    _set_cached(cache_key, payload)
    return payload


def _extract_location(question: str) -> str:
    for city in _CITY_ALIASES:
        if city in question:
            return city
    cleaned = question
    cleaned = re.sub(r"(今天|明天|后天|现在|目前|实时|当地|的|天气|气温|温度|会不会|下雨|下雪|怎么样|如何|查询|查一下|帮我|看看|？|\?)", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip(" ，,。")
    if len(cleaned) <= 24:
        return cleaned
    return ""


def _geocode(location: str) -> dict[str, object] | None:
    name = _CITY_ALIASES.get(location, location)
    with httpx.Client(timeout=10) as client:
        response = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "zh", "format": "json"},
        )
        response.raise_for_status()
        data = response.json()
    results = data.get("results") or []
    if not results:
        return None
    item = results[0]
    return {
        "name": item.get("name") or location,
        "country": item.get("country") or "",
        "admin1": item.get("admin1") or "",
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "timezone": item.get("timezone") or "",
    }


def _fetch_weather(latitude: float, longitude: float) -> dict[str, object]:
    with httpx.Client(timeout=10) as client:
        response = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        response.raise_for_status()
        return response.json()


def _format_weather(location: dict[str, object], weather: dict[str, object]) -> str:
    current = weather.get("current") or {}
    units = weather.get("current_units") or {}
    daily = weather.get("daily") or {}
    code = int(current.get("weather_code") or 0)
    temp_unit = units.get("temperature_2m") or "°C"
    wind_unit = units.get("wind_speed_10m") or "km/h"
    city = "，".join(str(item) for item in [location.get("country"), location.get("admin1"), location.get("name")] if item)
    lines = [
        f"{city} 当前天气：{_weather_text(code)}，气温 {current.get('temperature_2m')} {temp_unit}，体感 {current.get('apparent_temperature')} {temp_unit}，湿度 {current.get('relative_humidity_2m')}%，降水 {current.get('precipitation')} mm，风速 {current.get('wind_speed_10m')} {wind_unit}。",
    ]
    dates = daily.get("time") or []
    max_t = daily.get("temperature_2m_max") or []
    min_t = daily.get("temperature_2m_min") or []
    probs = daily.get("precipitation_probability_max") or []
    codes = daily.get("weather_code") or []
    if dates:
        lines.append("未来几天：")
        for index, date in enumerate(dates[:3]):
            lines.append(
                f"- {date}：{_weather_text(int(codes[index] if index < len(codes) else 0))}，{min_t[index] if index < len(min_t) else '-'}~{max_t[index] if index < len(max_t) else '-'} {temp_unit}，最高降水概率 {probs[index] if index < len(probs) else '-'}%。"
            )
    return "\n".join(lines)


def _weather_text(code: int) -> str:
    return _WEATHER_CODES.get(code, f"天气代码 {code}")


def _get_cached(cache_key: str) -> dict[str, object] | None:
    ttl = settings.tool_weather_cache_ttl_seconds
    cached = _WEATHER_CACHE.get(cache_key)
    if ttl <= 0 or cached is None:
        return None
    expires_at, payload = cached
    if expires_at < time():
        _WEATHER_CACHE.pop(cache_key, None)
        return None
    return payload


def _set_cached(cache_key: str, payload: dict[str, object]) -> None:
    ttl = settings.tool_weather_cache_ttl_seconds
    if ttl > 0:
        _WEATHER_CACHE[cache_key] = (time() + ttl, payload)
