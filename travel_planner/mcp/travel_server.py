#***********************************************
#      Filename: travel_server.py
#   Description: 旅游路书 MCP 服务端（基于高德地图 API + FastMCP）
#***********************************************
#
# 提供旅游路书生成所需的位置/天气/POI/路径规划/距离测算等 MCP 工具。
# 传输方式：stdio。可被任意 MCP 客户端（Claude Desktop / 自研 Agent）接入。
#
# 启动方式：
#   python -m travel_planner.mcp.travel_server
#


import os
from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP


# ===== 配置 =====
# 高德地图 API Key（优先环境变量，回退到默认 key）
AMAP_MAPS_API_KEY = os.environ.get("AMAP_MAPS_API_KEY", "")

# 创建 MCP 服务
mcp = FastMCP("travel-planner-maps")


# ===== 地理编码 / 定位 =====

@mcp.tool()
def maps_regeocode(location: str) -> Dict[str, Any]:
    """将一个高德经纬度坐标（格式: 经度,纬度）转换为行政区划地址信息。

    旅游场景：用户给出景点坐标时，反查所在城市/区域。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/geocode/regeo",
            params={"key": AMAP_MAPS_API_KEY, "location": location},
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"逆地理编码失败: {data.get('info') or data.get('infocode')}"}

        comp = data["regeocode"]["addressComponent"]
        return {
            "province": comp.get("province"),
            "city": comp.get("city"),
            "district": comp.get("district"),
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool()
def maps_geo(address: str, city: Optional[str] = None) -> Dict[str, Any]:
    """将详细的结构化地址转换为经纬度坐标。支持对地标性名胜景区、建筑物名称解析为经纬度坐标。

    旅游场景：把景点名称（如"故宫博物院"）转为坐标，便于后续路径规划。
    """
    try:
        params: Dict[str, Any] = {"key": AMAP_MAPS_API_KEY, "address": address}
        if city:
            params["city"] = city

        response = requests.get(
            "https://restapi.amap.com/v3/geocode/geo", params=params
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"地理编码失败: {data.get('info') or data.get('infocode')}"}

        results = []
        for geo in data.get("geocodes", []):
            results.append({
                "country": geo.get("country"),
                "province": geo.get("province"),
                "city": geo.get("city"),
                "district": geo.get("district"),
                "location": geo.get("location"),
                "level": geo.get("level"),
            })
        return {"results": results}
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


# ===== 天气查询 =====

@mcp.tool()
def maps_weather(city: str, date: str = "") -> Dict[str, Any]:
    """根据城市名称或 adcode 查询指定城市的天气预报。

    旅游场景：出行前查询目的地天气，用于路书的"实用注意事项"章节。

    参数：
        city: 城市名（如"北京"）或 adcode
        date: 可选，指定日期（YYYY-MM-DD），不传则返回全部预报
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={
                "key": AMAP_MAPS_API_KEY,
                "city": city,
                "extensions": "all",
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"天气查询失败: {data.get('info') or data.get('infocode')}"}

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return {"error": "暂无预报数据"}

        formated: Dict[str, Any] = {"城市": forecasts[0].get("city")}
        for forecast in forecasts[0].get("casts", []):
            if date and forecast.get("date") != date:
                continue
            for key, name in {
                "dayweather": "天气",
                "daytemp": "白天气温",
                "nighttemp": "夜间气温",
                "daywind": "风向",
                "daypower": "风力",
            }.items():
                if forecast.get(key):
                    formated[name] = forecast[key]

        return formated
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


# ===== POI 搜索（景点/餐厅/酒店） =====

@mcp.tool()
def maps_text_search(
    keywords: str,
    city: str = "",
    citylimit: str = "false",
    top_k: int = 5,
) -> Dict[str, Any]:
    """关键词搜索 POI（景点、餐厅、酒店、车站等），返回相关信息。

    旅游场景：按关键词搜索目的地景点/美食/住宿，是路书情报搜集的核心工具。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": AMAP_MAPS_API_KEY,
                "keywords": keywords,
                "city": city,
                "citylimit": citylimit,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"关键词搜索失败: {data.get('info') or data.get('infocode')}"}

        pois = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "tel": poi.get("tel"),
                "type": poi.get("type"),
                "location": poi.get("location"),
                "typecode": poi.get("typecode"),
            })

        return {"pois": pois[:top_k]}
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool()
def maps_around_search(
    location: str,
    radius: str = "1000",
    keywords: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """周边搜索：根据坐标和关键词，搜索指定半径内的 POI。

    旅游场景：在某个景点附近搜索餐厅/酒店/地铁站，支撑路书的"周边配套"。

    参数：
        location: 中心坐标（经度,纬度）
        radius: 搜索半径（米），默认 1000
        keywords: 关键词（如"餐厅""酒店""停车场"）
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/around",
            params={
                "key": AMAP_MAPS_API_KEY,
                "location": location,
                "radius": radius,
                "keywords": keywords,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"周边搜索失败: {data.get('info') or data.get('infocode')}"}

        pois = []
        for poi in data.get("pois", []):
            pois.append({
                "id": poi.get("id"),
                "name": poi.get("name"),
                "address": poi.get("address"),
                "tel": poi.get("tel"),
                "type": poi.get("type"),
                "location": poi.get("location"),
                "distance": poi.get("distance"),
            })

        return {"pois": pois[:top_k]}
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool()
def maps_search_detail(id: str) -> Dict[str, Any]:
    """查询 POI ID 的详细信息（关键词搜索/周边搜索返回的 id）。

    旅游场景：拿到景点 id 后查询营业时间、电话、评分等详细信息。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/place/detail",
            params={"key": AMAP_MAPS_API_KEY, "id": id},
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"详情查询失败: {data.get('info') or data.get('infocode')}"}

        if not data.get("pois"):
            return {"error": "未找到 POI"}

        poi = data["pois"][0]
        result = {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "location": poi.get("location"),
            "address": poi.get("address"),
            "tel": poi.get("tel"),
            "city": poi.get("cityname"),
            "type": poi.get("type"),
            "tag": poi.get("tag"),
            "alias": poi.get("alias"),
        }
        if poi.get("biz_ext"):
            result["business_info"] = poi["biz_ext"]
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


# ===== 路径规划 =====

@mcp.tool()
def maps_direction_transit_integrated(
    origin: str,
    destination: str,
    city: str,
    cityd: str,
) -> Dict[str, Any]:
    """综合公共交通路径规划（火车/公交/地铁），跨城场景必须传起点与终点城市。

    旅游场景：规划跨城/市内公共交通方案（如机场到酒店、城市间高铁）。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/transit/integrated",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
                "city": city,
                "cityd": cityd,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"公交路径规划失败: {data.get('info') or data.get('infocode')}"}

        transits = []
        for transit in data.get("route", {}).get("transits", []):
            transits.append({
                "duration": transit.get("duration"),
                "walking_distance": transit.get("walking_distance"),
                "cost": transit.get("cost"),
                "segments_summary": [
                    {
                        "instruction": seg.get("instruction", ""),
                        "mode": "railway" if seg.get("railway") else (
                            "bus" if seg.get("bus", {}).get("buslines") else "walking"
                        ),
                    }
                    for seg in transit.get("segments", [])
                ],
            })

        return {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "transits": transits,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool()
def maps_direction_driving(origin: str, destination: str) -> Dict[str, Any]:
    """驾车路径规划，返回距离、时长与导航步骤。

    旅游场景：自驾游路书的交通方案规划。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/driving",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"驾车路径规划失败: {data.get('info') or data.get('infocode')}"}

        paths = []
        for path in data["route"]["paths"]:
            paths.append({
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "tolls": path.get("tolls"),
                "steps_count": len(path.get("steps", [])),
            })

        return {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "paths": paths,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


@mcp.tool()
def maps_direction_walking(origin: str, destination: str) -> Dict[str, Any]:
    """步行路径规划（100km 以内），返回距离、时长与导航步骤。

    旅游场景：景点间步行距离估算，辅助路书的"行程节奏"安排。
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/direction/walking",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origin": origin,
                "destination": destination,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"步行路径规划失败: {data.get('info') or data.get('infocode')}"}

        paths = []
        for path in data["route"]["paths"]:
            paths.append({
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps_count": len(path.get("steps", [])),
            })

        return {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "paths": paths,
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


# ===== 距离测量 =====

@mcp.tool()
def maps_distance(origins: str, destination: str, type: str = "1") -> Dict[str, Any]:
    """测量多个起点到一个终点的距离，支持驾车/步行/球面距离。

    旅游场景：批量估算多个备选酒店到景点的距离，辅助住宿选择。

    参数：
        origins: 起点坐标，多个用 | 分隔（如 "116.4,39.9|116.5,39.8"）
        destination: 终点坐标
        type: 距离类型（0=直线, 1=驾车, 3=步行）
    """
    try:
        response = requests.get(
            "https://restapi.amap.com/v3/distance",
            params={
                "key": AMAP_MAPS_API_KEY,
                "origins": origins,
                "destination": destination,
                "type": type,
            },
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "1":
            return {"error": f"距离测量失败: {data.get('info') or data.get('infocode')}"}

        results = []
        for r in data.get("results", []):
            results.append({
                "origin_id": r.get("origin_id"),
                "distance_km": round(int(r.get("distance", 0)) / 1000, 2),
                "duration_min": round(int(r.get("duration", 0)) / 60, 1),
            })

        return {"results": results}
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}"}


# ===== 入口 =====

if __name__ == "__main__":
    mcp.run(transport="stdio")
