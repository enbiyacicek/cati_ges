"""OSM Overpass erişimi: yedek sunucular, önbellek ve tek-istek sorgular.

Ana sunucu (overpass-api.de) yoğunken bağlantıyı reddedebiliyor; bu modül sırayla
diğer genel sunuculara geçer.
"""
from __future__ import annotations

import time

from .config import CACHE_DIR, USER_AGENT

# osmnx "overpass_url" sonuna /interpreter ekler
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    import osmnx as ox
    s = ox.settings
    s.use_cache = True
    s.cache_folder = str(CACHE_DIR / "osmnx")
    s.overpass_rate_limit = False  # yedek sunucularda /status uç noktası yok
    s.requests_timeout = 90
    s.log_console = False
    try:
        s.http_user_agent = USER_AGENT
    except Exception:
        pass
    _configured = True


def _set_endpoint(url: str):
    import osmnx as ox
    if hasattr(ox.settings, "overpass_url"):
        ox.settings.overpass_url = url
    else:  # eski osmnx sürümleri
        ox.settings.overpass_endpoint = url


def _is_empty_error(e: Exception) -> bool:
    return "InsufficientResponse" in type(e).__name__


def _with_fallback(fn, *args, **kwargs):
    """fn'i sırayla her Overpass sunucusunda dener."""
    _configure()
    errors = []
    for i, url in enumerate(OVERPASS_ENDPOINTS):
        _set_endpoint(url)
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_empty_error(e):
                raise
            errors.append(f"{url.split('/')[2]}: {type(e).__name__}")
            time.sleep(1 + i)
    raise ConnectionError("OSM sunucularına ulaşılamadı (" + "; ".join(errors) +
                          "). Birkaç dakika sonra tekrar deneyin.")


def features(polygon, tags: dict):
    """features_from_polygon + yedek sunucular. Sonuç yoksa boş GeoDataFrame."""
    import geopandas as gpd
    import osmnx as ox
    try:
        f = _with_fallback(ox.features_from_polygon, polygon, tags=tags)
    except Exception as e:
        if _is_empty_error(e):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        raise
    return f[~f.geometry.is_empty].reset_index()


def graph(polygon, network_type: str = "drive"):
    import osmnx as ox
    return _with_fallback(ox.graph_from_polygon, polygon, network_type=network_type, simplify=True)


def geocode(query: str):
    import osmnx as ox
    _configure()
    return ox.geocode(query)
