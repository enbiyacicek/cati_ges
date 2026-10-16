"""Coğrafi yardımcı fonksiyonlar: alan hesabı, Web Mercator karo matematiği."""
from __future__ import annotations

import math

EARTH_R = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_R  # 20037508.34...


def geodesic_area_km2(geom) -> float:
    """EPSG:4326 geometrinin jeodezik alanı (km²)."""
    from pyproj import Geod
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geom)
    return abs(area) / 1e6


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    x = lon * ORIGIN_SHIFT / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * EARTH_R
    return x, y


def tile_origin_merc(x: int, y: int, z: int) -> tuple[float, float]:
    """Karonun sol-üst köşesinin Web Mercator koordinatı."""
    res = tile_resolution(z)
    return -ORIGIN_SHIFT + x * 256 * res, ORIGIN_SHIFT - y * 256 * res


def tile_resolution(z: int) -> float:
    """Web Mercator piksel boyutu (m/piksel, ekvatorda)."""
    return 2 * ORIGIN_SHIFT / (256 * 2 ** z)


def tile_range(bounds: tuple[float, float, float, float], z: int):
    """bounds=(minx, miny, maxx, maxy) lon/lat -> (x0, y0, x1, y1) kapsayan karolar."""
    minx, miny, maxx, maxy = bounds
    x0, y0 = lonlat_to_tile(minx, maxy, z)
    x1, y1 = lonlat_to_tile(maxx, miny, z)
    return x0, y0, x1, y1


def count_tiles(bounds, z: int) -> int:
    x0, y0, x1, y1 = tile_range(bounds, z)
    return (x1 - x0 + 1) * (y1 - y0 + 1)
