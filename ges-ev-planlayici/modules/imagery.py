"""Uydu görüntüsü indirme -> coğrafi referanslı GeoTIFF.

Harici `tms2geotiff`/`leafmap` bağımlılığı yerine, karoları paralel indirip
birleştiren ve rasterio ile GeoTIFF (EPSG:3857) yazan hafif bir uygulama.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .config import MAX_TILES, TILE_SOURCES, USER_AGENT
from .geo_utils import lonlat_to_merc, tile_origin_merc, tile_range, tile_resolution


class ImageryError(RuntimeError):
    pass


def _fetch_tile(session: requests.Session, url: str, retries: int = 3) -> np.ndarray | None:
    for _ in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and r.content:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except requests.RequestException:
            pass
    return None


def download_geotiff(bounds, out_path: str | Path, zoom: int = 19,
                     source: str = "Esri World Imagery", progress_cb=None,
                     workers: int = 16):
    """bounds=(minlon, minlat, maxlon, maxlat). GeoTIFF yolunu, RGB dizisini,
    affine dönüşümü ve CRS'yi döndürür."""
    import rasterio
    from rasterio.transform import from_origin

    template = TILE_SOURCES[source]
    x0, y0, x1, y1 = tile_range(bounds, zoom)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    if nx * ny > MAX_TILES:
        raise ImageryError(
            f"Seçilen alan {nx * ny} karo gerektiriyor (sınır {MAX_TILES}). "
            "Daha küçük bir alan seçin veya yakınlaştırma seviyesini düşürün.")

    mosaic = np.zeros((ny * 256, nx * 256, 3), dtype=np.uint8)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    jobs = {}
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                url = template.format(x=tx, y=ty, z=zoom)
                jobs[ex.submit(_fetch_tile, session, url)] = (tx, ty)
        for i, fut in enumerate(as_completed(jobs), 1):
            tx, ty = jobs[fut]
            tile = fut.result()
            if tile is None:
                missing += 1
            else:
                r0, c0 = (ty - y0) * 256, (tx - x0) * 256
                mosaic[r0:r0 + 256, c0:c0 + 256] = tile[:256, :256]
            if progress_cb:
                progress_cb(i / len(jobs))
    if missing == len(jobs):
        raise ImageryError("Hiçbir uydu karosu indirilemedi. Ağ bağlantısını veya kaynağı kontrol edin.")

    # Mozaiği tam olarak seçilen sınırlara kırp
    res = tile_resolution(zoom)
    ox, oy = tile_origin_merc(x0, y0, zoom)
    mx0, my0 = lonlat_to_merc(bounds[0], bounds[1])
    mx1, my1 = lonlat_to_merc(bounds[2], bounds[3])
    c0 = max(int((mx0 - ox) / res), 0)
    c1 = min(int(np.ceil((mx1 - ox) / res)), mosaic.shape[1])
    r0 = max(int((oy - my1) / res), 0)
    r1 = min(int(np.ceil((oy - my0) / res)), mosaic.shape[0])
    img = mosaic[r0:r1, c0:c1]
    transform = from_origin(ox + c0 * res, oy - r0 * res, res, res)

    out_path = Path(out_path)
    with rasterio.open(out_path, "w", driver="GTiff", height=img.shape[0], width=img.shape[1],
                       count=3, dtype="uint8", crs="EPSG:3857", transform=transform,
                       compress="deflate", tiled=True) as dst:
        dst.write(img.transpose(2, 0, 1))
        dst.update_tags(SOURCE=source, ZOOM=str(zoom))
    return out_path, img, transform, "EPSG:3857", missing


def read_geotiff(path):
    """Kullanıcının yüklediği ortofotoyu okur; 8-bit RGB dizisine dönüştürür."""
    import rasterio
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ImageryError("Yüklenen GeoTIFF'te koordinat sistemi (CRS) yok.")
        bands = min(src.count, 3)
        arr = src.read(list(range(1, bands + 1))).astype(np.float32)
        if bands == 1:
            arr = np.repeat(arr, 3, axis=0)
        if src.dtypes[0] != "uint8":  # 2-98 persentil germe
            lo, hi = np.percentile(arr, (2, 98))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return arr.astype(np.uint8).transpose(1, 2, 0), src.transform, src.crs.to_string()
