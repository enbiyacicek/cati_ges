"""OSM bina poligonları + uydu görüntüsünden YOLO11-seg eğitim veri seti üretir.

Örnek:
    python training/prepare_dataset.py --place "Ortahisar, Trabzon" --place "Atakum, Samsun" \
        --max-km2 2 --out dataset
    python training/prepare_dataset.py --bbox 39.70,40.98,39.73,41.00 --out dataset

Notlar
* OSM poligonları ile uydu görüntüsü arasında birkaç metrelik kayma olabilir. Daha iyi
  sonuç için üretilen etiketleri CVAT / Roboflow gibi bir araçta gözden geçirin.
* Bina kapsaması iyi olan (OSM'de binaların eksiksiz çizildiği) alanlar seçin.
* Etiketler tek sınıftır: 0 = roof
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules import imagery  # noqa: E402


def area_bounds(args):
    import osmnx as ox
    out = []
    for b in args.bbox or []:
        minx, miny, maxx, maxy = map(float, b.split(","))
        out.append((f"bbox_{len(out)}", (minx, miny, maxx, maxy)))
    for p in args.place or []:
        gdf = ox.geocode_to_gdf(p)
        out.append((p.replace(",", "_").replace(" ", ""), tuple(gdf.total_bounds)))
    return out


def tile_bounds(bounds, max_km2):
    """Büyük alanları ~max_km2 büyüklüğünde parçalara böler."""
    minx, miny, maxx, maxy = bounds
    side_deg = np.sqrt(max_km2) / 111.0
    xs = np.arange(minx, maxx, side_deg)
    ys = np.arange(miny, maxy, side_deg)
    for x in xs:
        for y in ys:
            yield (x, y, min(x + side_deg, maxx), min(y + side_deg, maxy))


def polygon_to_label(poly, chip_box, transform_inv, size):
    from shapely.geometry import MultiPolygon
    clipped = poly.intersection(chip_box)
    if clipped.is_empty:
        return []
    parts = clipped.geoms if isinstance(clipped, MultiPolygon) else [clipped]
    lines = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area <= 0:
            continue
        pts = []
        for x, y in list(part.exterior.coords)[:-1]:
            c, r = transform_inv * (x, y)
            pts += [min(max(c / size, 0), 1), min(max(r / size, 0), 1)]
        if len(pts) >= 6:
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", action="append", help="OSM yer adı (birden fazla verilebilir)")
    ap.add_argument("--bbox", action="append", help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--source", default="Esri World Imagery")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--max-km2", type=float, default=1.0, help="Parça başına alan")
    ap.add_argument("--max-parts", type=int, default=20, help="Yer başına en fazla parça")
    ap.add_argument("--neg-ratio", type=float, default=0.1, help="Binasız karoların tutulma oranı")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--out", default="dataset")
    args = ap.parse_args()

    import cv2
    import geopandas as gpd
    import osmnx as ox
    from rasterio.transform import Affine
    from shapely.geometry import box

    out = Path(args.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    n_img = n_obj = 0

    for name, bounds in area_bounds(args):
        parts = list(tile_bounds(bounds, args.max_km2))
        rng.shuffle(parts)
        for pi, pb in enumerate(parts[: args.max_parts]):
            print(f"[{name}] parça {pi + 1}: {pb}")
            try:
                bld = ox.features_from_bbox(pb, tags={"building": True})
            except Exception as e:
                print("  OSM verisi yok:", e)
                continue
            bld = bld[bld.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if bld.empty:
                continue
            tif = out / f"_tmp_{pi}.tif"
            try:
                _, img, tr, crs, _ = imagery.download_geotiff(pb, tif, zoom=args.zoom, source=args.source)
            except Exception as e:
                print("  Görüntü indirilemedi:", e)
                continue
            bld = gpd.GeoDataFrame(geometry=bld.geometry.values, crs="EPSG:4326").to_crs(crs)
            sidx = bld.sindex
            H, W = img.shape[:2]
            S = args.size
            for r in range(0, H - S + 1, S):
                for c in range(0, W - S + 1, S):
                    chip = img[r:r + S, c:c + S]
                    if chip.mean() < 3:
                        continue
                    ctr = tr * Affine.translation(c, r)
                    x0, y1 = ctr * (0, 0)
                    x1, y0 = ctr * (S, S)
                    cb = box(x0, y0, x1, y1)
                    hits = bld.geometry.iloc[list(sidx.query(cb, predicate="intersects"))]
                    lines = []
                    for poly in hits:
                        lines += polygon_to_label(poly, cb, ~ctr, S)
                    if not lines and rng.random() > args.neg_ratio:
                        continue
                    split = "val" if rng.random() < args.val else "train"
                    stem = f"{name}_{pi}_{r}_{c}"
                    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), chip[:, :, ::-1],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                    n_img += 1
                    n_obj += len(lines)
            tif.unlink(missing_ok=True)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: roof\n")
    print(f"Bitti: {n_img} görüntü, {n_obj} çatı etiketi -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
