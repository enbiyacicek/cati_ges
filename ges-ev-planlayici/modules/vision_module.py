"""Çatı tespiti: YOLO11-seg (ONNX veya PyTorch) ya da OSM bina ayak izi yedek modu.

* .onnx modeller onnxruntime ile çalışır (PyTorch gerekmez, CPU'da hızlı).
* .pt modeller ultralytics kuruluysa çalışır.
* Model yoksa veya kullanıcı seçerse OSM bina poligonları kullanılır.
Büyük görüntüler örtüşmeli pencerelere bölünür; kenarda kesilen çatılar birleştirilir.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import MODELS_DIR


# --------------------------------------------------------------------------- #
# Model bulma / yükleme
# --------------------------------------------------------------------------- #
def find_model() -> Path | None:
    env = os.environ.get("ROOF_MODEL_PATH")
    if env and Path(env).exists():
        return Path(env)
    for pattern in ("*.onnx", "*.pt"):
        found = sorted(MODELS_DIR.glob(pattern))
        if found:
            return found[0]
    return None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def postprocess_seg(out0: np.ndarray, protos: np.ndarray, imgsz: int,
                    conf: float = 0.35, iou: float = 0.5, max_det: int = 300):
    """YOLOv8/11-seg ONNX çıktısını (1, 4+nc+nm, N) ve protos (1, nm, mh, mw)
    maskelere çevirir. Dönen: [(bool mask[imgsz, imgsz], skor), ...]"""
    import cv2

    nm, mh, mw = protos.shape[1:]
    preds = out0[0].T  # (N, 4+nc+nm)
    nc = preds.shape[1] - 4 - nm
    scores = preds[:, 4:4 + nc].max(axis=1)
    keep = scores >= conf
    if not keep.any():
        return []
    preds, scores = preds[keep], scores[keep]
    xywh = preds[:, :4]
    coefs = preds[:, 4 + nc:]
    x1y1 = xywh[:, :2] - xywh[:, 2:] / 2
    boxes_xywh = np.concatenate([x1y1, xywh[:, 2:]], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)[:max_det]
    if idx.size == 0:
        return []
    masks = _sigmoid(coefs[idx] @ protos[0].reshape(nm, -1)).reshape(-1, mh, mw)
    out = []
    for m, i in zip(masks, idx):
        m = cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        x, y, w, h = boxes_xywh[i]
        crop = np.zeros_like(m, dtype=bool)
        r0, r1 = max(int(y), 0), min(int(np.ceil(y + h)), imgsz)
        c0, c1 = max(int(x), 0), min(int(np.ceil(x + w)), imgsz)
        crop[r0:r1, c0:c1] = True
        binm = (m > 0.5) & crop
        if binm.sum() > 0:
            out.append((binm, float(scores[i])))
    return out


class OnnxSegModel:
    def __init__(self, path: Path, conf: float = 0.35, iou: float = 0.5):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(os.cpu_count() or 2, 2)
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.fp16 = "float16" in inp.type
        shp = inp.shape
        self.imgsz = shp[2] if isinstance(shp[2], int) else 640
        self.conf, self.iou = conf, iou

    def predict(self, chip: np.ndarray):
        h, w = chip.shape[:2]
        pad = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad[:h, :w] = chip
        x = pad.transpose(2, 0, 1)[None].astype(np.float16 if self.fp16 else np.float32) / 255.0
        out0, protos = self.sess.run(None, {self.input_name: x})[:2]
        res = postprocess_seg(out0.astype(np.float32), protos.astype(np.float32),
                              self.imgsz, self.conf, self.iou)
        return [(m[:h, :w], s) for m, s in res if m[:h, :w].any()]


class UltralyticsSegModel:
    def __init__(self, path: Path, conf: float = 0.35, iou: float = 0.5, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def predict(self, chip: np.ndarray):
        r = self.model.predict(chip[:, :, ::-1], imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        masks = r.masks.data.cpu().numpy() > 0.5
        scores = r.boxes.conf.cpu().numpy()
        return [(m, float(s)) for m, s in zip(masks, scores) if m.any()]


def load_model(path: Path, conf: float = 0.35):
    if path.suffix.lower() == ".onnx":
        return OnnxSegModel(path, conf=conf)
    return UltralyticsSegModel(path, conf=conf)


# --------------------------------------------------------------------------- #
# Pencereli çıkarım + vektörleştirme
# --------------------------------------------------------------------------- #
def sliding_windows(h: int, w: int, tile: int, overlap: int):
    stride = max(tile - overlap, 1)
    rows = list(range(0, max(h - tile, 0) + 1, stride))
    cols = list(range(0, max(w - tile, 0) + 1, stride))
    if rows[-1] + tile < h:
        rows.append(h - tile)
    if cols[-1] + tile < w:
        cols.append(w - tile)
    for r in rows:
        for c in cols:
            yield max(r, 0), max(c, 0)


def mask_to_polygon(mask: np.ndarray, transform):
    """Tek bir örnek maskesini (bool) rasterio.features.shapes ile poligona çevirir."""
    from rasterio import features
    from shapely.geometry import shape

    polys = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask,
                                                  transform=transform) if v == 1]
    if not polys:
        return None
    poly = max(polys, key=lambda p: p.area)
    poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def merge_overlapping(polys, scores, thr: float = 0.25):
    """Pencere örtüşmelerinde iki kez tespit edilen / kenarda bölünen çatıları birleştirir."""
    from shapely import STRtree
    from shapely.ops import unary_union

    n = len(polys)
    if n == 0:
        return [], []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = STRtree(polys)
    left, right = tree.query(polys, predicate="intersects")
    for i, j in zip(left, right):
        if i >= j:
            continue
        inter = polys[i].intersection(polys[j]).area
        if inter / max(min(polys[i].area, polys[j].area), 1e-9) > thr:
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out_p, out_s = [], []
    for members in groups.values():
        g = unary_union([polys[k] for k in members])
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        out_p.append(g)
        out_s.append(max(scores[k] for k in members))
    return out_p, out_s


def detect_roofs_yolo(image: np.ndarray, transform, crs: str, model, overlap: int = 160,
                      progress_cb=None):
    import geopandas as gpd
    from rasterio.transform import Affine

    h, w = image.shape[:2]
    tile = getattr(model, "imgsz", 640)
    windows = list(sliding_windows(h, w, tile, overlap))
    polys, scores = [], []
    for k, (r, c) in enumerate(windows, 1):
        chip = image[r:r + tile, c:c + tile]
        if chip.mean() < 3:  # boş (indirilemeyen) karo
            continue
        win_tr = transform * Affine.translation(c, r)
        for mask, s in model.predict(chip):
            p = mask_to_polygon(mask, win_tr)
            if p is not None:
                polys.append(p)
                scores.append(s)
        if progress_cb:
            progress_cb(k / len(windows))
    polys, scores = merge_overlapping(polys, scores)
    gdf = gpd.GeoDataFrame({"guven": scores}, geometry=polys, crs=crs)
    return gdf.to_crs("EPSG:4326")


def detect_roofs_osm(polygon_4326):
    """OSM bina ayak izleri (yedek / test modu)."""
    from . import osm_client

    b = osm_client.features(polygon_4326, {"building": True})
    if b.empty:
        return b
    b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    keep = [c for c in ("building", "name", "building:levels", "height") if c in b.columns]
    b = b[keep + ["geometry"]].reset_index(drop=True)
    b["guven"] = np.nan
    return b.set_crs("EPSG:4326", allow_override=True)


def prepare_roofs(gdf_4326, analysis_polygon, min_area_m2: float = 25.0):
    """Alana kırpar, UTM'ye çevirir, alan hesaplar, gürültüyü filtreler."""
    if gdf_4326.empty:
        return gdf_4326, None
    g = gdf_4326[gdf_4326.geometry.representative_point().within(analysis_polygon)].copy()
    if g.empty:
        return g, None
    utm = g.estimate_utm_crs()
    g = g.to_crs(utm)
    g["geometry"] = g.geometry.buffer(0).simplify(0.3)
    g["Cati_Alani_m2"] = g.geometry.area
    g = g[g["Cati_Alani_m2"] >= min_area_m2].reset_index(drop=True)
    g.insert(0, "Bina_ID", np.arange(1, len(g) + 1))
    return g, utm
