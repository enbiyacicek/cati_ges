"""OSMnx ile mekânsal bağlam, mesafe analizi ve TOPSIS / Entropi / duyarlılık."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONTEXT_BUFFER_M, MAIN_ROAD_CLASSES, POI_TAGS


# --------------------------------------------------------------------------- #
# OSM verisi
# --------------------------------------------------------------------------- #
def _tag_match(df, key, values):
    if key not in df.columns:
        return np.zeros(len(df), bool)
    col = df[key]
    if values is True:
        return col.notna().to_numpy()
    if isinstance(values, str):
        values = [values]
    return col.isin(values).to_numpy()


def buffered_polygon(polygon_4326, utm_crs, buffer_m=CONTEXT_BUFFER_M):
    import geopandas as gpd
    return gpd.GeoSeries([polygon_4326], crs="EPSG:4326").to_crs(utm_crs) \
        .buffer(buffer_m).to_crs("EPSG:4326").iloc[0]


def fetch_osm_context(polygon_4326, utm_crs, network: bool = False, notes: list | None = None):
    """Analiz alanı + tampon içindeki ana yol, otopark, şarj istasyonu ve POI'ler.
    Sunucuyu yormamak için hepsi TEK Overpass isteğiyle çekilip yerelde ayrıştırılır."""
    import geopandas as gpd
    from . import osm_client

    notes = notes if notes is not None else []
    big = buffered_polygon(polygon_4326, utm_crs)
    parking_vals = ["parking", "parking_entrance"]
    tags = {k: list(v) for k, v in POI_TAGS.items()}
    tags["highway"] = MAIN_ROAD_CLASSES
    tags["amenity"] = sorted(set(tags.get("amenity", []) + parking_vals + ["charging_station"]))
    try:
        allf = osm_client.features(big, tags)
    except ConnectionError as e:
        notes.append(f"{e} Yol/otopark/istasyon kriterleri hesaplanamadı (eşit kabul edildi).")
        allf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    empty = gpd.GeoDataFrame(geometry=[], crs=utm_crs)
    ctx = {"roads": empty, "parking": empty, "charging": empty, "poi": empty, "graph": None}
    if not allf.empty:
        allf = allf.to_crs(utm_crs)
        roads = allf[_tag_match(allf, "highway", MAIN_ROAD_CLASSES)]
        ctx["roads"] = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        ctx["parking"] = allf[_tag_match(allf, "amenity", parking_vals)]
        ctx["charging"] = allf[_tag_match(allf, "amenity", "charging_station")]
        poi_mask = np.zeros(len(allf), bool)
        for k, v in POI_TAGS.items():
            poi_mask |= _tag_match(allf, k, v)
        ctx["poi"] = allf[poi_mask]

    if network:
        import osmnx as ox
        try:
            ctx["graph"] = ox.project_graph(osm_client.graph(big), to_crs=utm_crs)
        except Exception:
            ctx["graph"] = None
    return ctx


# --------------------------------------------------------------------------- #
# Mesafeler
# --------------------------------------------------------------------------- #
def _sample_points(geoms, step: float = 5.0) -> np.ndarray:
    """Çizgi/poligon geometrilerini step metrede bir örneklenmiş noktalara çevirir."""
    import shapely
    if len(geoms) == 0:
        return np.empty((0, 2))
    arr = shapely.segmentize(np.asarray(geoms), step)
    pts = shapely.get_coordinates(arr)
    cents = shapely.get_coordinates(shapely.centroid(np.asarray(geoms)))
    return np.vstack([pts, cents])


def nearest_distance(buildings_utm, target_gdf, cap: float | None = None, step: float = 5.0):
    """Her bina sınırından hedef geometrilere en kısa Öklid mesafesi (m)."""
    import shapely
    from scipy.spatial import cKDTree

    n = len(buildings_utm)
    if target_gdf is None or len(target_gdf) == 0:
        return np.full(n, cap if cap is not None else np.nan)
    tree = cKDTree(_sample_points(target_gdf.geometry.values, step))
    bgeoms = shapely.segmentize(buildings_utm.geometry.values, step)
    coords, idx = shapely.get_coordinates(bgeoms, return_index=True)
    d, _ = tree.query(coords, k=1)
    out = np.array(pd.Series(d).groupby(idx).min().reindex(range(n)), dtype=float)  # yazılabilir kopya
    # Bina hedefle kesişiyorsa (ör. katlı otopark binası) mesafe 0
    tree_g = shapely.STRtree(np.asarray(target_gdf.geometry.values))
    hit, _ = tree_g.query(buildings_utm.geometry.values, predicate="intersects")
    out[np.unique(hit)] = 0.0
    if cap is not None:
        out = np.minimum(out, cap)
    return out


def network_distance(buildings_utm, target_gdf, G, cap: float):
    """Yol ağı üzerinden en yakın hedefe mesafe (m): bina→en yakın düğüm + ağ yolu."""
    import networkx as nx
    from scipy.spatial import cKDTree

    n = len(buildings_utm)
    if G is None or target_gdf is None or len(target_gdf) == 0:
        return None
    nodes = np.array([[d["x"], d["y"]] for _, d in G.nodes(data=True)])
    node_ids = np.array(list(G.nodes))
    tree = cKDTree(nodes)
    tp = target_gdf.geometry.representative_point()
    t_d, t_i = tree.query(np.c_[tp.x, tp.y])
    sources = set(node_ids[t_i])
    Gu = G.to_undirected()
    dist = nx.multi_source_dijkstra_path_length(Gu, sources, cutoff=cap, weight="length")
    bp = buildings_utm.geometry.representative_point()
    b_d, b_i = tree.query(np.c_[bp.x, bp.y])
    net = np.array([dist.get(node_ids[i], cap) for i in b_i])
    return np.minimum(b_d + net, cap)


def poi_density(buildings_utm, poi_gdf, radius: float = 500.0):
    from scipy.spatial import cKDTree
    if poi_gdf is None or len(poi_gdf) == 0:
        return np.zeros(len(buildings_utm))
    pp = poi_gdf.geometry.representative_point()
    tree = cKDTree(np.c_[pp.x, pp.y])
    c = buildings_utm.geometry.centroid
    return tree.query_ball_point(np.c_[c.x, c.y], r=radius, return_length=True).astype(float)


def compute_criteria(roofs, ctx, charging_cap: float = 2000.0, use_network: bool = False):
    g = roofs.copy()
    far = 5000.0
    g["Ana_Yol_Mesafe_m"] = nearest_distance(g, ctx["roads"], cap=far)
    g["Otopark_Mesafe_m"] = nearest_distance(g, ctx["parking"], cap=far)
    g["Sarj_Ist_Mesafe_m"] = nearest_distance(g, ctx["charging"], cap=charging_cap)
    if use_network and ctx.get("graph") is not None:
        for col, key, cap in (("Otopark_Mesafe_m", "parking", far),
                              ("Sarj_Ist_Mesafe_m", "charging", charging_cap)):
            nd = network_distance(g, ctx[key], ctx["graph"], cap)
            if nd is not None:
                g[col] = nd
    g["POI_Yogunlugu"] = poi_density(g, ctx["poi"])
    return g


# --------------------------------------------------------------------------- #
# ÇKKV
# --------------------------------------------------------------------------- #
def topsis(matrix: np.ndarray, weights, benefit) -> np.ndarray:
    """Klasik TOPSIS (vektör normalizasyonu). Skorlar 0–1."""
    X = np.asarray(matrix, float)
    w = np.asarray(weights, float)
    w = w / w.sum()
    b = np.asarray(benefit, bool)
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    V = X / norm * w
    best = np.where(b, V.max(axis=0), V.min(axis=0))
    worst = np.where(b, V.min(axis=0), V.max(axis=0))
    d_plus = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - worst) ** 2).sum(axis=1))
    denom = d_plus + d_minus
    return np.where(denom > 0, d_minus / np.where(denom > 0, denom, 1), 0.5)


def entropy_weights(matrix: np.ndarray, benefit) -> np.ndarray:
    """Shannon entropisi ile objektif ağırlıklar (veri ne kadar ayırt ediciyse o kadar yüksek)."""
    X = np.asarray(matrix, float)
    b = np.asarray(benefit, bool)
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    R = np.where(b, (X - mn) / rng, (mx - X) / rng) + 1e-9
    P = R / R.sum(axis=0)
    m = X.shape[0]
    e = -(P * np.log(P)).sum(axis=0) / np.log(max(m, 2))
    d = 1 - e
    d[np.ptp(X, axis=0) == 0] = 0  # sabit kriter ayırt edici değil
    if d.sum() == 0:
        return np.full(X.shape[1], 1 / X.shape[1])
    return d / d.sum()


def sensitivity(matrix, weights, benefit, names, top_k: int = 5, delta: float = 0.3, steps: int = 7):
    """Her kriterin ağırlığı ±delta oranında değiştirilir; ilk-k listesinin korunma oranı
    ve Spearman sıra korelasyonu raporlanır."""
    from scipy.stats import spearmanr
    w0 = np.asarray(weights, float) / np.sum(weights)
    base = topsis(matrix, w0, benefit)
    base_top = set(np.argsort(-base)[:top_k])
    rows = []
    for j, name in enumerate(names):
        for f in np.linspace(1 - delta, 1 + delta, steps):
            w = w0.copy()
            w[j] *= f
            w /= w.sum()
            s = topsis(matrix, w, benefit)
            top = set(np.argsort(-s)[:top_k])
            rho = spearmanr(base, s).statistic if len(s) > 2 else 1.0
            rows.append({"Kriter": name, "Agirlik_Degisimi_%": round((f - 1) * 100),
                         "IlkK_Korunma_%": 100 * len(top & base_top) / max(len(base_top), 1),
                         "Spearman_rho": float(np.nan_to_num(rho, nan=1.0))})
    return pd.DataFrame(rows)


CRITERIA = [
    # (sütun, görünen ad, fayda mı?)
    ("Yillik_Enerji_kWh", "Yıllık Enerji Üretimi", True),
    ("Ana_Yol_Mesafe_m", "Ana Yola Mesafe", False),
    ("Otopark_Mesafe_m", "Otoparka Mesafe", False),
    ("Sarj_Ist_Mesafe_m", "Mevcut Şarj İst. Mesafe", True),
    ("POI_Yogunlugu", "POI (Talep) Yoğunluğu", True),
]


def rank(g, weights: dict, weight_mode: str = "manuel"):
    """weights: {sütun: ağırlık}. Ağırlığı 0 olan kriterler dışlanır."""
    used = [(c, n, b) for c, n, b in CRITERIA if weights.get(c, 0) > 0]
    cols = [c for c, _, _ in used]
    X = np.array(g[cols], dtype=float)
    ben = [b for _, _, b in used]
    if weight_mode == "entropi":
        w = entropy_weights(X, ben)
    else:
        w = np.array([weights[c] for c in cols], float)
        w = w / w.sum()
    out = g.copy()
    out["Uygunluk_Skoru"] = topsis(X, w, ben)
    out = out.sort_values("Uygunluk_Skoru", ascending=False).reset_index(drop=True)
    out.insert(1, "Sira", np.arange(1, len(out) + 1))
    used_w = pd.DataFrame({"Kriter": [n for _, n, _ in used], "Sutun": cols,
                           "Yon": ["Fayda (max)" if b else "Maliyet (min)" for b in ben],
                           "Agirlik": w})
    sens = sensitivity(X, w, ben, [n for _, n, _ in used])
    return out, used_w, sens
