"""Uçtan uca analiz akışı. Arayüzden bağımsızdır (CLI/test için de kullanılabilir)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

from . import imagery, mcdm_module, solar_module, vision_module


@dataclass
class Result:
    gdf: object
    utm_crs: object
    pvgis: dict
    weights: object
    sensitivity: object
    ctx: dict
    mode_used: str
    notes: list = field(default_factory=list)


def _noop(*_a, **_k):
    pass


def run_analysis(polygon_4326, p: dict, step=_noop, progress=_noop, uploaded_tif: str | None = None):
    """step(msg) aşama bildirir, progress(oran) ilerleme çubuğunu günceller."""
    notes = []
    mode = p["detect_mode"]  # "otomatik" | "yolo" | "osm"
    model_path = vision_module.find_model()
    if mode == "yolo" and model_path is None:
        raise RuntimeError("YOLO modu seçildi ama models/ klasöründe model (.onnx/.pt) yok.")
    use_yolo = mode == "yolo" or (mode == "otomatik" and model_path is not None)

    # 1-2) Çatı tespiti
    if use_yolo:
        if uploaded_tif:
            step("Yüklenen ortofoto okunuyor…")
            img, transform, crs = imagery.read_geotiff(uploaded_tif)
        else:
            step(f"Uydu görüntüsü indiriliyor ({p['source']}, zoom {p['zoom']})…")
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
            _, img, transform, crs, missing = imagery.download_geotiff(
                polygon_4326.bounds, tmp, zoom=p["zoom"], source=p["source"], progress_cb=progress)
            if missing:
                notes.append(f"{missing} uydu karosu indirilemedi; bu kısımlarda çatı tespit edilemedi.")
        step(f"YOLO11-seg ile çatılar bölütleniyor ({model_path.name})…")
        model = vision_module.load_model(model_path, conf=p.get("conf", 0.35))
        raw = vision_module.detect_roofs_yolo(img, transform, crs, model, progress_cb=progress)
        mode_used = f"YOLO11-seg ({model_path.name})"
    else:
        step("OpenStreetMap bina ayak izleri çekiliyor (test modu)…")
        raw = vision_module.detect_roofs_osm(polygon_4326)
        mode_used = "OSM bina ayak izleri"
        if mode == "otomatik":
            notes.append("Eğitilmiş çatı modeli bulunmadığı için OSM bina ayak izleri kullanıldı.")

    roofs, utm = vision_module.prepare_roofs(raw, polygon_4326, p["min_area"])
    if roofs is None or len(roofs) == 0:
        raise RuntimeError("Seçilen alanda uygun büyüklükte çatı bulunamadı.")
    step(f"{len(roofs)} çatı bulundu. PVGIS ile güneş potansiyeli hesaplanıyor…")

    # 3) GES
    c = polygon_4326.centroid
    roofs, pv, pv_notes = solar_module.compute_solar(roofs, (c.y, c.x), p, progress_cb=progress)
    notes += pv_notes

    # 4) Mekânsal bağlam + TOPSIS
    step("Yollar, otoparklar, şarj istasyonları ve POI'ler OSM'den çekiliyor…")
    ctx = mcdm_module.fetch_osm_context(polygon_4326, utm, network=p["use_network"])
    if p["use_network"] and ctx.get("graph") is None:
        notes.append("Yol ağı indirilemedi; Öklid mesafesi kullanıldı.")
    if len(ctx["charging"]) == 0:
        notes.append("Çevrede kayıtlı şarj istasyonu yok; bu kriter tüm binalar için eşit kabul edildi.")
    step("Mesafeler hesaplanıyor ve TOPSIS uygulanıyor…")
    roofs = mcdm_module.compute_criteria(roofs, ctx, charging_cap=p["charging_cap"],
                                         use_network=p["use_network"])
    ranked, wdf, sens = mcdm_module.rank(roofs, p["weights"], p["weight_mode"])
    return Result(ranked, utm, pv, wdf, sens, ctx, mode_used, notes)


def summarize(res: Result, top_n: int = 5) -> dict:
    g = res.gdf
    top = g.head(top_n)
    return {
        "Tespit yöntemi": res.mode_used,
        "Analiz edilen çatı sayısı": f"{len(g):,}",
        "Toplam çatı alanı (m²)": f"{g['Cati_Alani_m2'].sum():,.0f}",
        "Toplam kurulu güç (MWp)": f"{g['Kurulu_Guc_kWp'].sum() / 1000:,.2f}",
        "Toplam yıllık üretim (MWh)": f"{g['Yillik_Enerji_kWh'].sum() / 1000:,.1f}",
        "Yıllık şarj edilebilecek araç (tam şarj)": f"{g['Yillik_Sarj_Edilen_Arac'].sum():,.0f}",
        "Engellenen CO₂ (ton/yıl)": f"{g['Engellenen_CO2_Ton'].sum():,.1f}",
        "PVGIS özgül üretim (kWh/kWp, optimum)": f"{res.pvgis['E_y']:,.0f} ({res.pvgis['api']})",
        f"İlk {top_n} bina toplam üretim (MWh)": f"{top['Yillik_Enerji_kWh'].sum() / 1000:,.1f}",
    }
