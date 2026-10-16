"""PVGIS API ile GES potansiyeli ve EV şarj kapasitesi hesabı."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import requests

from .config import CACHE_DIR, PVGIS_URLS, USER_AGENT


class PVGISError(RuntimeError):
    pass


@lru_cache(maxsize=512)
def _pvgis_cached(lat: float, lon: float, loss: float, angle, aspect, optimal: bool):
    cache_file = CACHE_DIR / f"pvgis_{lat}_{lon}_{loss}_{angle}_{aspect}_{optimal}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    params = {"lat": lat, "lon": lon, "peakpower": 1, "loss": loss, "outputformat": "json"}
    if optimal:
        params["optimalangles"] = 1
    else:
        params["angle"] = angle
        params["aspect"] = aspect
    last_err = None
    for url in PVGIS_URLS:
        try:
            r = requests.get(url, params=params, timeout=40, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                d = r.json()
                fixed = d["outputs"]["totals"]["fixed"]
                mounting = d["inputs"]["mounting_system"]["fixed"]
                res = {
                    "E_y": float(fixed["E_y"]),
                    "H_i_y": float(fixed.get("H(i)_y", np.nan)),
                    "monthly": [float(m["E_m"]) for m in d["outputs"]["monthly"]["fixed"]],
                    "angle": float(mounting["slope"]["value"]),
                    "aspect": float(mounting["azimuth"]["value"]),
                    "api": url.split("/api/")[1].split("/")[0],
                }
                cache_file.write_text(json.dumps(res))
                return res
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.RequestException, KeyError, ValueError) as e:
            last_err = str(e)
    raise PVGISError(last_err)


def get_pvgis(lat, lon, loss=14.0, angle=None, aspect=None, optimal=False):
    """1 kWp sistemin yıllık üretimi (E_y, kWh/kWp) ve aylık değerleri."""
    return _pvgis_cached(round(lat, 3), round(lon, 3), float(loss),
                         None if angle is None else round(float(angle)),
                         None if aspect is None else int(round(aspect / 15.0) * 15),
                         bool(optimal))


# --------------------------------------------------------------------------- #
# Çatı yönelimi
# --------------------------------------------------------------------------- #
def roof_faces(geom_utm, tilt: float):
    """Eğimli çatı varsayımı: mahya (ridge) en küçük döndürülmüş dikdörtgenin
    uzun kenarı boyuncadır. İki yüzeyin PVGIS aspect değerlerini (0=Güney,
    90=Batı, -90=Doğu) ve kuzeye bakmayan yüzeylerin listesini döndürür."""
    rect = geom_utm.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    e1 = (xs[1] - xs[0], ys[1] - ys[0])
    e2 = (xs[2] - xs[1], ys[2] - ys[1])
    long_e = e1 if math.hypot(*e1) >= math.hypot(*e2) else e2
    ridge = math.degrees(math.atan2(long_e[0], long_e[1])) % 180  # pusula, 0=K
    faces = []
    for compass in ((ridge + 90) % 360, (ridge + 270) % 360):
        aspect = ((compass - 180 + 180) % 360) - 180  # PVGIS: 0 = güney
        if abs(aspect) <= 100:  # kuzeye bakan yüzey kullanılmaz
            faces.append(aspect)
    return ridge, faces


def compute_solar(roofs_utm, center_latlon, p: dict, progress_cb=None):
    """p: panel_eff, use_factor, loss, roof_mode('duz'|'egimli'), tilt, battery_kwh,
    ev_consumption (kWh/100 km), co2_factor (kg/kWh), fallback_ey."""
    g = roofs_utm.copy()
    lat, lon = center_latlon
    notes = []
    kwp_per_m2 = p["panel_eff"]  # STC: 1000 W/m² -> verim × 1 kW/m²

    state = {"down": False}

    def ey_or_fallback(**kw):
        try:
            if state["down"]:  # API bir kez düştüyse her çatı için tekrar bekleme
                raise PVGISError("önceki istek başarısız")
            return get_pvgis(lat, lon, loss=p["loss"], **kw)
        except PVGISError as e:
            state["down"] = True
            if notes:
                return {"E_y": p["fallback_ey"], "monthly": [p["fallback_ey"] / 12] * 12,
                        "angle": p.get("tilt", 30), "aspect": 0, "api": "yedek"}
            notes.append(f"PVGIS'e erişilemedi ({e}); yedek değer {p['fallback_ey']} kWh/kWp kullanıldı.")
            return {"E_y": p["fallback_ey"], "monthly": [p["fallback_ey"] / 12] * 12,
                    "angle": p.get("tilt", 30), "aspect": 0, "api": "yedek"}

    base = ey_or_fallback(optimal=True)
    if p["roof_mode"] == "duz":
        g["Panel_Alani_m2"] = g["Cati_Alani_m2"] * p["use_factor"]
        g["Kurulu_Guc_kWp"] = g["Panel_Alani_m2"] * kwp_per_m2
        g["Ozgul_Uretim_kWh_kWp"] = base["E_y"]
        g["Panel_Egimi"] = base["angle"]
        g["Panel_Yonu"] = "Güney (optimum)"
    else:
        tilt = p["tilt"]
        cos_t = math.cos(math.radians(tilt))
        kwp, ey, yon = [], [], []
        n = len(g)
        for k, geom in enumerate(g.geometry, 1):
            _, faces = roof_faces(geom, tilt)
            face_area = geom.area / 2 / cos_t * p["use_factor"]
            tot_kwp, tot_e = 0.0, 0.0
            for a in faces:
                r = ey_or_fallback(angle=tilt, aspect=a)
                k_ = face_area * kwp_per_m2
                tot_kwp += k_
                tot_e += k_ * r["E_y"]
            kwp.append(tot_kwp)
            ey.append(tot_e / tot_kwp if tot_kwp else 0.0)
            yon.append(", ".join(f"{a:+.0f}°" for a in faces) or "uygun yüzey yok")
            if progress_cb:
                progress_cb(k / n)
        g["Panel_Alani_m2"] = np.array(kwp) / kwp_per_m2
        g["Kurulu_Guc_kWp"] = kwp
        g["Ozgul_Uretim_kWh_kWp"] = ey
        g["Panel_Egimi"] = tilt
        g["Panel_Yonu"] = yon

    g["Yillik_Enerji_kWh"] = g["Kurulu_Guc_kWp"] * g["Ozgul_Uretim_kWh_kWp"]
    g["Yillik_Sarj_Edilen_Arac"] = g["Yillik_Enerji_kWh"] / p["battery_kwh"]
    g["Gunluk_Tam_Sarj"] = g["Yillik_Sarj_Edilen_Arac"] / 365.0
    g["Yillik_EV_km"] = g["Yillik_Enerji_kWh"] / p["ev_consumption"] * 100.0
    g["Engellenen_CO2_Ton"] = g["Yillik_Enerji_kWh"] * p["co2_factor"] / 1000.0
    return g, base, sorted(set(notes))
