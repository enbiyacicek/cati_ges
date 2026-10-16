"""GES Destekli EV Şarj İstasyonu Yer Seçimi – Web-CBS uygulaması (Streamlit)."""
from __future__ import annotations

import hmac
import tempfile

import folium
import streamlit as st
from folium.plugins import Draw, Fullscreen, MeasureControl
from shapely.geometry import shape
from streamlit_folium import st_folium

from modules import config, pipeline, report_module, vision_module
from modules.geo_utils import count_tiles, geodesic_area_km2
from modules.mcdm_module import CRITERIA

st.set_page_config(page_title="GES + EV Şarj Yer Seçimi", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
# Erişim şifresi (APP_PASSWORD ortam değişkeni tanımlıysa)
# --------------------------------------------------------------------------- #
def password_gate():
    if not config.APP_PASSWORD or st.session_state.get("auth"):
        return
    st.title("⚡ GES + EV Şarj Yer Seçimi")
    pw = st.text_input("Erişim şifresi", type="password")
    if pw:
        if hmac.compare_digest(pw, config.APP_PASSWORD):
            st.session_state.auth = True
            st.rerun()
        st.error("Şifre hatalı.")
    st.stop()


password_gate()

ss = st.session_state
ss.setdefault("center", [39.0, 35.0])
ss.setdefault("zoom", 6)
ss.setdefault("polygon", None)
ss.setdefault("result", None)

# --------------------------------------------------------------------------- #
# Kenar çubuğu – parametreler
# --------------------------------------------------------------------------- #
model_path = vision_module.find_model()
with st.sidebar:
    st.header("⚙️ Parametreler")

    with st.expander("🛰️ Çatı tespiti", expanded=True):
        detect_label = st.radio(
            "Yöntem", ["Otomatik", "YOLO11-seg", "OSM bina ayak izi"],
            help="Otomatik: models/ klasöründe model varsa YOLO, yoksa OSM kullanılır.")
        detect_mode = {"Otomatik": "otomatik", "YOLO11-seg": "yolo", "OSM bina ayak izi": "osm"}[detect_label]
        if model_path:
            st.success(f"Model bulundu: {model_path.name}")
        else:
            st.info("Eğitilmiş çatı modeli yok → OSM test modu kullanılacak.")
        source = st.selectbox("Uydu görüntüsü kaynağı", list(config.TILE_SOURCES))
        zoom = st.select_slider("Yakınlaştırma seviyesi", options=[17, 18, 19, 20], value=19)
        conf = st.slider("Model güven eşiği", 0.1, 0.9, 0.35, 0.05)
        uploaded = st.file_uploader("…veya kendi ortofotonuzu yükleyin (GeoTIFF)", type=["tif", "tiff"])
        min_area = st.number_input("Minimum çatı alanı (m²)", 5.0, 500.0, 25.0, 5.0)

    with st.expander("☀️ Güneş enerjisi", expanded=True):
        panel_eff = st.slider("Panel verimliliği (%)", 15.0, 25.0, 21.0, 0.5) / 100
        use_factor = st.slider("Çatı kullanım faktörü", 0.3, 0.9, 0.65, 0.05)
        loss = st.slider("Sistem kayıpları (%)", 5.0, 25.0, 14.0, 1.0)
        roof_label = st.radio("Çatı varsayımı", ["Düz çatı (optimum eğim)", "Eğimli çatı (yönelime göre)"])
        roof_mode = "duz" if roof_label.startswith("Düz") else "egimli"
        tilt = st.slider("Eğimli çatı açısı (°)", 10, 45, 25, disabled=roof_mode == "duz")
        fallback_ey = st.number_input("PVGIS'e ulaşılamazsa özgül üretim (kWh/kWp)", 800, 2200, 1400, 50)

    with st.expander("🚗 Elektrikli araç ve emisyon", expanded=True):
        battery = st.number_input("Ortalama batarya kapasitesi (kWh)", 20, 150, 60, 5)
        consumption = st.number_input("Ortalama tüketim (kWh/100 km)", 10.0, 30.0, 18.0, 0.5)
        co2 = st.number_input("Şebeke emisyon faktörü (kg CO₂/kWh)", 0.10, 1.00, 0.42, 0.01)

    with st.expander("📊 TOPSIS", expanded=True):
        weight_mode = st.radio("Ağırlıklandırma", ["Manuel", "Entropi (objektif)"], horizontal=True)
        defaults = {"Yillik_Enerji_kWh": 0.40, "Ana_Yol_Mesafe_m": 0.30, "Otopark_Mesafe_m": 0.15,
                    "Sarj_Ist_Mesafe_m": 0.15, "POI_Yogunlugu": 0.0}
        weights = {}
        if weight_mode == "Manuel":
            for col, name, benefit in CRITERIA:
                weights[col] = st.slider(f"{name} ({'fayda' if benefit else 'maliyet'})",
                                         0.0, 1.0, defaults[col], 0.05)
            tot = sum(weights.values())
            if tot == 0:
                st.error("En az bir kriterin ağırlığı 0'dan büyük olmalı.")
            else:
                st.caption("Normalize: " + " · ".join(f"{v / tot:.2f}" for v in weights.values()))
        else:
            chosen = st.multiselect("Kullanılacak kriterler", [n for _, n, _ in CRITERIA],
                                    default=[n for c, n, _ in CRITERIA if defaults[c] > 0])
            weights = {c: (1.0 if n in chosen else 0.0) for c, n, _ in CRITERIA}
        charging_cap = st.number_input("Şarj istasyonu mesafe üst sınırı (m)", 500, 10000, 2000, 250,
                                       help="Bu mesafeden uzaktaki binalar eşit kabul edilir.")
        use_network = st.checkbox("Otopark/istasyon mesafesinde yol ağı mesafesi kullan", value=False,
                                  help="Daha gerçekçi ama daha yavaş.")
        top_n = st.slider("Önerilecek istasyon sayısı", 1, 20, 5)

# --------------------------------------------------------------------------- #
# Başlık + yer arama
# --------------------------------------------------------------------------- #
st.title("⚡ GES Destekli EV Şarj İstasyonu Yer Seçimi")
st.caption("Haritada analiz alanını dikdörtgen veya poligon aracıyla çizin, ardından **Analizi Başlat**'a basın.")

c1, c2 = st.columns([4, 1])
with c1:
    query = st.text_input("Yer ara", placeholder="ör. Kampüs, mahalle veya ilçe adı",
                          label_visibility="collapsed")
with c2:
    if st.button("🔍 Git", width="stretch") and query:
        try:
            import osmnx as ox
            lat, lon = ox.geocode(query)
            ss.center, ss.zoom = [lat, lon], 16
        except Exception:
            st.warning("Yer bulunamadı.")


def base_map(center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, max_zoom=21)
    for name, url in config.TILE_SOURCES.items():
        folium.TileLayer(url, attr=config.TILE_ATTRIBUTION[name], name=name,
                         max_zoom=21, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    Fullscreen().add_to(m)
    return m


# --------------------------------------------------------------------------- #
# Alan seçimi haritası
# --------------------------------------------------------------------------- #
if ss.result is None:
    m = base_map(ss.center, ss.zoom)
    Draw(export=False, position="topleft",
         draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                       "rectangle": {"shapeOptions": {"color": "#ffcc00"}},
                       "polygon": {"shapeOptions": {"color": "#ffcc00"}}},
         edit_options={"edit": False}).add_to(m)
    MeasureControl(primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)
    folium.LayerControl().add_to(m)
    out = st_folium(m, key="draw_map", height=560, use_container_width=True,
                    returned_objects=["last_active_drawing"])
    drawing = (out or {}).get("last_active_drawing")
    if drawing and drawing.get("geometry"):
        ss.polygon = shape(drawing["geometry"])

    if ss.polygon is not None:
        poly = ss.polygon
        area = geodesic_area_km2(poly)
        tiles = count_tiles(poly.bounds, zoom)
        a, b, c = st.columns(3)
        a.metric("Seçilen alan", f"{area:,.3f} km²")
        b.metric("Gereken uydu karosu", f"{tiles:,}" if detect_mode != "osm" else "—")
        c.metric("Alan sınırı", f"{config.MAX_AREA_KM2} km²")
        too_big = area > config.MAX_AREA_KM2
        if too_big:
            st.error("Seçilen alan çok büyük. Lütfen daha küçük bir alan çizin.")
        if st.button("🚀 Analizi Başlat", type="primary", disabled=too_big or sum(weights.values()) == 0):
            params = dict(detect_mode=detect_mode, source=source, zoom=zoom, conf=conf,
                          min_area=min_area, panel_eff=panel_eff, use_factor=use_factor, loss=loss,
                          roof_mode=roof_mode, tilt=tilt, fallback_ey=fallback_ey,
                          battery_kwh=battery, ev_consumption=consumption, co2_factor=co2,
                          weights=weights, weight_mode="entropi" if weight_mode.startswith("Entropi") else "manuel",
                          charging_cap=charging_cap, use_network=use_network)
            tif = None
            if uploaded is not None:
                tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                with open(tif, "wb") as f:
                    f.write(uploaded.getbuffer())
            done = False
            with st.status("Analiz çalışıyor…", expanded=True) as status:
                bar = st.progress(0.0)
                try:
                    res = pipeline.run_analysis(
                        poly, params, step=lambda msg: (status.write(msg), bar.progress(0.0)),
                        progress=lambda r: bar.progress(min(max(r, 0.0), 1.0)), uploaded_tif=tif)
                    status.update(label="Analiz tamamlandı ✔", state="complete", expanded=False)
                    ss.result, ss.params, ss.top_n = res, params, top_n
                    done = True
                except Exception as e:
                    status.update(label="Analiz başarısız", state="error")
                    st.exception(e)
            if done:
                st.rerun()
    else:
        st.info("👆 Haritanın sol üstündeki ▭ (dikdörtgen) veya ⬠ (poligon) aracıyla alan çizin.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sonuçlar
# --------------------------------------------------------------------------- #
res: pipeline.Result = ss.result
g = res.gdf
top_n = ss.top_n
summary = pipeline.summarize(res, top_n)

if st.button("↩️ Yeni analiz"):
    ss.result = None
    ss.polygon = None
    st.rerun()

for n in res.notes:
    st.warning(n)

k = st.columns(5)
k[0].metric("Çatı sayısı", f"{len(g):,}")
k[1].metric("Kurulu güç", f"{g['Kurulu_Guc_kWp'].sum() / 1000:,.2f} MWp")
k[2].metric("Yıllık üretim", f"{g['Yillik_Enerji_kWh'].sum() / 1000:,.0f} MWh")
k[3].metric("Şarj (tam/yıl)", f"{g['Yillik_Sarj_Edilen_Arac'].sum():,.0f}")
k[4].metric("Engellenen CO₂", f"{g['Engellenen_CO2_Ton'].sum():,.0f} t/yıl")

tab_map, tab_table, tab_sens, tab_method = st.tabs(
    ["🗺️ Harita", "📋 Sonuç tablosu", "🎯 Duyarlılık", "📖 Yöntem"])

with tab_map:
    import branca.colormap as cm

    g4326 = g.to_crs("EPSG:4326")
    b = g4326.total_bounds
    m = base_map([(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], 17)
    cmap = cm.LinearColormap(["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
                             vmin=0, vmax=1, caption="TOPSIS Uygunluk Skoru")
    cols = ["Sira", "Bina_ID", "Uygunluk_Skoru", "Cati_Alani_m2", "Yillik_Enerji_kWh",
            "Yillik_Sarj_Edilen_Arac", "geometry"]
    gj = g4326[cols].copy()
    for c in ("Uygunluk_Skoru",):
        gj[c] = gj[c].round(3)
    for c in ("Cati_Alani_m2", "Yillik_Enerji_kWh", "Yillik_Sarj_Edilen_Arac"):
        gj[c] = gj[c].round(0)
    folium.GeoJson(
        gj.to_json(), name="Çatılar (uygunluk)",
        style_function=lambda f: {"fillColor": cmap(f["properties"]["Uygunluk_Skoru"]),
                                  "color": "#222", "weight": 0.5, "fillOpacity": 0.75},
        highlight_function=lambda f: {"weight": 2.5, "color": "#00e5ff"},
        tooltip=folium.GeoJsonTooltip(
            fields=["Sira", "Uygunluk_Skoru", "Cati_Alani_m2", "Yillik_Enerji_kWh"],
            aliases=["Sıra", "Skor", "Çatı alanı (m²)", "Yıllık üretim (kWh)"])).add_to(m)
    cmap.add_to(m)

    fg = folium.FeatureGroup(name=f"Önerilen {top_n} istasyon").add_to(m)
    top = g4326.head(top_n)
    for r, pt in zip(top.itertuples(), top.geometry.representative_point()):
        html = f"""
        <div style="font-family:sans-serif;font-size:13px;min-width:230px">
          <b style="color:#1a9641">⚡ Optimum EV Şarj İstasyonu Önerisi #{r.Sira}</b><hr style="margin:4px 0">
          <b>Bina ID:</b> {r.Bina_ID}<br>
          <b>TOPSIS skoru:</b> {r.Uygunluk_Skoru:.3f}<br>
          <b>Çatı alanı:</b> {r.Cati_Alani_m2:,.0f} m²<br>
          <b>Kurulu güç:</b> {r.Kurulu_Guc_kWp:,.1f} kWp<br>
          <b>Yıllık üretim:</b> {r.Yillik_Enerji_kWh:,.0f} kWh<br>
          <b>Şarj edilebilir araç:</b> {r.Yillik_Sarj_Edilen_Arac:,.0f} / yıl
              (~{r.Gunluk_Tam_Sarj:.1f} / gün)<br>
          <b>Engellenen CO₂:</b> {r.Engellenen_CO2_Ton:,.1f} t/yıl<br>
          <b>Ana yola:</b> {r.Ana_Yol_Mesafe_m:,.0f} m · <b>Otoparka:</b> {r.Otopark_Mesafe_m:,.0f} m
        </div>"""
        folium.Marker([pt.y, pt.x], popup=folium.Popup(html, max_width=320),
                      tooltip=f"Öneri #{r.Sira}",
                      icon=folium.Icon(color="green", icon="bolt", prefix="fa")).add_to(fg)

    if len(res.ctx.get("charging", [])):
        ex = folium.FeatureGroup(name="Mevcut şarj istasyonları").add_to(m)
        for pt in res.ctx["charging"].to_crs("EPSG:4326").geometry.representative_point():
            folium.CircleMarker([pt.y, pt.x], radius=6, color="#1565c0", fill=True,
                                fill_opacity=0.9, tooltip="Mevcut şarj istasyonu").add_to(ex)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, key="result_map", height=620, use_container_width=True, returned_objects=[])

with tab_table:
    df = report_module.table(g)
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"Uygunluk_Skoru": st.column_config.ProgressColumn(
                     "Uygunluk", min_value=0.0, max_value=1.0, format="%.3f")})
    st.subheader("Kullanılan TOPSIS ağırlıkları")
    st.dataframe(res.weights.style.format({"Agirlik": "{:.3f}"}), hide_index=True)
    st.subheader(f"İlk {min(15, len(g))} binanın yıllık üretimi (MWh)")
    chart = df.head(15).assign(Bina=lambda d: "#" + d["Sira"].astype(str) + " (ID " + d["Bina_ID"].astype(str) + ")")
    st.bar_chart(chart.set_index("Bina")["Yillik_Enerji_kWh"] / 1000)

with tab_sens:
    st.markdown(f"Her kriterin ağırlığı ±%30 değiştirildiğinde **ilk {min(5, len(g))} öneri listesinin "
                "ne kadarının korunduğu**. Çizgiler 100'e yakınsa sonuç ağırlık seçimine karşı sağlamdır.")
    sens = res.sensitivity
    st.line_chart(sens.pivot(index="Agirlik_Degisimi_%", columns="Kriter", values="IlkK_Korunma_%"))
    st.dataframe(sens.groupby("Kriter").agg(
        En_Dusuk_Korunma_Yuzdesi=("IlkK_Korunma_%", "min"),
        En_Dusuk_Spearman=("Spearman_rho", "min")).round(3), width="stretch")

with tab_method:
    pv = res.pvgis
    st.markdown(f"""
**1. Çatı tespiti:** {res.mode_used}. Poligonlar yerel UTM projeksiyonuna (`{res.utm_crs.to_string() if res.utm_crs else ''}`)
çevrilip alanları hesaplandı; {ss.params['min_area']:.0f} m²'den küçükler elendi.

**2. GES hesabı (PVGIS {pv['api']}):** 1 kWp sistem için optimum eğimde (≈{pv['angle']:.0f}°) yıllık özgül üretim
**{pv['E_y']:,.0f} kWh/kWp**, sistem kaybı %{ss.params['loss']:.0f}.

- Panel alanı = Çatı alanı × Kullanım faktörü ({ss.params['use_factor']})
- Kurulu güç (kWp) = Panel alanı × Panel verimi ({ss.params['panel_eff']:.2%}) × 1 kW/m²
- Yıllık enerji = Kurulu güç × Özgül üretim
- Şarj edilen araç = Yıllık enerji / {ss.params['battery_kwh']} kWh
- CO₂ = Yıllık enerji × {ss.params['co2_factor']} kg/kWh

**3. Mekânsal kriterler:** OpenStreetMap'ten ana yollar, otoparklar, mevcut şarj istasyonları ve talep noktaları
(analiz alanı + {config.CONTEXT_BUFFER_M} m tampon) çekildi. Mesafeler bina sınırından ölçüldü
({'yol ağı' if ss.params['use_network'] else 'Öklid'}).

**4. TOPSIS:** Vektör normalizasyonu, ağırlıklandırılmış ideal/anti-ideal çözümlere uzaklık; skor = D⁻/(D⁺+D⁻).
""")
    st.line_chart({"Aylık üretim (kWh/kWp)": pv["monthly"]})

# --------------------------------------------------------------------------- #
# İndirmeler
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("📥 İndir")
params_view = {
    "Tespit yöntemi": res.mode_used, "Panel verimi": f"{ss.params['panel_eff']:.1%}",
    "Kullanım faktörü": ss.params["use_factor"], "Sistem kaybı (%)": ss.params["loss"],
    "Çatı varsayımı": "Düz" if ss.params["roof_mode"] == "duz" else f"Eğimli ({ss.params['tilt']}°)",
    "Batarya (kWh)": ss.params["battery_kwh"], "Tüketim (kWh/100 km)": ss.params["ev_consumption"],
    "CO₂ faktörü (kg/kWh)": ss.params["co2_factor"], "Ağırlıklandırma": ss.params["weight_mode"],
    "Mesafe türü": "Yol ağı" if ss.params["use_network"] else "Öklid",
}
d = st.columns(4)
d[0].download_button("GeoJSON", report_module.to_geojson(g), "ges_ev_sonuclar.geojson",
                     "application/geo+json", width="stretch")
d[1].download_button("CSV", report_module.to_csv(g), "ges_ev_sonuclar.csv", "text/csv",
                     width="stretch")
d[2].download_button("Excel", report_module.to_excel(g, summary, params_view, res.weights, res.sensitivity),
                     "ges_ev_sonuclar.xlsx",
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     width="stretch")


@st.cache_data(show_spinner="PDF hazırlanıyor…")
def _pdf(res_id: int):
    return report_module.to_pdf(g, summary, params_view, res.weights, res.notes, top_n=max(top_n, 10))


d[3].download_button("PDF rapor", _pdf(id(res)), "ges_ev_rapor.pdf", "application/pdf",
                     width="stretch")
