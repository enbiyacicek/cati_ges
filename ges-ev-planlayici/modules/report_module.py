"""Çıktılar: GeoJSON, CSV, Excel ve PDF rapor."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd

EXPORT_COLS = [
    "Sira", "Bina_ID", "Uygunluk_Skoru", "Cati_Alani_m2", "Panel_Alani_m2", "Kurulu_Guc_kWp",
    "Ozgul_Uretim_kWh_kWp", "Yillik_Enerji_kWh", "Yillik_Sarj_Edilen_Arac", "Gunluk_Tam_Sarj",
    "Yillik_EV_km", "Engellenen_CO2_Ton", "Ana_Yol_Mesafe_m", "Otopark_Mesafe_m",
    "Sarj_Ist_Mesafe_m", "POI_Yogunlugu", "Panel_Egimi", "Panel_Yonu", "Enlem", "Boylam",
]


def table(gdf) -> pd.DataFrame:
    g = gdf.copy()
    c = g.geometry.centroid.to_crs("EPSG:4326")
    g["Enlem"], g["Boylam"] = c.y.round(6), c.x.round(6)
    cols = [k for k in EXPORT_COLS if k in g.columns]
    df = pd.DataFrame(g[cols])
    return df.round(3)


def to_geojson(gdf) -> bytes:
    g = gdf.to_crs("EPSG:4326")
    keep = [k for k in EXPORT_COLS if k in g.columns and k not in ("Enlem", "Boylam")]
    return g[keep + ["geometry"]].to_json(drop_id=True, ensure_ascii=False).encode("utf-8")


def to_csv(gdf) -> bytes:
    return table(gdf).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def to_excel(gdf, summary: dict, params: dict, weights_df, sens_df) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        table(gdf).to_excel(xw, sheet_name="Sonuclar", index=False)
        pd.DataFrame(list(summary.items()), columns=["Gösterge", "Değer"]).to_excel(xw, sheet_name="Ozet", index=False)
        pd.DataFrame(list(params.items()), columns=["Parametre", "Değer"]).to_excel(xw, sheet_name="Parametreler", index=False)
        weights_df.to_excel(xw, sheet_name="TOPSIS_Agirliklar", index=False)
        sens_df.to_excel(xw, sheet_name="Duyarlilik", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200]) + 2
                ws.column_dimensions[col[0].column_letter].width = min(width, 40)
            ws.freeze_panes = "A2"
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for reg, bold in [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        (str(Path(__file__).parent.parent / "assets/DejaVuSans.ttf"),
         str(Path(__file__).parent.parent / "assets/DejaVuSans-Bold.ttf")),
    ]:
        if Path(reg).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("TRFont", reg))
            pdfmetrics.registerFont(TTFont("TRFont-Bold", bold))
            return "TRFont", "TRFont-Bold"
    return "Helvetica", "Helvetica-Bold"


def map_png(gdf, top_n: int = 5) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    gdf.plot(column="Uygunluk_Skoru", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
             edgecolor="#333", linewidth=0.2, legend=True,
             legend_kwds={"label": "TOPSIS Uygunluk Skoru", "shrink": 0.7})
    top = gdf.nsmallest(top_n, "Sira")
    c = top.geometry.centroid
    ax.scatter(c.x, c.y, s=120, marker="*", c="#0b5", edgecolors="black", zorder=5)
    for (x, y, s) in zip(c.x, c.y, top["Sira"]):
        ax.annotate(str(s), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def to_pdf(gdf, summary: dict, params: dict, weights_df, notes: list[str], top_n: int = 10) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    reg, bold = _font()
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName=bold, fontSize=16)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=12)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName=reg, fontSize=9, leading=12)

    def tbl(data, widths=None, fs=8):
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), reg), ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), fs),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e3d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f0")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="GES Destekli EV Şarj İstasyonu Yer Seçimi Raporu")
    el = [Paragraph("GES Destekli EV Şarj İstasyonu Yer Seçimi Raporu", h1),
          Paragraph(datetime.now().strftime("Oluşturulma: %d.%m.%Y %H:%M"), body), Spacer(1, 8),
          Paragraph("Özet", h2),
          tbl([["Gösterge", "Değer"]] + [[k, str(v)] for k, v in summary.items()], [9 * cm, 8 * cm], 9),
          Spacer(1, 10), Paragraph("Uygunluk Haritası", h2),
          Image(io.BytesIO(map_png(gdf)), width=16 * cm, height=12.8 * cm),
          PageBreak(), Paragraph(f"En Uygun {top_n} Bina", h2)]
    top = table(gdf).head(top_n)
    head = ["Sıra", "Bina", "Skor", "Çatı m²", "kWp", "MWh/yıl", "Araç/yıl", "CO₂ t", "Yol m", "Otopark m"]
    rows = [[int(r.Sira), int(r.Bina_ID), f"{r.Uygunluk_Skoru:.3f}", f"{r.Cati_Alani_m2:,.0f}",
             f"{r.Kurulu_Guc_kWp:,.1f}", f"{r.Yillik_Enerji_kWh / 1000:,.1f}",
             f"{r.Yillik_Sarj_Edilen_Arac:,.0f}", f"{r.Engellenen_CO2_Ton:,.1f}",
             f"{r.Ana_Yol_Mesafe_m:,.0f}", f"{r.Otopark_Mesafe_m:,.0f}"] for r in top.itertuples()]
    el += [tbl([head] + rows), Spacer(1, 10), Paragraph("TOPSIS Kriterleri ve Ağırlıkları", h2),
           tbl([["Kriter", "Yön", "Ağırlık"]] +
               [[r.Kriter, r.Yon, f"{r.Agirlik:.3f}"] for r in weights_df.itertuples()],
               [8 * cm, 5 * cm, 3 * cm], 9),
           Spacer(1, 10), Paragraph("Parametreler", h2),
           tbl([["Parametre", "Değer"]] + [[k, str(v)] for k, v in params.items()], [9 * cm, 8 * cm], 8),
           Spacer(1, 10), Paragraph("Yöntem", h2),
           Paragraph(
               "Çatılar YOLO11-seg örnek bölütleme modeli veya OpenStreetMap bina ayak izleriyle elde "
               "edilmiş, alanlar yerel UTM projeksiyonunda hesaplanmıştır. Kurulu güç = panel alanı × "
               "panel verimi (STC, 1 kW/m²); yıllık üretim, PVGIS (JRC) API'sinden alınan 1 kWp özgül "
               "üretim değeriyle hesaplanmıştır. Mesafeler OSM verisi üzerinden metre cinsinden ölçülmüş, "
               "binalar TOPSIS yöntemiyle 0–1 arasında puanlanmıştır.", body)]
    for n in notes:
        el.append(Paragraph("Not: " + n, body))
    doc.build(el)
    return buf.getvalue()
