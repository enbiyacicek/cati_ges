"""Uygulama genel ayarları. Ortam değişkenleriyle (env) değiştirilebilir."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = Path(os.environ.get("GES_CACHE_DIR", "/tmp/ges_ev_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Uydu görüntüsü karo kaynakları ({z}/{x}/{y} XYZ şeması)
TILE_SOURCES = {
    "Esri World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
}
TILE_ATTRIBUTION = {
    "Esri World Imagery": "Tiles &copy; Esri, Maxar, Earthstar Geographics",
    "Google Satellite": "Imagery &copy; Google",
}

# Analiz sınırları (ücretsiz CPU sunucusunda çökmemek için)
MAX_AREA_KM2 = float(os.environ.get("MAX_AREA_KM2", "2.0"))
MAX_TILES = int(os.environ.get("MAX_TILES", "1200"))

# OSM bağlam verisi için analiz alanının dışına eklenecek tampon (m)
CONTEXT_BUFFER_M = 600

# PVGIS
PVGIS_URLS = [
    "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc",
    "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc",
]

# Ana yol sınıfları (OSM highway etiketi)
MAIN_ROAD_CLASSES = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
]

# Talep çeken noktalar (POI) – şarj talebi göstergesi
POI_TAGS = {
    "amenity": ["school", "university", "college", "hospital", "clinic",
                "marketplace", "fuel", "townhall", "library", "place_of_worship",
                "restaurant", "cafe", "bank", "cinema"],
    "shop": ["mall", "supermarket", "department_store"],
    "tourism": ["hotel", "museum", "attraction"],
    "leisure": ["sports_centre", "stadium", "park"],
}

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
USER_AGENT = "GES-EV-Planlayici/1.0 (academic research; contact via app owner)"
