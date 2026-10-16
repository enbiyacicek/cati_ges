# ⚡ GES Destekli EV Şarj İstasyonu Yer Seçimi (Web-CBS)

Haritada seçilen bölgedeki bina çatılarını tespit eden, PVGIS ile güneş enerjisi potansiyelini
hesaplayan ve TOPSIS ile en uygun elektrikli araç şarj istasyonu konumlarını öneren web uygulaması.
Hiçbir masaüstü CBS yazılımı gerekmez; tarayıcıdan çalışır.

## Akış

1. **Alan seçimi** – Haritada dikdörtgen/poligon çizilir (yer arama kutusu ile konuma gidilebilir).
2. **Çatı tespiti**
   - `models/` klasöründe model varsa: uydu görüntüsü (Esri / Google, zoom 19) GeoTIFF olarak indirilir,
     örtüşmeli 640 px pencerelerle **YOLO11-seg** çalıştırılır, maskeler `rasterio.features.shapes`
     ile poligona çevrilir, pencere kenarında bölünen çatılar birleştirilir.
   - Model yoksa: **OpenStreetMap bina ayak izleri** (test modu).
   - Kendi ortofotonuzu (GeoTIFF) yükleyebilirsiniz.
3. **GES hesabı** – PVGIS v5.3 (yoksa v5.2) API'si. Düz çatı: optimum eğim. Eğimli çatı: mahya yönü
   poligondan hesaplanır, kuzeye bakmayan yüzeyler kendi yönleriyle PVGIS'e sorulur.
4. **Mekânsal analiz + TOPSIS** – OSM'den ana yollar, otoparklar, mevcut şarj istasyonları ve talep
   noktaları (POI) çekilir; bina sınırından mesafeler (Öklid veya yol ağı) hesaplanır. Ağırlıklar
   manuel veya **Entropi** yöntemiyle; ±%30 **duyarlılık analizi** otomatik yapılır.
5. **Çıktılar** – Skora göre renklendirilmiş harita, öneri ikonları ve açılır pencereler;
   GeoJSON, CSV, Excel ve PDF rapor.

### Formüller
| Büyüklük | Formül |
|---|---|
| Panel alanı (m²) | Çatı alanı × Kullanım faktörü |
| Kurulu güç (kWp) | Panel alanı × Panel verimi (STC, 1 kW/m²) |
| Yıllık enerji (kWh) | Kurulu güç × PVGIS E_y (kWh/kWp) |
| Şarj edilen araç (tam şarj/yıl) | Yıllık enerji / Batarya kapasitesi |
| EV menzili (km/yıl) | Yıllık enerji / Tüketim × 100 |
| Engellenen CO₂ (t) | Yıllık enerji × Emisyon faktörü / 1000 |

> %21 verimde 1 kWp ≈ 4,76 m² panel alanı eder (orijinal taslaktaki sabit 5 m²/kWp'ye denk).

## Ücretsiz yayınlama: Streamlit Community Cloud

Docker gerekmez, ücretsizdir. İnternet olan her bilgisayar ve telefondan bir link ile açılır.

1. **GitHub hesabı açın:** <https://github.com> → *Sign up*.
2. **Yeni depo (repository) oluşturun:** sağ üstte **+ → New repository** → ad verin
   (ör. `ges-ev-planlayici`) → **Public** → *Create repository*.
3. **Dosyaları yükleyin:** depo sayfasında **uploading an existing file** bağlantısına tıklayın,
   zip'ten çıkardığınız klasörün **içindeki** tüm dosya ve klasörleri sürükleyip bırakın
   (`app.py`, `requirements.txt`, `packages.txt`, `modules/`, `training/`, `models/`, `.streamlit/`)
   → *Commit changes*.
   > `.streamlit` gibi noktayla başlayan klasörler gizli olabilir; görünmüyorsa Windows'ta
   > Dosya Gezgini → Görünüm → *Gizli öğeler*'i açın.
4. **Uygulamayı yayınlayın:** <https://share.streamlit.io> → *Continue with GitHub* ile giriş →
   **Create app** → *Deploy a public app from GitHub* →
   - Repository: `kullanıcı-adınız/ges-ev-planlayici`
   - Branch: `main`
   - Main file path: `app.py`
   - **Advanced settings → Python version: 3.11**
   → **Deploy**. İlk kurulum 5–10 dakika sürer.
5. Bitince uygulama `https://<sizin-adınız>.streamlit.app` adresinde açılır; bu linki paylaşın.

**İsteğe bağlı ayarlar** (uygulama sayfası → ⋮ → *Settings → Secrets*):
```toml
APP_PASSWORD = "gizli-sifre"   # uygulamaya şifre koyar
MAX_AREA_KM2 = "2"             # en büyük analiz alanı (km²)
```

**Ücretsiz katmanın sınırları:** bellek yaklaşık 1 GB'dır; bu yüzden alan sınırı varsayılan 2 km²'dir.
12 saat ziyaret almayan uygulama uyku moduna geçer, sonraki ziyaretçi "uyandır" düğmesine basar
(~1 dk). Kodu güncellemek için GitHub'daki dosyayı değiştirmeniz yeterlidir; uygulama kendiliğinden
yenilenir.

> Daha güçlü bir sunucuya (VPS, üniversite sunucusu) taşımak isterseniz `Dockerfile` hazırdır.

## Kendi bilgisayarınızda çalıştırma
```bash
pip install -r requirements.txt
streamlit run app.py
```
veya Docker ile: `docker build -t ges-ev . && docker run -p 7860:7860 ges-ev` → <http://localhost:7860>

## Çatı modelini eğitme (YOLO11-seg)

Hazır bir "çatı" modeli olmadığı için iki betik eklendi. **GPU gerektirir** – Google Colab (ücretsiz T4)
yeterlidir.

```bash
pip install -r requirements-train.txt

# 1) OSM binaları + uydu görüntüsünden etiketli veri seti üret
python training/prepare_dataset.py --place "Ortahisar, Trabzon" --place "Atakum, Samsun" \
       --place "Kadıköy, İstanbul" --max-km2 1 --max-parts 15 --out dataset

# 2) Eğit ve ONNX'e çevir  →  models/roof_yolo11.onnx
python training/train.py --data dataset/data.yaml --model yolo11s-seg.pt --epochs 100
```
Sonra `models/roof_yolo11.onnx` dosyasını GitHub deposundaki `models/` klasörüne yükleyin; uygulama otomatik olarak YOLO moduna
geçer (ONNX, PyTorch olmadan CPU'da çalışır).

**Kalite ipuçları:** OSM ile görüntü arasında birkaç metre kayma olabilir. Daha iyi sonuç için
etiketleri CVAT veya Roboflow'da gözden geçirin; farklı şehir dokularından (yoğun konut, sanayi,
kampüs, kırsal) örnekler ekleyin. Eğitimden sonra `mAP50` (maske) değerini raporlayın.

## Bilinen sınırlamalar
- **Görüntü lisansı:** Google ve Esri karolarının toplu indirilmesi kullanım şartlarıyla sınırlıdır.
  Akademik/prototip kullanım dışında kendi ortofotonuzu (belediye, HGM vb.) yükleme seçeneğini kullanın.
- Gölgelenme (komşu bina, ağaç) ve çatı üstü engeller (baca, klima, su deposu) kullanım faktörüyle
  yaklaşık olarak temsil edilir.
- Şebeke/trafo kapasitesi ve arazi mülkiyeti gibi kriterler açık veride bulunmadığından dahil değildir.
- OSM verisinin eksiksizliği bölgeye göre değişir.
- Ücretsiz sunucuda GPU yoktur; büyük alanlarda YOLO modu birkaç dakika sürebilir.

## Proje yapısı
```
app.py                    Streamlit arayüzü
modules/
  config.py               Ayarlar (env ile değiştirilebilir)
  geo_utils.py            Karo matematiği, jeodezik alan
  imagery.py              Uydu karosu indirme → GeoTIFF
  vision_module.py        YOLO11-seg (ONNX/PT) + OSM yedek modu, vektörleştirme
  solar_module.py         PVGIS API, çatı yönelimi, GES/EV/CO₂ hesabı
  mcdm_module.py          OSMnx bağlam verisi, mesafeler, TOPSIS, Entropi, duyarlılık
  report_module.py        GeoJSON / CSV / Excel / PDF
  pipeline.py             Uçtan uca akış
training/                 Veri seti üretimi ve model eğitimi
models/                   Eğitilmiş model buraya
packages.txt              Streamlit Cloud sistem paketleri (PDF yazı tipi)
Dockerfile                İsteğe bağlı: kendi sunucunuzda Docker ile çalıştırma
```
