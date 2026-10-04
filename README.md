# 🛩️ Teknofest Havacılıkta Yapay Zekâ - Görev 1: Nesne Tespiti

> **Drone görüntülerinden gerçek zamanlı nesne tespiti, kamera hareket kompanzasyonlu hareket algılama ve iniş alanı uygunluk değerlendirmesi yapan otonom yapay zekâ sistemi.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8.x-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/Lisans-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📋 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
- [✨ Özellikler](#-özellikler)
- [🏗️ Mimari](#️-mimari)
  - [Pipeline Akış Şeması](#pipeline-akış-şeması)
  - [Modül ve Dizin Yapısı](#modül-ve-dizin-yapısı)
- [🚀 Kurulum](#-kurulum)
  - [Gereksinimler](#gereksinimler)
  - [Adım Adım Kurulum](#adım-adım-kurulum)
  - [Model Ağırlıklarının İndirilmesi](#model-ağırlıklarının-indirilmesi)
- [💻 Kullanım](#-kullanım)
  - [1. Yarışma Modu (Canlı Sunucu Entegrasyonu)](#1-yarışma-modu-canlı-sunucu-entegrasyonu)
  - [2. Lokal Test Modu (Video Doğrulama)](#2-lokal-test-modu-video-doğrulama)
  - [3. Google Colab Modu](#3-google-colab-modu)
- [⚙️ Konfigürasyon](#️-konfigürasyon)
- [📊 Pipeline Detayları](#-pipeline-detayları)
  - [1. Nesne Tespiti (Hybrid YOLOv8 + SAHI)](#1-nesne-tespiti-hybrid-yolov8--sahi)
  - [2. Kamera Kompanzasyonlu Hareket Algılama](#2-kamera-kompanzasyonlu-hareket-algılama)
  - [3. İniş Alanı Uygunluk Değerlendirmesi](#3-iniş-alanı-uygunluk-değerlendirmesi)
  - [4. BBox ve Zamansal Kararlılık](#4-bbox-ve-zamansal-kararlılık)
- [📁 Çıktı Formatı](#-çıktı-formatı)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)
- [📄 Lisans](#-lisans)
- [📞 İletişim](#-iletişim)

---

## 🎯 Proje Hakkında

Bu proje, **Teknofest Havacılıkta Yapay Zekâ Yarışması (Görev #1: Nesne Tespiti)** kapsamında insansız hava araçları (İHA) tarafından kaydedilen hava görüntülerinden gerçek zamanlı olarak nesneleri tespit etmek, durumlarını analiz etmek ve yarışma değerlendirme sunucusuna standart JSON formatında aktarmak üzere tasarlanmıştır.

Sistem, İHA kamerasından gelen görüntülerde aşağıdaki **4 ana nesne sınıfını** ve bunlara ait dinamik durumları eşzamanlı olarak tespit eder:

| Sınıf ID | Sınıf Adı | Açıklama | Değerlendirilen Durum |
| :---: | :--- | :--- | :--- |
| **0** | **Taşıt (Vehicle)** | Karayolu ve saha taşıtları | **Hareket Algılama:** Hareketli (`1`) veya Hareketsiz (`0`) |
| **1** | **İnsan (Human)** | Yayalar ve sahada bulunan kişiler | **Konum & Varlık Tespiti:** SAHI ile yüksek doğruluk |
| **2** | **Uçan Araba Park Alanı (UAP)** | Şehir içi hava taşımacılığı park alanları | **İniş Uygunluğu:** Uygun (`1`) veya Uygun Değil (`0`) |
| **3** | **Uçan Ambulans İniş Alanı (UAİ)** | Acil tıbbi iniş ve kurtarma bölgeleri | **İniş Uygunluğu:** Uygun (`1`) veya Uygun Değil (`0`) |

### Çoklu Sensör ve Yüksek Çözünürlük Desteği
- **RGB Kamera:** Gündüz koşullarında 1080p (Full HD) ve 4K (Ultra HD) görüntülerde yüksek keskinlikte tespit.
- **Termal Kamera (LWIR):** Gece, sis veya düşük kontrastlı ortamlarda termal parlaklık dağılımına uyarlanmış tespit ve takip yeteneği.

---

## ✨ Özellikler

- 🔍 **YOLOv8 Tabanlı Gerçek Zamanlı Nesne Tespiti:** Yüksek çıkarım hızı ve mAP başarımına sahip modern evrişimli mimari.
- 🧩 **SAHI (Slicing Aided Hyper Inference):** Yüksek irtifadan çekilen İHA görüntülerinde mikro boyutlu hedefleri (özellikle insanları) yakalamak için örtüşen pencereli (sliding window) dilimleme.
- 🎯 **BoT-SORT Çoklu Nesne Takibi:** Kamera hareket kompanzasyonu (GMC) destekli, yüksek kare atlamalarında bile nesne kimliğini (track ID) koruyan takip algoritması.
- 🔄 **Kamera Kompanzasyonlu Hareket Algılama:** Drone hareketinden kaynaklanan görüntü kaymasını ORB öznitelikleri ve RANSAC Homography ile nötrleyerek araçların gerçek hareketini hesaplama.
- 📈 **Hysteresis Çift Eşik & EMA Yumuşatma:** Araç hızındaki gürültüleri filtreleyerek hareket durumu dalgalanmalarını (flip-flop) ortadan kaldıran dinamik eşikleme.
- 🚁 **İniş Alanı Uygunluk Değerlendirmesi:** UAP ve UAİ pistlerinin kamera kadrajı içinde eksiksiz yer alması (tam görünürlük) ve pist üzerinde insan/taşıt engeli bulunmaması durumunu milisaniyeler içinde doğrulama.
- 📐 **BBox Boyut ve Kenar Stabilizasyonu:** Kadraj kenarlarında oluşan kutu genişleme anomalilerini önleyen ve ardışık karelerde kutu boyutlarını pürüzsüzleştiren adaptif filtreleme.
- 👥 **İnsan Tespiti Zamansal Güven Filtreleme:** Çim ve gölge kaynaklı 1-2 karelik anlık yanlış pozitifleri (FP) temizleyen geçmiş tabanlı güven yumuşatması.
- 🌐 **Yarışma Sunucusu HTTP İstemcisi:** REST API üzerinden görüntü alımı, JSON yanıt paketi oluşturma ve hata durumlarında güvenli yeniden bağlanma desteği.
- 🎬 **Lokal Test ve Analiz Araçları:** Yarışma simülasyonu sağlayan görselleştirilmiş video kaydedici, canlı önizleme ve detaylı performans metrikleri.

---

## 🏗️ Mimari

### Pipeline Akış Şeması

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                 Kamera / Video / HTTP Girdisi                │
  │                  (RGB & Termal, Full HD / 4K)               │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Nesne Tespiti (Hybrid YOLOv8 + SAHI)             │
  │       • 1280px Tam Kare Tespit + BoT-SORT Çoklu Takip       │
  │       • 640x640 Örtüşmeli Dilimler (SAHI Slicing)           │
  │       • NMS Merge & Güven / BBox Kararlılık Filtresi        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │         Kamera Kompanzasyonlu Hareket Algılama              │
  │       • Arka Plan ORB Öznitelik Eşleme                      │
  │       • RANSAC Homography (İHA Hareketi Tahmini)            │
  │       • Araç Kalıntı Ötelenme (Residual Displacement)        │
  │       • EMA Yumuşatma & Hysteresis Çift Eşik Kararı         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             İniş Alanı Uygunluk Değerlendirmesi             │
  │       • UAP & UAİ Sınır Kontrolü (Kadraj İçi / Dışı)        │
  │       • Engel / Taşıt / İnsan Çakışma Analizi (IoU)         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    Sonuç Formatlama                         │
  │       • Yarışma Standartlarında JSON Objesinin Üretimi       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌─────────────────────────────┐ ┌─────────────────────────────┐
  │  Yarışma HTTP REST Sunucusu │ │    Lokal Çıktı & MP4 Video  │
  │   (/api/send_result İletimi)│ │   (Annotated Video & JSON)  │
  └─────────────────────────────┘ └─────────────────────────────┘
```

### Modül ve Dizin Yapısı

```
teknofest-nesne-tespiti/
├── src/
│   ├── config.py           # Merkezi konfigürasyon ve eşik parametreleri
│   ├── detector.py         # YOLO nesne tespit ve temel çıkarım modülü  
│   ├── sahi_detector.py    # SAHI dilimli inference ve hibrit birleştirici
│   ├── motion_detector.py  # Homography tabanlı hareket algılama modülü
│   ├── landing_evaluator.py# UAP/UAİ iniş uygunluğu değerlendirici
│   ├── formatter.py        # Yarışma formatına uygun JSON veri dönüştürücü
│   └── pipeline.py         # Yarışma sunucusu entegrasyonu ve ana orchestration
├── scripts/
│   ├── test_local.py       # Lokal video test, görselleştirme ve raporlama aracı
│   └── extract_frames.py   # Videolardan kare çıkarma ve veri seti hazırlama aracı
├── docs/
│   ├── yarisma_sartlari.md # Yarışma kuralları, puanlama ve isterler rehberi
│   └── termal_goruntuler.md# Termal görüntü çözümleri ve optimizasyon kılavuzu
├── assets/                 # Model ağırlıkları (Git LFS) ve test referansları
├── requirements.txt        # Python bağımlılıkları
├── setup.py                # Paket kurulum konfigürasyonu
├── .gitignore              # Git yok sayma kuralları
├── .gitattributes          # Git LFS model dosyası tanımlamaları
├── LICENSE                 # MIT Açık Kaynak Lisansı
└── README.md               # Proje dokümantasyonu
```

---

## 🚀 Kurulum

### Gereksinimler

- **İşletim Sistemi:** Windows 10/11, Ubuntu 20.04/22.04 LTS veya macOS
- **Python:** 3.10 veya üzeri
- **Donanım:** CUDA 11.8+ destekli NVIDIA GPU (Önerilen: 6 GB+ VRAM) veya CPU

### Adım Adım Kurulum

1. **Depoyu Klonlayın:**
   ```bash
   git clone https://github.com/kullanici-adi/teknofest-nesne-tespiti.git
   cd teknofest-nesne-tespiti
   ```

2. **Sanal Ortam Oluşturun ve Aktif Edin:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Gerekli Paketleri Yükleyin:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Model Ağırlıklarını İndirin (`best.pt`):**
   Model ağırlıkları yüksek boyutlu olduğu için Git LFS üzerinden yönetilmektedir:
   ```bash
   # Git LFS ile depoyu çektiyseniz:
   git lfs pull

   # veya model ağırlık dosyasını doğrudan assets/ dizinine kopyalayın:
   # assets/best.pt veya kök dizinde best.pt
   ```

---

## 💻 Kullanım

Sistem 3 temel çalışma moduna sahiptir:

### 1. Yarışma Modu (Canlı Sunucu Entegrasyonu)

Yarışma sunucusuna bağlanarak kareleri HTTP GET ile çeker, çıkarım yapar ve sonuçları HTTP POST ile anlık iletir:

```bash
# Varsayılan konfigürasyon ile çalıştırma
python src/pipeline.py --server http://localhost:8000

# Özel API anahtarı ile çalıştırma
python src/pipeline.py --server http://192.168.1.100:8000 --api-key YARISMA_GIZLI_ANAHTAR
```

### 2. Lokal Test Modu (Video Doğrulama)

Kayıtlı videolar üzerinde tam pipeline testleri gerçekleştirir, tespit edilen nesneleri, hareket durumlarını ve iniş alanlarını renk kodlu kutularla işaretleyerek video ve JSON kaydeder:

```bash
# Temel video testi (sonuçlar varsayılan olarak results/ klasörüne yazılır)
python scripts/test_local.py --video test_video.mp4 --output results/

# Canlı OpenCV penceresinde önizleme yaparak test etme
python scripts/test_local.py --video test_video.mp4 --show

# SAHI dilimlemeyi devre dışı bırakarak sadece tam kare (hızlı) test etme
python scripts/test_local.py --video test_video.mp4 --no-sahi

# İlk 300 kareyi hızlıca analiz etme
python scripts/test_local.py --video test_video.mp4 --max-frames 300
```

### 3. Google Colab Modu

GPU gücünden bulutta faydalanmak için `Teknofest_Colab.ipynb` not defteri hazır olarak sunulmuştur:

1. Depoyu Google Drive'ınıza veya Colab ortamına aktarın.
2. `Teknofest_Colab.ipynb` dosyasını Google Colab'de açın.
3. Çalışma zamanı türünü **T4 GPU** olarak seçin ve hücreleri sırasıyla çalıştırın.

---

## ⚙️ Konfigürasyon

Merkezi konfigürasyon `src/config.py` içerisinden yönetilir. Önemli parametreler ve varsayılan değerleri:

| Parametre | Varsayılan | Açıklama |
| :--- | :---: | :--- |
| `MODEL_PATH` | `best.pt` | Eğitilmiş YOLOv8 ağırlık dosyasının dosya yolu |
| `CONFIDENCE_THRESHOLD` | `0.15` | Tam kare nesne tespit genel güven eşiği |
| `INPUT_SIZE` | `1280` | Model giriş çözünürlüğü (piksel) |
| `HALF_PRECISION` | `True` | GPU hızlandırması için FP16 yarı hassasiyet |
| `SAHI_ENABLED` | `True` | SAHI dilimli çıkarım mekanizmasının aktifliği |
| `SAHI_SLICE_WIDTH` | `640` | SAHI pencere genişliği (piksel) |
| `SAHI_SLICE_HEIGHT` | `640` | SAHI pencere yüksekliği (piksel) |
| `SAHI_OVERLAP_RATIO` | `0.35` | Dilimler arasındaki örtüşme oranı (%35) |
| `SAHI_MERGE_IOU_THRESHOLD` | `0.35` | Dilim sonuçları ile tam kare sonuçlarını birleştirme IoU eşiği |
| `TRACKER_TYPE` | `botsort.yaml` | Çoklu nesne takipçisi (GMC destekli BoT-SORT) |
| `ORB_FEATURES` | `1000` | Kamera hareketi tahmini için çıkarılacak köşe sayısı |
| `RANSAC_THRESHOLD` | `5.0` | Kamera homography matrisi tahmini için RANSAC hata eşiği |
| `MOTION_HYSTERESIS_LOW` | `1.5` | Hareketli → Hareketsiz durumuna geçiş alt sınır piksel eşiği |
| `MOTION_HYSTERESIS_HIGH` | `3.0` | Hareketsiz → Hareketli durumuna geçiş üst sınır piksel eşiği |
| `MOTION_EMA_ALPHA` | `0.15` | Ötelenme miktarı zamansal yumuşatma katsayısı (EMA) |
| `MOTION_MIN_FRAMES` | `10` | Kesin hareket kararı vermek için gereken minimum gözlem sayısı |
| `LANDING_OVERLAP_IOU_THRESHOLD` | `0.01` | İniş alanı üzerinde engel kontrolü için maksimum izin verilen çakışma |
| `LANDING_EDGE_MARGIN` | `5` | Pistin kadraj sınırına yaklaşma payı (kısmen dışarıda kontrolü) |

---

## 📊 Pipeline Detayları

### 1. Nesne Tespiti (Hybrid YOLOv8 + SAHI)
- **1280px Giriş Çözünürlüğü:** Model, hava araçlarından çekilen küçük nesnelerin kaybolmaması için 1280 piksel giriş ölçeğiyle çalıştırılır.
- **640x640 SAHI Dilimleme:** Görüntü %35 örtüşmeyle 640x640 pencerelere bölünür. Her pencere bağımsız olarak çıkarıma tabi tutulur.
- **Hibrit NMS Birleştirme:** Tam kare çıkarımından elde edilen genel bağlam ve BoT-SORT takip kimlikleri (track ID), SAHI dilimlerinden gelen yüksek hassasiyetli küçük nesne koordinatlarıyla akıllı IoU birleştirme algoritması sayesinde harmanlanır.

### 2. Kamera Kompanzasyonlu Hareket Algılama
İHA kamerası uçuş sırasında sürekli titrediği veya ötelendiği için sahnedeki tüm nesneler optik olarak hareket ediyor gibi algılanır. Gerçek araç hareketini saptamak için şu adımlar uygulanır:
1. **Taşıt Maskeleme:** Bilinen araç bölgeleri maskelenerek sadece statik arka plandan (zemin, binalar, yollar) ORB öznitelik noktaları toplanır.
2. **RANSAC Homography:** Ardışık kareler arasındaki zemin öznitelikleri eşleştirilip homografi matrisi $H$ türetilir. Böylece kameranın bağıl hareketi modellenir.
3. **Kalıntı Ötelenme (Residual Displacement):** Takip edilen aracın önceki merkez noktası homografi ile yeni kareye izdüşürülür. İzdüşüm noktası ile aracın gerçek yeni merkezi arasındaki fark, kameradan bağımsız salt araç hareketini verir:
   $$\Delta_{residual} = \| c_t - H \cdot c_{t-1} \|$$
4. **EMA & Hysteresis Çift Eşik:** Ham hareket değerleri EMA filtresi ile yumuşatılır. 1.5 px ve 3.0 px çift eşikli Hysteresis mantığı ile titremelerden kaynaklanan durum sıçramaları (flip-flop) engellenir.
5. **Ağırlıklı Zamansal Oylama:** Son 10 karelik pencere içinde yakın zamana yüksek ağırlık verilerek hareket kararı (`0` veya `1`) oylanır.

### 3. İniş Alanı Uygunluk Değerlendirmesi
UAP (Sınıf 2) ve UAİ (Sınıf 3) alanları için iniş güvenliği şu 2 aşamalı protokolle denetlenir:
1. **Kadraj Sınır Kontrolü (Full Visibility):** İniş alanının 4 köşesi de görüntü kenarlarından `LANDING_EDGE_MARGIN` kadar içeride olmalıdır. Kısmen kadraj dışına taşan alanlar tespit edilir ancak inişe **uygun değil (`0`)** olarak işaretlenir.
2. **Engel / Çakışma Kontrolü (Obstruction Check):** Pist alanı sınırlayıcı kutusu ile sahnedeki taşıt veya insan kutuları arasında IoU ve kesişim alanı oranı hesaplanır. Herhangi bir nesne ile çakışma varsa pist **uygun değil (`0`)**, tamamen temizse **uygun (`1`)** kararı verilir.

### 4. BBox ve Zamansal Kararlılık
- **Kenar Anomalisi Kontrolü:** Kadrajın sınırlarına giren veya çıkan araçların tespit kutularının orantısız büyümesi, global EMA araç genişliği referans alınarak sınırlandırılır.
- **Zamansal İnsan Güven Yumuşatması:** Yayalar için güven skoru zamansal EMA ile takip edilir; tek karelik çim/leke parlamaları elenerek sadece ardışık karelerde tutarlı olan tespitler aktarılır.

---

## 📁 Çıktı Formatı

Sistem, yarışma sunucusuna ve yerel JSON raporlarına her kare için aşağıdaki standart JSON şemasını üretir:

```json
{
  "frame_index": 0,
  "detections": [
    {
      "class_id": 0,
      "cls": 0,
      "bbox": [124.5, 450.2, 210.8, 560.1],
      "confidence": 0.9524,
      "motion_status": 1,
      "landing_status": -1
    },
    {
      "class_id": 1,
      "cls": 1,
      "bbox": [650.0, 320.4, 672.3, 380.1],
      "confidence": 0.8812,
      "motion_status": -1,
      "landing_status": -1
    },
    {
      "class_id": 2,
      "cls": 2,
      "bbox": [800.0, 200.0, 1050.0, 450.0],
      "confidence": 0.9145,
      "motion_status": -1,
      "landing_status": 1
    }
  ]
}
```

### Durum Kodları Referansı
- **`motion_status`:** `1` (Hareketli), `0` (Hareketsiz), `-1` (Uygulanamaz / Araç dışı sınıflar)
- **`landing_status`:** `1` (İnişe Uygun), `0` (İnişe Uygun Değil), `-1` (Uygulanamaz / İniş alanı dışı sınıflar)

---

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak isterseniz lütfen aşağıdaki adımları takip ediniz:

1. Bu depoyu çatallayın (Fork).
2. Yeni bir özellik dalı oluşturun:
   ```bash
   git checkout -b ozellik/yeni-ozellik
   ```
3. Değişikliklerinizi kaydedin:
   ```bash
   git commit -m "feat: Yeni özellik eklendi"
   ```
4. Dalınıza gönderin:
   ```bash
   git push origin ozellik/yeni-ozellik
   ```
5. Bir **Pull Request (PR)** açın.

---

## 📄 Lisans

Bu proje **MIT Lisansı** kapsamında lisanslanmıştır. Detaylı bilgi için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

---

## 📞 İletişim

Sorularınız, iş birliği ve hata bildirimleri için:

- **Takım:** Teknofest Havacılıkta Yapay Zekâ Ekibi
- **E-posta:** iletisim@example.com
- **Proje Bağlantısı:** [GitHub Repository](https://github.com/kullanici-adi/teknofest-nesne-tespiti)
