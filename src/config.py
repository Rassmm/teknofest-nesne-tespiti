# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Merkezi Konfigürasyon
# ==============================================================================

import os

# ------------------------------------------------------------------------------
# Model Ayarları
# ------------------------------------------------------------------------------
# Proje kök dizini (src/ klasörünün bir üstü)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "best.pt")
CONFIDENCE_THRESHOLD = 0.15
NMS_IOU_THRESHOLD = 0.45
INPUT_SIZE = 1280  # YOLO giriş boyutu (piksel) - İnsanları görmek için artırıldı
HALF_PRECISION = True  # FP16 kullan (GPU varsa)

# ------------------------------------------------------------------------------
# Sınıf Tanımları (Yarışma Formatı)
# ------------------------------------------------------------------------------
VEHICLE_CLASS_ID = 0      # Taşıt
HUMAN_CLASS_ID = 1        # İnsan
UAP_CLASS_ID = 2          # Uçan Araba Park Alanı
UAI_CLASS_ID = 3          # Uçan Ambulans İniş Alanı

CLASS_NAMES = {
    VEHICLE_CLASS_ID: "Tasit",
    HUMAN_CLASS_ID: "Insan",
    UAP_CLASS_ID: "UAP",
    UAI_CLASS_ID: "UAI",
}

LANDING_ZONE_CLASSES = {UAP_CLASS_ID, UAI_CLASS_ID}
VEHICLE_CLASSES = {VEHICLE_CLASS_ID}

# Model sınıf eşleştirmesi
# Model çıktısı: {0: 'UAI', 1: 'UAP', 2: 'car', 3: 'pedestrian'}
# Yarışma formatı: {0: 'Tasit', 1: 'Insan', 2: 'UAP', 3: 'UAI'}
# Anahtar: modelin sınıf ID'si → Değer: yarışma sınıf ID'si
MODEL_CLASS_MAPPING = {
    0: 3,   # Model 'UAI' (0) → Yarışma UAİ (3)
    1: 2,   # Model 'UAP' (1) → Yarışma UAP (2)
    2: 0,   # Model 'car' (2) → Yarışma Taşıt (0)
    3: 1,   # Model 'pedestrian' (3) → Yarışma İnsan (1)
}

# ------------------------------------------------------------------------------
# Hareket Algılama Ayarları
# ------------------------------------------------------------------------------
# Kullanılacak takip algoritması ("botsort.yaml" veya "bytetrack.yaml")
# İHA (drone) çekimlerinde kamera hareketini kompanze ettiği için BoT-SORT önerilir.
TRACKER_TYPE = "botsort.yaml"

# Kamera kompanzasyonu için ORB öznitelik sayısı
ORB_FEATURES = 1000
# Homography RANSAC eşiği (piksel)
RANSAC_THRESHOLD = 5.0
# Hareket eşiği - kamera kompanzasyonu sonrası kalan hareket (piksel)
MOTION_PIXEL_THRESHOLD = 1.5
# Hysteresis eşikleri — flip-flop önlemek için çift eşik sistemi
# Hareketli→Hareketsiz geçişi için DÜŞÜK eşik (bu eşiğin altına düşmeli)
MOTION_HYSTERESIS_LOW = 1.5
# Hareketsiz→Hareketli geçişi için YÜKSEK eşik (bu eşiği geçmeli)
# Homography hatası tipik 1-3px olduğundan, gerçek hareket 4+ px olmalı
MOTION_HYSTERESIS_HIGH = 3.0
# EMA (Exponential Moving Average) yumuşatma katsayısı (0-1, düşük = daha pürüzsüz)
MOTION_EMA_ALPHA = 0.15
# Hareket karar vermek için gereken minimum kare sayısı
MOTION_MIN_FRAMES = 10
# Ağırlıklı oylama eşiği (hareketli kararı için gerekli oran)
MOTION_VOTE_THRESHOLD = 0.65
# Hareket takibi için IoU eşiği (araç eşleştirme)
TRACKING_IOU_THRESHOLD = 0.3

# ------------------------------------------------------------------------------
# BBox Kararlılık Ayarları
# ------------------------------------------------------------------------------
# Kenar bölgesi tespiti için margin (piksel) — bu margin içindeki bbox'lar kenar objesi
BBOX_EDGE_MARGIN = 15
# İzin verilen maksimum bbox alan değişim oranı (ardışık frame'ler arası)
BBOX_MAX_AREA_CHANGE_RATIO = 1.25
# BBox boyut yumuşatma katsayısı (0-1, düşük = daha pürüzsüz geçiş)
BBOX_SMOOTHING_ALPHA = 0.35

# Optik akış parametreleri (Lucas-Kanade)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(3, 30, 0.01),  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
)

# ------------------------------------------------------------------------------
# İniş Uygunluğu Değerlendirme Ayarları
# ------------------------------------------------------------------------------
# Alan üzerinde cisim kontrolü için IoU eşiği
LANDING_OVERLAP_IOU_THRESHOLD = 0.01
# Kenar kontrolü - alan kenardan bu kadar piksel içindeyse "kısmen dışarıda"
LANDING_EDGE_MARGIN = 5

# ------------------------------------------------------------------------------
# Sunucu İletişim Ayarları
# ------------------------------------------------------------------------------
SERVER_URL = "http://localhost:8000"
API_GET_FRAME = "/api/get_frame"
API_SEND_RESULT = "/api/send_result"
API_KEY = ""
REQUEST_TIMEOUT = 30  # saniye

# ------------------------------------------------------------------------------
# Video / Test Ayarları
# ------------------------------------------------------------------------------
VIDEO_FPS = 7.5
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")

# ------------------------------------------------------------------------------
# SAHI (Slicing Aided Hyper Inference) Ayarları
# ------------------------------------------------------------------------------
# Küçük nesnelerin (özellikle insanlar) tespitini iyileştirmek için
# frame'i dilimlere bölüp ayrı ayrı inference yapar ve sonuçları birleştirir.
SAHI_ENABLED = True                # SAHI pipeline'ı aktif/pasif
SAHI_SLICE_HEIGHT = 640            # Dilim yüksekliği (piksel)
SAHI_SLICE_WIDTH = 640             # Dilim genişliği (piksel)
SAHI_OVERLAP_RATIO = 0.35          # Dilimler arası örtüşme oranı artırıldı
SAHI_MERGE_IOU_THRESHOLD = 0.35    # NMS birleştirme IoU eşiği (daha agresif temizlik için biraz düşürüldü)
SAHI_SLICE_CONF_THRESHOLD = 0.20   # Yanlış pozitifleri azaltmak için güven eşiği biraz artırıldı
SAHI_FULL_FRAME_ENABLED = True     # Tam frame inference da yapılsın mı (hibrit mod)
SAHI_SLICE_INPUT_SIZE = 1280       # Yeni model 1280'e göre eğitildiği için, dilimler bu boyuta upscale edilip gönderilir

# ------------------------------------------------------------------------------
# Görselleştirme Renkleri (BGR)
# ------------------------------------------------------------------------------
VIS_COLORS = {
    VEHICLE_CLASS_ID: (0, 200, 255),   # Turuncu
    HUMAN_CLASS_ID: (0, 255, 0),       # Yeşil
    UAP_CLASS_ID: (255, 150, 0),       # Mavi
    UAI_CLASS_ID: (0, 0, 255),         # Kırmızı
}
