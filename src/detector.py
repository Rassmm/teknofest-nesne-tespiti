# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Nesne Tespit Modülü
# ==============================================================================
# YOLO best.pt modelini yükleyip çıkarım (inference) yapan modül.
# Farklı çözünürlüklerde (Full HD / 4K) ve formatlarda çalışabilir.
# ==============================================================================

import numpy as np
from ultralytics import YOLO

import config


class Detection:
    """Tek bir tespit sonucunu temsil eder."""

    __slots__ = [
        "class_id",
        "bbox",
        "confidence",
        "motion_status",
        "landing_status",
        "track_id",
    ]

    def __init__(self, class_id: int, bbox: tuple, confidence: float, track_id: int = -1):
        self.class_id = class_id          # Sınıf ID (0-3)
        self.bbox = bbox                  # (x1, y1, x2, y2) piksel koordinatları
        self.confidence = confidence      # Güven skoru [0, 1]
        self.motion_status = -1           # -1: geçersiz, 0: hareketsiz, 1: hareketli
        self.landing_status = -1          # -1: iniş alanı değil, 0: uygun değil, 1: uygun
        self.track_id = track_id          # Takip ID'si (-1: takip yok)

    def center(self) -> tuple:
        """Bounding box merkez noktasını döndürür."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        """Bounding box alanını döndürür."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict:
        """Sözlük formatına dönüştürür."""
        return {
            "class_id": self.class_id,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "motion_status": self.motion_status,
            "landing_status": self.landing_status,
            "track_id": self.track_id,
        }

    def __repr__(self):
        name = config.CLASS_NAMES.get(self.class_id, f"cls_{self.class_id}")
        return (
            f"Detection({name}, conf={self.confidence:.2f}, "
            f"bbox=[{self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f}], "
            f"motion={self.motion_status}, landing={self.landing_status})"
        )


class ObjectDetector:
    """
    YOLO tabanlı nesne tespit sınıfı.

    best.pt modelini yükler ve video karelerinde çıkarım yapar.
    Opsiyonel olarak ByteTrack ile nesne takibi de yapabilir.
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
        use_tracking: bool = True,
    ):
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or config.NMS_IOU_THRESHOLD
        self.use_tracking = use_tracking
        self.tracker_type = getattr(config, "TRACKER_TYPE", "botsort.yaml")

        # Model yükleme
        print(f"[Detector] Model yükleniyor: {self.model_path}")
        self.model = YOLO(self.model_path)
        print(f"[Detector] Model sınıfları: {self.model.names}")
        print(f"[Detector] Güven eşiği: {self.conf_threshold}, IoU eşiği: {self.iou_threshold}")

        # Sınıf eşleştirmesi
        self.class_mapping = config.MODEL_CLASS_MAPPING
        if not self.class_mapping:
            # 1:1 eşleştirme (model sınıf ID = yarışma sınıf ID)
            self.class_mapping = {i: i for i in self.model.names.keys()}

        # BBox boyut geçmişi (track_id → (x1, y1, x2, y2))
        self._bbox_history = {}
        
        # Ortalama araç genişliği takibi (kenardaki devasa kutuları kırpmak için)
        self._global_w_ema = None

    def _map_class_id(self, model_class_id: int) -> int:
        """Model sınıf ID'sini yarışma sınıf ID'sine dönüştürür."""
        return self.class_mapping.get(model_class_id, model_class_id)

    def detect(self, frame: np.ndarray) -> list:
        """
        Tek bir kare üzerinde nesne tespiti yapar.

        Args:
            frame: BGR formatında numpy dizisi (OpenCV).

        Returns:
            Detection nesnelerinin listesi.
        """
        frame_h, frame_w = frame.shape[:2]

        if self.use_tracking:
            results = self.model.track(
                frame,
                tracker=self.tracker_type,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=config.INPUT_SIZE,
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=config.INPUT_SIZE,
                verbose=False,
            )

        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Bounding box koordinatları
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()

                # Sınıf ID ve güven skoru
                model_cls = int(boxes.cls[i].item())
                mapped_cls = self._map_class_id(model_cls)
                conf = float(boxes.conf[i].item())

                # Takip ID'si
                track_id = -1
                if self.use_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                # Kenar bbox kırpma: frame sınırına yapıştır
                x1, y1, x2, y2 = self._clamp_bbox_to_frame(
                    x1, y1, x2, y2, frame_w, frame_h
                )

                # Araç genişliğini güncelle ve kenardaki aşırı genişlemeyi sınırla
                if mapped_cls in config.VEHICLE_CLASSES:
                    w = x2 - x1
                    h = y2 - y1
                    margin = config.BBOX_EDGE_MARGIN
                    is_at_edge = (
                        x1 < margin or y1 < margin or 
                        x2 > frame_w - margin or y2 > frame_h - margin
                    )
                    
                    if not is_at_edge and w > 0 and h > 0:
                        # Sadece tam görünen araçların genişliğini kaydet
                        if self._global_w_ema is None:
                            self._global_w_ema = w
                        else:
                            self._global_w_ema = 0.95 * self._global_w_ema + 0.05 * w
                            
                    elif is_at_edge and self._global_w_ema is not None:
                        # Kenardaki bir araç ise ve ortalama genişliği biliyorsak
                        max_w = self._global_w_ema * 1.6  # Ortalama genişliğin %60 fazlasına kadar izin ver
                        if w > max_w:
                            # Genişliği merkez etrafında daralt
                            cx = (x1 + x2) / 2
                            x1 = cx - max_w / 2
                            x2 = cx + max_w / 2

                # BBox boyut kararlılık kontrolü
                x1, y1, x2, y2 = self._stabilize_bbox_size(
                    track_id, x1, y1, x2, y2, frame_w, frame_h
                )

                det = Detection(
                    class_id=mapped_cls,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    track_id=track_id,
                )

                # Varsayılan hareket ve iniş durumları
                if mapped_cls in config.VEHICLE_CLASSES:
                    det.motion_status = 1  # Başlangıçta hareketli varsay (yeni giren araç)
                # İniş alanları için başlangıç değeri
                if mapped_cls in config.LANDING_ZONE_CLASSES:
                    det.landing_status = 0  # Başlangıçta uygun değil varsay

                detections.append(det)

        return detections

    @staticmethod
    def _clamp_bbox_to_frame(
        x1: float, y1: float, x2: float, y2: float,
        frame_w: int, frame_h: int,
    ) -> tuple:
        """
        BBox koordinatlarını frame sınırları içine kırpar.

        YOLO bazen frame dışına taşan koordinatlar üretir;
        bu fonksiyon bunları frame sınırına yapıştırır.

        Args:
            x1, y1, x2, y2: BBox koordinatları
            frame_w, frame_h: Frame boyutları

        Returns:
            Kırpılmış (x1, y1, x2, y2)
        """
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(frame_w), x2)
        y2 = min(float(frame_h), y2)
        return x1, y1, x2, y2

    def _stabilize_bbox_size(
        self, track_id: int,
        x1: float, y1: float, x2: float, y2: float,
        frame_w: int, frame_h: int,
    ) -> tuple:
        """
        Ardışık frame'ler arasında bbox boyutunun ani değişimini önler ve titremeyi (jitter) azaltır.

        Kenar bölgelerinde (nesne kameraya giriyor/çıkıyor) aşırı büyümeyi engeller.
        Normal bölgelerde ise IoU/YOLO gürültüsü kaynaklı genleşme/daralma (jitter)
        sorununu EMA (Exponential Moving Average) ile pürüzsüzleştirir.

        Args:
            track_id: Nesne takip ID
            x1, y1, x2, y2: Mevcut bbox
            frame_w, frame_h: Frame boyutları

        Returns:
            Kararlılaştırılmış (x1, y1, x2, y2)
        """
        if track_id < 0:
            return x1, y1, x2, y2

        if track_id not in self._bbox_history:
            self._bbox_history[track_id] = (x1, y1, x2, y2)
            return x1, y1, x2, y2

        prev_x1, prev_y1, prev_x2, prev_y2 = self._bbox_history[track_id]
        
        prev_w = prev_x2 - prev_x1
        prev_h = prev_y2 - prev_y1
        curr_w = x2 - x1
        curr_h = y2 - y1
        curr_cx = (x1 + x2) / 2
        curr_cy = (y1 + y2) / 2

        if prev_w <= 0 or prev_h <= 0 or curr_w <= 0 or curr_h <= 0:
            self._bbox_history[track_id] = (x1, y1, x2, y2)
            return x1, y1, x2, y2

        # 1. Kenar durumunda ani büyümeyi sınırla
        margin = config.BBOX_EDGE_MARGIN
        is_at_edge = (
            x1 < margin or
            y1 < margin or
            x2 > frame_w - margin or
            y2 > frame_h - margin
        )

        area_ratio = (curr_w * curr_h) / (prev_w * prev_h)

        if is_at_edge and area_ratio > config.BBOX_MAX_AREA_CHANGE_RATIO:
            # Aşırı büyümeyi yumuşat (Kenar için sert yumuşatma)
            alpha = config.BBOX_SMOOTHING_ALPHA
            curr_w = alpha * curr_w + (1 - alpha) * prev_w
            curr_h = alpha * curr_h + (1 - alpha) * prev_h
        else:
            # 2. Normal durumda (kenar değilse) jitter'ı engellemek için hafif EMA uygula
            # Boyutlarda (genişlik/yükseklik) yumuşatma yap, pozisyonu (merkezi) bozma
            # alpha yüksek olmalı ki hareket gecikmesi az olsun, ama jitter geçsin
            alpha_normal = 0.6
            curr_w = alpha_normal * curr_w + (1 - alpha_normal) * prev_w
            curr_h = alpha_normal * curr_h + (1 - alpha_normal) * prev_h

        # Yeni boyutlarla merkezi koruyarak x1, y1, x2, y2 hesapla
        x1 = curr_cx - curr_w / 2
        y1 = curr_cy - curr_h / 2
        x2 = curr_cx + curr_w / 2
        y2 = curr_cy + curr_h / 2

        # Frame sınırlarına tekrar kırp
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(frame_w), x2)
        y2 = min(float(frame_h), y2)

        # Geçmişi güncelle
        self._bbox_history[track_id] = (x1, y1, x2, y2)
        return x1, y1, x2, y2

    def get_model_info(self) -> dict:
        """Model bilgilerini döndürür."""
        return {
            "model_path": self.model_path,
            "class_names": self.model.names,
            "class_mapping": self.class_mapping,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "use_tracking": self.use_tracking,
        }
