# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - SAHI (Slicing Aided Hyper Inference)
# ==============================================================================
# Drone görüntülerinde küçük nesnelerin (özellikle insanlar) tespitini
# iyileştirmek için frame'i örtüşen dilimlere bölüp ayrı ayrı inference yapar.
#
# Pipeline:
#   1. (Opsiyonel) Tam frame inference — büyük nesneler + tracking
#   2. Frame'i 256×256 dilimlere böl (overlap ile)
#   3. Her dilim üzerinde YOLO inference (tracking kapalı)
#   4. Dilim koordinatlarını orijinal frame'e dönüştür (stitching)
#   5. Tam frame + dilim sonuçlarını NMS ile birleştir (merge)
# ==============================================================================

import numpy as np
from ultralytics import YOLO

import config
from detector import Detection


class SAHIDetector:
    """
    SAHI tabanlı nesne tespit sınıfı.

    Mevcut YOLO modelini kullanarak frame'i dilimlere bölüp
    her dilim üzerinde ayrı inference yapar, sonuçları stitching + NMS
    ile birleştirir.

    Hibrit mod: Tam frame (tracking ile) + dilimli inference (tracking kapalı)
    → NMS ile birleştirme. Tracking bilgisi tam frame'den gelir.
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

        # SAHI parametreleri
        self.slice_h = config.SAHI_SLICE_HEIGHT
        self.slice_w = config.SAHI_SLICE_WIDTH
        self.overlap_ratio = config.SAHI_OVERLAP_RATIO
        self.merge_iou = config.SAHI_MERGE_IOU_THRESHOLD
        self.slice_conf = config.SAHI_SLICE_CONF_THRESHOLD
        self.full_frame_enabled = config.SAHI_FULL_FRAME_ENABLED
        self.slice_input_size = config.SAHI_SLICE_INPUT_SIZE

        # Model yükleme
        # ÖNEMLİ: Tam frame tracking ve dilimli inference için AYRI model nesneleri
        # kullanıyoruz. BoT-SORT tracker'ın GMC (Global Motion Compensation) modülü
        # dahili olarak optik akış piramitleri tutar. Aynı model üzerinde farklı
        # boyutlarda (1920x1080 vs 256x256) inference yapılırsa piramit boyutları
        # uyumsuz olur ve "prevPyr.size() == nextPyr.size()" hatası verir.
        print(f"[SAHI] Model yükleniyor: {self.model_path}")
        self.model = YOLO(self.model_path)           # Tam frame + tracking
        self.slice_model = YOLO(self.model_path)     # Dilimler için (tracking yok)
        print(f"[SAHI] Model sınıfları: {self.model.names}")
        print(f"[SAHI] Dilim boyutu: {self.slice_w}×{self.slice_h}, "
              f"Overlap: {self.overlap_ratio}, Merge IoU: {self.merge_iou}")

        # Sınıf eşleştirmesi
        self.class_mapping = config.MODEL_CLASS_MAPPING
        if not self.class_mapping:
            self.class_mapping = {i: i for i in self.model.names.keys()}

        # BBox boyut geçmişi (track_id → (x1, y1, x2, y2))
        self._bbox_history = {}

        # Ortalama araç genişliği takibi
        self._global_w_ema = None

        # İstatistikler
        self._last_slice_count = 0
        self._last_full_det_count = 0
        self._last_slice_det_count = 0
        self._last_merged_det_count = 0
        
        # İnsan tespitleri için temporal confidence (güven) stabilizasyonu
        self._human_conf_history = {}
        self._human_id_counter = 0

    def _map_class_id(self, model_class_id: int) -> int:
        """Model sınıf ID'sini yarışma sınıf ID'sine dönüştürür."""
        return self.class_mapping.get(model_class_id, model_class_id)

    # ──────────────────────────────────────────────────────────────────────
    # 1. Frame Dilimleme (Slicing)
    # ──────────────────────────────────────────────────────────────────────

    def _generate_slices(self, frame_h: int, frame_w: int) -> list:
        """
        Frame'i örtüşen dilimlere böler ve her dilimin koordinatlarını döndürür.

        Args:
            frame_h: Frame yüksekliği
            frame_w: Frame genişliği

        Returns:
            [(x_start, y_start, x_end, y_end), ...] dilim koordinatları listesi
        """
        slices = []
        step_h = int(self.slice_h * (1 - self.overlap_ratio))
        step_w = int(self.slice_w * (1 - self.overlap_ratio))

        # Minimum 1 piksel adım (sonsuz döngü önleme)
        step_h = max(1, step_h)
        step_w = max(1, step_w)

        y = 0
        while y < frame_h:
            y_end = min(y + self.slice_h, frame_h)

            x = 0
            while x < frame_w:
                x_end = min(x + self.slice_w, frame_w)
                slices.append((x, y, x_end, y_end))

                if x_end >= frame_w:
                    break
                x += step_w

            if y_end >= frame_h:
                break
            y += step_h

        return slices

    # ──────────────────────────────────────────────────────────────────────
    # 2. Tam Frame Inference (Tracking ile)
    # ──────────────────────────────────────────────────────────────────────

    def _detect_full_frame(self, frame: np.ndarray) -> list:
        """
        Tam frame üzerinde YOLO inference (opsiyonel tracking ile).

        Büyük nesneler (araçlar, UAP, UAI) ve tracking bilgisi buradan gelir.
        """
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

        return self._parse_results(results, frame.shape, offset_x=0, offset_y=0,
                                   use_tracking=self.use_tracking)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Dilim Inference (Tracking kapalı)
    # ──────────────────────────────────────────────────────────────────────

    def _detect_slices(self, frame: np.ndarray) -> list:
        """
        Frame'i dilimlere bölüp her dilim üzerinde ayrı inference yapar.

        Tracking kapalıdır çünkü dilimli inference'da track ID'leri tutarsız olur.
        Koordinatlar orijinal frame'e dönüştürülür (stitching).
        """
        frame_h, frame_w = frame.shape[:2]
        slices = self._generate_slices(frame_h, frame_w)
        self._last_slice_count = len(slices)

        all_slice_detections = []

        for (sx, sy, ex, ey) in slices:
            # Dilimi kes
            slice_img = frame[sy:ey, sx:ex]

            # Çok küçük dilimleri atla
            if slice_img.shape[0] < 32 or slice_img.shape[1] < 32:
                continue

            # Dilim üzerinde inference (ayrı model, tracking kapalı, düşük eşik)
            results = self.slice_model(
                slice_img,
                conf=self.slice_conf,
                iou=self.iou_threshold,
                imgsz=self.slice_input_size,
                verbose=False,
            )

            # Sonuçları parse et ve koordinatları orijinal frame'e dönüştür
            slice_dets = self._parse_results(
                results, slice_img.shape,
                offset_x=sx, offset_y=sy,
                use_tracking=False,
            )

            # Tüm sınıfları SAHI'den al (artık sadece insan değil, küçük araçları ve alanları da destekliyoruz)
            all_slice_detections.extend(slice_dets)

        return all_slice_detections

    # ──────────────────────────────────────────────────────────────────────
    # 4. Sonuç Parse Etme + Stitching
    # ──────────────────────────────────────────────────────────────────────

    def _parse_results(
        self,
        results,
        img_shape: tuple,
        offset_x: int = 0,
        offset_y: int = 0,
        use_tracking: bool = False,
    ) -> list:
        """
        YOLO sonuçlarını parse eder ve koordinatları orijinal frame'e dönüştürür.

        Args:
            results: YOLO inference sonuçları
            img_shape: Inference yapılan görüntünün boyutu (h, w, c)
            offset_x: Dilimin orijinal frame'deki X başlangıcı
            offset_y: Dilimin orijinal frame'deki Y başlangıcı
            use_tracking: Tracking bilgisi var mı

        Returns:
            Detection nesnelerinin listesi (koordinatlar orijinal frame'de)
        """
        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Bounding box koordinatları (dilim koordinatlarında)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()

                # Stitching: dilim koordinatlarını orijinal frame'e dönüştür
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                # Sınıf ID ve güven skoru
                model_cls = int(boxes.cls[i].item())
                mapped_cls = self._map_class_id(model_cls)
                conf = float(boxes.conf[i].item())

                # Takip ID'si
                track_id = -1
                if use_tracking and boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                det = Detection(
                    class_id=mapped_cls,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    track_id=track_id,
                )

                # Varsayılan durumlar
                if mapped_cls in config.VEHICLE_CLASSES:
                    det.motion_status = 1
                if mapped_cls in config.LANDING_ZONE_CLASSES:
                    det.landing_status = 0

                detections.append(det)

        return detections

    # ──────────────────────────────────────────────────────────────────────
    # 5. NMS ile Birleştirme (Merge)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        """İki bounding box arasındaki IoU değerini hesaplar."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _merge_nms(self, detections: list) -> list:
        """
        Sınıf bazlı NMS ile duplikasyonları temizler.

        Tam frame + dilim inference sonuçlarını birleştirirken aynı nesneyi
        birden fazla kez tespit etmiş olabiliriz. NMS bunları temizler.

        Kurallar:
          - Aynı sınıftaki tespitler birbiriyle karşılaştırılır
          - IoU > eşik ise düşük güvenli olan çıkarılır
          - Track ID'si olan tespit korunur (tam frame'den gelen)

        Args:
            detections: Tüm tespitlerin birleşik listesi
        Returns:
            NMS uygulanmış temiz Detection listesi
        """
        if not detections:
            return []

        # Sınıf bazlı grupla
        class_groups = {}
        for det in detections:
            cls_id = det.class_id
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)

        merged = []

        for cls_id, dets in class_groups.items():
            # Güven skoruna göre azalan sırala
            # Track ID'si olanlar önce gelsin (aynı confidence'da)
            dets.sort(key=lambda d: (d.confidence, 1 if d.track_id >= 0 else 0),
                      reverse=True)

            keep = []
            suppressed = set()

            for i, det_i in enumerate(dets):
                if i in suppressed:
                    continue

                keep.append(det_i)

                for j in range(i + 1, len(dets)):
                    if j in suppressed:
                        continue

                    det_j = dets[j]
                    iou = self._compute_iou(det_i.bbox, det_j.bbox)
                    
                    # Kutu kapsama kontrolü (küçük kutu büyük kutunun içinde mi?)
                    area_i = det_i.area()
                    area_j = det_j.area()
                    
                    x1 = max(det_i.bbox[0], det_j.bbox[0])
                    y1 = max(det_i.bbox[1], det_j.bbox[1])
                    x2 = min(det_i.bbox[2], det_j.bbox[2])
                    y2 = min(det_i.bbox[3], det_j.bbox[3])
                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    
                    # Eğer bir kutu diğerinin %70'inden fazlasını kaplıyorsa duplikasyondur
                    is_contained = False
                    if area_j > 0 and inter_area / area_j > 0.7:
                        is_contained = True  # j kutusu i'nin içinde
                    elif area_i > 0 and inter_area / area_i > 0.7:
                        is_contained = True  # i kutusu j'nin içinde
                    
                    if iou > self.merge_iou or is_contained:
                        # Duplikasyon: düşük güvenli olanı bastır
                        # Ama track_id olan korunur (tam frame'den gelen)
                        if det_i.track_id >= 0 and det_j.track_id < 0:
                            suppressed.add(j)
                        elif det_i.track_id < 0 and det_j.track_id >= 0:
                            # j'nin track_id'si var, i'yi değiştir
                            keep[-1] = det_j
                            suppressed.add(i)
                            break
                        else:
                            suppressed.add(j)

            merged.extend(keep)

        return merged

    # ──────────────────────────────────────────────────────────────────────
    # 6. BBox Post-Processing (detector.py'den alınan stabilizasyon)
    # ──────────────────────────────────────────────────────────────────────

    def _post_process(self, detections: list, frame_w: int, frame_h: int) -> list:
        """
        Birleştirilmiş tespitlere son işlemleri uygular:
          - Frame sınırlarına kırpma
          - Kenar bbox genişleme kontrolü
          - BBox boyut stabilizasyonu
          - İnsan tespitleri için temporal confidence stabilizasyonu
        """
        # İnsan tespitleri için zaman boyutunda güven (confidence) stabilizasyonu
        detections = self._stabilize_human_confidence(detections)
        
        processed = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Frame sınırlarına kırp
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(frame_w), x2)
            y2 = min(float(frame_h), y2)

            # Araç genişlik kontrolü
            if det.class_id in config.VEHICLE_CLASSES:
                w = x2 - x1
                h = y2 - y1
                margin = config.BBOX_EDGE_MARGIN
                is_at_edge = (
                    x1 < margin or y1 < margin or
                    x2 > frame_w - margin or y2 > frame_h - margin
                )

                if not is_at_edge and w > 0 and h > 0:
                    if self._global_w_ema is None:
                        self._global_w_ema = w
                    else:
                        self._global_w_ema = 0.95 * self._global_w_ema + 0.05 * w
                elif is_at_edge and self._global_w_ema is not None:
                    max_w = self._global_w_ema * 1.6
                    if w > max_w:
                        cx = (x1 + x2) / 2
                        x1 = cx - max_w / 2
                        x2 = cx + max_w / 2

            # BBox stabilizasyonu (tracking varsa)
            if det.track_id >= 0:
                x1, y1, x2, y2 = self._stabilize_bbox_size(
                    det.track_id, x1, y1, x2, y2, frame_w, frame_h
                )

            det.bbox = (x1, y1, x2, y2)
            processed.append(det)

        return processed

    def _stabilize_bbox_size(
        self, track_id: int,
        x1: float, y1: float, x2: float, y2: float,
        frame_w: int, frame_h: int,
    ) -> tuple:
        """Ardışık frame'ler arasında bbox boyutunun ani değişimini önler."""
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

        margin = config.BBOX_EDGE_MARGIN
        is_at_edge = (
            x1 < margin or y1 < margin or
            x2 > frame_w - margin or y2 > frame_h - margin
        )
        area_ratio = (curr_w * curr_h) / (prev_w * prev_h)

        if is_at_edge and area_ratio > config.BBOX_MAX_AREA_CHANGE_RATIO:
            alpha = config.BBOX_SMOOTHING_ALPHA
            curr_w = alpha * curr_w + (1 - alpha) * prev_w
            curr_h = alpha * curr_h + (1 - alpha) * prev_h
        else:
            alpha_normal = 0.6
            curr_w = alpha_normal * curr_w + (1 - alpha_normal) * prev_w
            curr_h = alpha_normal * curr_h + (1 - alpha_normal) * prev_h

        x1 = curr_cx - curr_w / 2
        y1 = curr_cy - curr_h / 2
        x2 = curr_cx + curr_w / 2
        y2 = curr_cy + curr_h / 2

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(float(frame_w), x2)
        y2 = min(float(frame_h), y2)

        self._bbox_history[track_id] = (x1, y1, x2, y2)
        return x1, y1, x2, y2

    # ──────────────────────────────────────────────────────────────────────
    # Ana Detect Metodu
    # ──────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """
        SAHI pipeline: tam frame + dilimli inference → merge NMS.

        Args:
            frame: BGR formatında numpy dizisi.

        Returns:
            Detection nesnelerinin listesi.
        """
        frame_h, frame_w = frame.shape[:2]
        all_detections = []

        # Adım 1: Tam frame inference (tracking ile)
        if self.full_frame_enabled:
            full_dets = self._detect_full_frame(frame)
            self._last_full_det_count = len(full_dets)
            all_detections.extend(full_dets)
        else:
            self._last_full_det_count = 0

        # Adım 2: Dilimli inference (tracking kapalı)
        slice_dets = self._detect_slices(frame)
        self._last_slice_det_count = len(slice_dets)
        all_detections.extend(slice_dets)

        # Adım 3: NMS ile birleştir
        merged = self._merge_nms(all_detections)
        self._last_merged_det_count = len(merged)

        # Adım 4: Post-processing (bbox stabilizasyonu vb.)
        merged = self._post_process(merged, frame_w, frame_h)

        return merged

    def _stabilize_human_confidence(self, detections: list) -> list:
        """
        İnsan tespitlerinin güven (confidence) skorlarını zamansal olarak yumuşatır.
        Bu sayede 1-2 frame süren çim/gölge gibi yanlış pozitifler filtrelenir.
        """
        alpha = 0.5
        iou_thresh = 0.1
        
        matched_ids = set()
        filtered_detections = []
        
        for det in detections:
            if det.class_id != config.HUMAN_CLASS_ID:
                filtered_detections.append(det)
                continue
                
            best_iou = 0
            best_id = -1
            for k, v in self._human_conf_history.items():
                if k in matched_ids:
                    continue
                iou = self._compute_iou(det.bbox, v["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = k
                    
            if best_iou > iou_thresh:
                # Eşleşme bulundu, confidence'ı EMA ile güncelle
                matched_ids.add(best_id)
                prev_conf = self._human_conf_history[best_id]["smoothed_conf"]
                new_conf = alpha * det.confidence + (1 - alpha) * prev_conf
                
                self._human_conf_history[best_id]["smoothed_conf"] = new_conf
                self._human_conf_history[best_id]["bbox"] = det.bbox
                self._human_conf_history[best_id]["missed_frames"] = 0
                
                det.confidence = new_conf
                # Sadece tutarlı olanları (0.45 üzeri) gönder
                if new_conf >= 0.45:
                    filtered_detections.append(det)
            else:
                # Yeni bir insan tespiti
                self._human_id_counter += 1
                new_id = self._human_id_counter
                # Yeni tespitlerde başlangıç confidence'ını düşük tut (tek frame'de fırlamasın)
                new_conf = alpha * det.confidence
                
                self._human_conf_history[new_id] = {
                    "bbox": det.bbox,
                    "smoothed_conf": new_conf,
                    "missed_frames": 0
                }
                
                det.confidence = new_conf
                if new_conf >= 0.45:
                    filtered_detections.append(det)

        # Eşleşmeyen (kaybolan) geçmiş tespitleri zayıflat ve sil
        for k in list(self._human_conf_history.keys()):
            if k not in matched_ids:
                self._human_conf_history[k]["missed_frames"] += 1
                self._human_conf_history[k]["smoothed_conf"] *= 0.8
                
                if self._human_conf_history[k]["missed_frames"] > 5 or self._human_conf_history[k]["smoothed_conf"] < 0.1:
                    del self._human_conf_history[k]

        return filtered_detections

    def get_sahi_stats(self) -> dict:
        """Son frame'in SAHI istatistiklerini döndürür."""
        return {
            "slice_count": self._last_slice_count,
            "full_frame_detections": self._last_full_det_count,
            "slice_detections": self._last_slice_det_count,
            "after_nms": self._last_merged_det_count,
        }

    def get_model_info(self) -> dict:
        """Model bilgilerini döndürür."""
        return {
            "model_path": self.model_path,
            "class_names": self.model.names,
            "class_mapping": self.class_mapping,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "use_tracking": self.use_tracking,
            "sahi_enabled": True,
            "slice_size": f"{self.slice_w}×{self.slice_h}",
            "overlap_ratio": self.overlap_ratio,
        }
