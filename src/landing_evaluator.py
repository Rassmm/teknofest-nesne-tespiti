# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - İniş Uygunluğu Değerlendirme Modülü
# ==============================================================================
# UAP (Uçan Araba Park Alanı) ve UAİ (Uçan Ambulans İniş Alanı) için
# iniş uygunluğunu değerlendiren modül.
#
# Kurallar:
#   1. Alan tamamının kare içinde olması gerekir (kenar kontrolü)
#   2. Alan üzerinde herhangi bir cisim (insan, taşıt, vb.) olmamalıdır
#   3. Kamera açısı nedeniyle yakın cisimler alan üzerinde gibi görünebilir
#   4. Kısmen görüntü dışında olan alan tespit edilmeli ama "uygun değil"
# ==============================================================================

import numpy as np

import config


class LandingEvaluator:
    """
    UAP ve UAİ iniş alanlarının uygunluğunu değerlendiren sınıf.

    İki ana kontrol yapar:
    1. Alanın tamamının kare içinde olup olmadığı
    2. Alan üzerinde herhangi bir cisim bulunup bulunmadığı
    """

    def __init__(self):
        self.edge_margin = config.LANDING_EDGE_MARGIN
        self.overlap_threshold = config.LANDING_OVERLAP_IOU_THRESHOLD

    def _is_fully_in_frame(self, bbox: tuple, frame_shape: tuple) -> bool:
        """
        Bounding box'ın tamamının kare sınırları içinde olup olmadığını kontrol eder.

        Args:
            bbox: (x1, y1, x2, y2)
            frame_shape: (height, width, channels)

        Returns:
            True: Tamamen kare içinde
            False: Kısmen veya tamamen kare dışında
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]

        # Kenar kontrolü: bbox kenarı frame kenarına çok yakınsa kısmen dışarıda
        if x1 < self.edge_margin:
            return False
        if y1 < self.edge_margin:
            return False
        if x2 > (w - self.edge_margin):
            return False
        if y2 > (h - self.edge_margin):
            return False

        return True

    def _check_overlap_with_objects(self, zone_bbox: tuple, detections: list) -> bool:
        """
        İniş alanı üzerinde herhangi bir cisim olup olmadığını kontrol eder.

        Args:
            zone_bbox: İniş alanının (x1, y1, x2, y2) koordinatları
            detections: Tüm tespit edilmiş nesneler listesi

        Returns:
            True: Alan üzerinde cisim var (iniş uygun DEĞİL)
            False: Alan temiz (iniş uygun)
        """
        zx1, zy1, zx2, zy2 = zone_bbox
        zone_area = max(0, zx2 - zx1) * max(0, zy2 - zy1)

        if zone_area <= 0:
            return True  # Geçersiz bbox → uygun değil
        for det in detections:
            # İniş alanlarını kendi aralarında karşılaştırma
            if det.class_id in config.LANDING_ZONE_CLASSES:
                continue

            dx1, dy1, dx2, dy2 = det.bbox

            # Kesişim hesapla
            overlap_x1 = max(zx1, dx1)
            overlap_y1 = max(zy1, dy1)
            overlap_x2 = min(zx2, dx2)
            overlap_y2 = min(zy2, dy2)

            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                # Herhangi bir kesişim var → uygun değil
                inter_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                ratio = inter_area / zone_area

                if ratio > self.overlap_threshold:
                    return True  # Cisim var

        return False  # Temiz

    def evaluate(self, frame: np.ndarray, detections: list) -> list:
        """
        Tüm tespitler için iniş uygunluğunu değerlendirir.

        Sadece UAP (class_id=2) ve UAİ (class_id=3) sınıfları için
        iniş durumu belirlenir. Diğer sınıflar -1 kalır.

        Args:
            frame: BGR formatında numpy dizisi (kare boyutu için)
            detections: Detection nesnelerinin listesi

        Returns:
            İniş durumu güncellenmiş Detection listesi
        """
        frame_shape = frame.shape

        for det in detections:
            if det.class_id not in config.LANDING_ZONE_CLASSES:
                det.landing_status = -1
                continue

            # Kontrol 1: Alan tamamı kare içinde mi?
            fully_visible = self._is_fully_in_frame(det.bbox, frame_shape)

            if not fully_visible:
                # Kısmen dışarıda → tespit edilir ama iniş uygun değil
                det.landing_status = 0
                continue

            # Kontrol 2: Alan üzerinde cisim var mı?
            has_obstruction = self._check_overlap_with_objects(det.bbox, detections)

            if has_obstruction:
                # Alan üzerinde cisim var → uygun değil
                det.landing_status = 0
            else:
                # Tamamen görünür ve temiz → uygun
                det.landing_status = 1

        return detections

    @staticmethod
    def get_landing_status_text(status: int) -> str:
        """İniş durumu ID'sini okunabilir metne çevirir."""
        mapping = {
            -1: "İniş alanı değil",
            0: "Uygun değil",
            1: "Uygun",
        }
        return mapping.get(status, "Bilinmiyor")
