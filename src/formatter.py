# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Sonuç Formatlama Modülü
# ==============================================================================
# Tespit sonuçlarını yarışma formatına dönüştüren modül.
# Sunucuya gönderilecek JSON verisini hazırlar.
# ==============================================================================

import json

import config


class ResultFormatter:
    """
    Tespit sonuçlarını yarışma sunucusunun beklediği formata dönüştürür.

    Her tespit için:
      - class_id: Sınıf ID (0: Taşıt, 1: İnsan, 2: UAP, 3: UAİ)
      - bbox: [x1, y1, x2, y2] piksel koordinatları
      - confidence: Güven skoru
      - landing_status: İniş durumu (-1, 0, 1)
      - motion_status: Hareket durumu (-1, 0, 1)
    """

    def format_single(self, detection) -> dict:
        """
        Tek bir Detection nesnesini sunucu formatına dönüştürür.

        Args:
            detection: Detection nesnesi

        Returns:
            Formatlanmış sözlük
        """
        x1, y1, x2, y2 = detection.bbox
        return {
            "class_id": int(detection.class_id),
            "cls": int(detection.class_id),
            "landing_status": int(detection.landing_status),
            "motion_status": int(detection.motion_status),
            "bbox": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2), 2),
                round(float(y2), 2),
            ],
            "confidence": round(float(detection.confidence), 4),
        }

    def format_frame_result(self, frame_index: int, detections: list) -> dict:
        """
        Bir kare için tüm tespit sonuçlarını formatlayıp tek bir sonuç sözlüğü oluşturur.

        Args:
            frame_index: Kare numarası (0-tabanlı veya sunucu tarafından belirlenen)
            detections: Detection nesnelerinin listesi

        Returns:
            Sunucuya gönderilecek tam sonuç sözlüğü
        """
        formatted_detections = []
        for det in detections:
            formatted_detections.append(self.format_single(det))

        return {
            "frame_index": frame_index,
            "detections": formatted_detections,
        }

    def to_json(self, frame_index: int, detections: list) -> str:
        """
        Sonucu JSON string'e dönüştürür.

        Args:
            frame_index: Kare numarası
            detections: Detection nesnelerinin listesi

        Returns:
            JSON string
        """
        result = self.format_frame_result(frame_index, detections)
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def validate_detections(detections: list) -> list:
        """
        Tespitlerin yarışma kurallarına uygunluğunu doğrular ve düzeltir.

        Kontroller:
          - Taşıt dışındaki sınıflarda hareket durumu -1 olmalı
          - UAP/UAİ dışındaki sınıflarda iniş durumu -1 olmalı
          - Hareket durumu sadece 0, 1 veya -1 olabilir
          - İniş durumu sadece 0, 1 veya -1 olabilir

        Args:
            detections: Detection nesnelerinin listesi

        Returns:
            Doğrulanmış ve düzeltilmiş Detection listesi
        """
        for det in detections:
            # Hareket durumu kontrolü
            if det.class_id not in config.VEHICLE_CLASSES:
                det.motion_status = -1
            elif det.motion_status not in (0, 1):
                det.motion_status = 0  # Varsayılan: hareketsiz

            # İniş durumu kontrolü
            if det.class_id not in config.LANDING_ZONE_CLASSES:
                det.landing_status = -1
            elif det.landing_status not in (0, 1):
                det.landing_status = 0  # Varsayılan: uygun değil

        return detections

    def print_summary(self, frame_index: int, detections: list):
        """Kare sonucunun özet bilgisini yazdırır."""
        counts = {}
        for det in detections:
            name = config.CLASS_NAMES.get(det.class_id, f"cls_{det.class_id}")
            counts[name] = counts.get(name, 0) + 1

        moving = sum(
            1 for d in detections
            if d.class_id in config.VEHICLE_CLASSES and d.motion_status == 1
        )
        suitable = sum(
            1 for d in detections
            if d.class_id in config.LANDING_ZONE_CLASSES and d.landing_status == 1
        )

        count_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(
            f"  Kare {frame_index:>5d} | "
            f"Toplam: {len(detections):>3d} | "
            f"{count_str} | "
            f"Hareketli taşıt: {moving} | "
            f"Uygun iniş: {suitable}"
        )
