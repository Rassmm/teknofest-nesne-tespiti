# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Ana Pipeline
# ==============================================================================
# Yarışma sunucusuyla iletişim kuran ve kare bazlı tespit yapan ana script.
# Akış: Sunucudan kare al → Tespit yap → Hareket durumu belirle →
#        İniş uygunluğu değerlendir → Sonucu formatla → Sunucuya gönder
# ==============================================================================

import sys
import time
import json
import argparse

import requests
import numpy as np
import cv2

import config
from detector import ObjectDetector
from sahi_detector import SAHIDetector
from motion_detector import MotionDetector
from landing_evaluator import LandingEvaluator
from formatter import ResultFormatter


class CompetitionClient:
    """Yarışma sunucusuyla HTTP üzerinden iletişim sağlar."""

    def __init__(self, server_url: str, api_key: str = ""):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def get_frame(self) -> tuple:
        """
        Sunucudan sıradaki kareyi alır.

        Returns:
            (frame_id, frame): frame_id (int/str) ve frame (numpy BGR dizisi)
            Kare yoksa (None, None)
        """
        try:
            url = f"{self.server_url}{config.API_GET_FRAME}"
            response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")

                # JSON yanıt (frame_id + image data)
                if "application/json" in content_type:
                    data = response.json()
                    frame_id = data.get("frame_id", data.get("id", 0))

                    # Görüntü URL olarak
                    if "url" in data:
                        img_resp = self.session.get(data["url"], timeout=config.REQUEST_TIMEOUT)
                        img_array = np.frombuffer(img_resp.content, np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    # Görüntü base64 olarak
                    elif "image" in data:
                        import base64
                        img_bytes = base64.b64decode(data["image"])
                        img_array = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    else:
                        frame = None

                    return frame_id, frame

                # Doğrudan görüntü yanıtı
                elif "image" in content_type:
                    frame_id = response.headers.get("X-Frame-Id", 0)
                    img_array = np.frombuffer(response.content, np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return frame_id, frame

            elif response.status_code == 204 or response.status_code == 404:
                # Video bitti
                return None, None

        except requests.exceptions.RequestException as e:
            print(f"[Pipeline] Sunucu bağlantı hatası: {e}")
            return None, None

        return None, None

    def send_result(self, frame_id, result_data: dict) -> tuple:
        """
        Tespit sonuçlarını sunucuya gönderir.

        Args:
            frame_id: Kare ID'si
            result_data: Formatlanmış sonuç sözlüğü

        Returns:
            (status_code, response_text)
        """
        try:
            url = f"{self.server_url}{config.API_SEND_RESULT}"
            payload = result_data
            payload["frame_id"] = frame_id

            response = self.session.post(
                url,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=config.REQUEST_TIMEOUT,
            )
            return response.status_code, response.text

        except requests.exceptions.RequestException as e:
            print(f"[Pipeline] Sonuç gönderme hatası: {e}")
            return -1, str(e)


class Pipeline:
    """
    Ana işlem pipeline'ı.

    Modülleri bir araya getirerek kare bazlı nesne tespiti,
    hareket algılama ve iniş uygunluğu değerlendirmesi yapar.
    """

    def __init__(self, use_tracking: bool = True):
        print("=" * 60)
        print("  Teknofest Nesne Tespiti Pipeline Başlatılıyor")
        print("=" * 60)

        # Modülleri başlat
        # SAHI aktifse SAHIDetector kullan, değilse klasik ObjectDetector
        if getattr(config, 'SAHI_ENABLED', False):
            print("[Pipeline] SAHI modu aktif — dilimli inference kullanılacak")
            self.detector = SAHIDetector(use_tracking=use_tracking)
            self.sahi_enabled = True
        else:
            self.detector = ObjectDetector(use_tracking=use_tracking)
            self.sahi_enabled = False
        self.motion_detector = MotionDetector()
        self.landing_evaluator = LandingEvaluator()
        self.formatter = ResultFormatter()

        # İstatistikler
        self.total_frames = 0
        self.total_detections = 0
        self.total_time = 0

        print("[Pipeline] Tüm modüller hazır.")
        print("=" * 60)

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> tuple:
        """
        Tek bir kareyi tüm pipeline'dan geçirir.

        Args:
            frame: BGR formatında numpy dizisi
            frame_index: Kare numarası

        Returns:
            (detections, result_dict): İşlenmiş tespitler ve formatlanmış sonuç
        """
        t_start = time.time()

        # 1. Nesne tespiti
        detections = self.detector.detect(frame)

        # 2. Hareket durumu belirleme (taşıtlar için)
        detections = self.motion_detector.update(frame, detections)

        # 3. İniş uygunluğu değerlendirme (UAP/UAİ için)
        detections = self.landing_evaluator.evaluate(frame, detections)

        # 4. Sonuçları doğrula
        detections = self.formatter.validate_detections(detections)

        # 5. Formatla
        result_dict = self.formatter.format_frame_result(frame_index, detections)

        # İstatistik güncelle
        elapsed = time.time() - t_start
        self.total_frames += 1
        self.total_detections += len(detections)
        self.total_time += elapsed

        # SAHI istatistiklerini ekle
        sahi_stats = None
        if self.sahi_enabled and hasattr(self.detector, 'get_sahi_stats'):
            sahi_stats = self.detector.get_sahi_stats()

        return detections, result_dict

    def get_stats(self) -> dict:
        """Pipeline istatistiklerini döndürür."""
        avg_time = self.total_time / max(1, self.total_frames)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "total_time_sec": round(self.total_time, 2),
            "avg_time_per_frame_ms": round(avg_time * 1000, 1),
            "avg_fps": round(avg_fps, 1),
        }

    def reset(self):
        """Pipeline'ı sıfırlar (yeni video/oturum için)."""
        self.motion_detector.reset()
        self.total_frames = 0
        self.total_detections = 0
        self.total_time = 0


def run_competition(server_url: str, api_key: str = ""):
    """
    Yarışma modunda çalıştırır: sunucudan kare al → işle → sonuç gönder.
    """
    pipeline = Pipeline(use_tracking=True)
    client = CompetitionClient(server_url, api_key)

    print(f"\n[Yarışma] Sunucu: {server_url}")
    print("[Yarışma] Kareler alınıyor...\n")

    frame_count = 0
    while True:
        # 1. Sunucudan kare al
        frame_id, frame = client.get_frame()

        if frame is None:
            print(f"\n[Yarışma] Video tamamlandı. Toplam kare: {frame_count}")
            break

        # 2. İşle
        detections, result = pipeline.process_frame(frame, frame_count)

        # 3. Sonucu gönder
        status_code, resp = client.send_result(frame_id, result)

        # 4. Özet yazdır
        pipeline.formatter.print_summary(frame_count, detections)

        if status_code != 200:
            print(f"    ⚠ Sunucu yanıt kodu: {status_code}")

        frame_count += 1

    # Bitiş istatistikleri
    stats = pipeline.get_stats()
    print("\n" + "=" * 60)
    print("  Yarışma Sonuçları")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Teknofest Nesne Tespiti Pipeline")
    parser.add_argument(
        "--server", type=str, default=config.SERVER_URL,
        help="Yarışma sunucusu URL'si"
    )
    parser.add_argument(
        "--api-key", type=str, default=config.API_KEY,
        help="API anahtarı"
    )
    args = parser.parse_args()

    run_competition(args.server, args.api_key)


if __name__ == "__main__":
    main()
