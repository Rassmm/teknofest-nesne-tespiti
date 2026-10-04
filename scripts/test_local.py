# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Lokal Test Script'i
# ==============================================================================
# Video üzerinde tüm pipeline'ı test eder, sonuçları görselleştirir ve kaydeder.
#
# Kullanım:
#   python scripts/test_local.py --video video.mp4
#   python scripts/test_local.py --video video.mp4 --output sonuclar/
#   python scripts/test_local.py --video video.mp4 --show   (canlı önizleme)
# ==============================================================================

import os
import sys
import time
import argparse
import json

# src/ modüllerini import edebilmek için yol ekle
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import cv2
import numpy as np

import config
from detector import ObjectDetector
from sahi_detector import SAHIDetector
from motion_detector import MotionDetector
from landing_evaluator import LandingEvaluator
from formatter import ResultFormatter


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Tespit sonuçlarını kare üzerine çizer.

    Renkler sınıfa göre belirlenir:
      - Taşıt: Turuncu + hareket durumu
      - İnsan: Yeşil
      - UAP: Mavi + iniş durumu
      - UAİ: Kırmızı + iniş durumu
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = config.VIS_COLORS.get(det.class_id, (255, 255, 255))
        class_name = config.CLASS_NAMES.get(det.class_id, f"cls_{det.class_id}")

        # Eğer taşıt ise hareket durumuna göre kutu rengini belirle (Kırmızı/Sarı)
        if det.class_id in config.VEHICLE_CLASSES:
            if det.motion_status == 1:
                color = (0, 0, 255)    # Hareketli -> Kırmızı (BGR)
            else:
                color = (0, 255, 255)  # Hareketsiz -> Sarı (BGR)

        # Bounding box
        thickness = 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Etiket oluştur
        label_parts = [f"{class_name} {det.confidence:.2f}"]

        # Hareket durumu (sadece taşıtlar)
        if det.class_id in config.VEHICLE_CLASSES:
            motion_text = "Hareketli" if det.motion_status == 1 else "Hareketsiz"
            motion_color = color  # Kutunun rengiyle aynı olsun
            label_parts.append(motion_text)
            # Hareket göstergesi: bbox köşesinde küçük daire
            indicator_pos = (x2 - 10, y1 + 10)
            cv2.circle(annotated, indicator_pos, 6, motion_color, -1)

        # İniş durumu (sadece UAP/UAİ)
        if det.class_id in config.LANDING_ZONE_CLASSES:
            if det.landing_status == 1:
                landing_text = "UYGUN"
                landing_color = (0, 255, 0)
            else:
                landing_text = "UYGUN DEGIL"
                landing_color = (0, 0, 255)
            label_parts.append(landing_text)
            # İniş alanı etrafında kesikli çerçeve efekti
            cv2.rectangle(annotated, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), landing_color, 1)

        # Track ID
        if det.track_id >= 0:
            label_parts.append(f"ID:{det.track_id}")

        label = " | ".join(label_parts)

        # Etiket arka planı
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 8),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_stats_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    det_count: int,
) -> np.ndarray:
    """Sol üst köşeye istatistik bilgilerini ekler."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Yarı saydam arka plan
    cv2.rectangle(overlay, (0, 0), (350, 100), (0, 0, 0), -1)
    frame_out = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame.copy())

    texts = [
        f"Kare: {frame_idx + 1}/{total_frames}",
        f"FPS: {fps:.1f}",
        f"Tespit: {det_count}",
    ]

    for i, text in enumerate(texts):
        cv2.putText(
            frame_out,
            text,
            (10, 25 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_out


def run_local_test(
    video_path: str,
    output_dir: str,
    show_preview: bool = False,
    save_video: bool = True,
    save_json: bool = True,
    max_frames: int = None,
    use_sahi: bool = True,
):
    """
    Lokal video üzerinde tam pipeline testi çalıştırır.

    Args:
        video_path: Video dosyası yolu
        output_dir: Çıktı dizini
        show_preview: Canlı önizleme göster
        save_video: Annotated video kaydet
        save_json: JSON sonuçlarını kaydet
        max_frames: Maksimum kare sayısı (None = tamamı)
    """
    # Giriş kontrolü
    if not os.path.exists(video_path):
        print(f"[HATA] Video bulunamadı: {video_path}")
        sys.exit(1)

    # Çıktı dizini
    os.makedirs(output_dir, exist_ok=True)

    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("  Teknofest Lokal Test")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Çözünürlük: {frame_w}x{frame_h}")
    print(f"  Video FPS: {video_fps:.1f}")
    print(f"  Toplam kare: {total_frames}")
    if max_frames:
        print(f"  İşlenecek kare: {max_frames}")
    sahi_active = use_sahi and getattr(config, 'SAHI_ENABLED', False)
    print(f"  SAHI: {'AKTİF' if sahi_active else 'PASIF'}")
    if sahi_active:
        print(f"  SAHI Dilim: {config.SAHI_SLICE_WIDTH}x{config.SAHI_SLICE_HEIGHT}, Overlap: {config.SAHI_OVERLAP_RATIO}")
    print("=" * 60)

    # Pipeline başlat
    pipeline_obj = None
    try:
        from pipeline import Pipeline
        pipeline_obj = Pipeline(use_tracking=True)
    except Exception as e:
        print(f"[UYARI] Pipeline import hatası: {e}")
        print("[UYARI] Modüller ayrı ayrı başlatılıyor...")
        # Yedek: modülleri ayrı başlat

    if pipeline_obj is None:
        if sahi_active:
            detector = SAHIDetector(use_tracking=True)
        else:
            detector = ObjectDetector(use_tracking=True)
        motion_det = MotionDetector()
        landing_eval = LandingEvaluator()
        formatter = ResultFormatter()
    else:
        detector = pipeline_obj.detector
        motion_det = pipeline_obj.motion_detector
        landing_eval = pipeline_obj.landing_evaluator
        formatter = pipeline_obj.formatter

    # SAHI durumunu kontrol et
    is_sahi_detector = hasattr(detector, 'get_sahi_stats')

    # Video yazıcı
    video_writer = None
    if save_video:
        output_video_path = os.path.join(output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = video_fps if video_fps > 0 else config.VIDEO_FPS
        video_writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (frame_w, frame_h))
        print(f"  Çıktı video: {output_video_path}")

    # JSON sonuçları
    all_results = []

    # İşleme döngüsü
    frame_idx = 0
    total_inference_time = 0
    process_total = max_frames if max_frames else total_frames

    print(f"\n{'─' * 60}")
    print("  İşleme başlıyor...")
    print(f"{'─' * 60}\n")

    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        # 1. Tespit
        detections = detector.detect(frame)

        # 2. Hareket durumu
        detections = motion_det.update(frame, detections)

        # 3. İniş uygunluğu
        detections = landing_eval.evaluate(frame, detections)

        # 4. Doğrulama
        detections = formatter.validate_detections(detections)

        t_elapsed = time.time() - t_start
        total_inference_time += t_elapsed
        current_fps = 1.0 / t_elapsed if t_elapsed > 0 else 0

        # Özet yazdır
        formatter.print_summary(frame_idx, detections)

        # SAHI istatistiklerini yazdır
        if is_sahi_detector:
            stats = detector.get_sahi_stats()
            print(f"           SAHI: {stats['slice_count']} dilim, "
                  f"full={stats['full_frame_detections']}, "
                  f"slice={stats['slice_detections']}, "
                  f"merged={stats['after_nms']}")

        # JSON'a ekle
        if save_json:
            result = formatter.format_frame_result(frame_idx, detections)
            all_results.append(result)

        # Görselleştir
        annotated = draw_detections(frame, detections)
        annotated = draw_stats_overlay(
            annotated, frame_idx, process_total, current_fps, len(detections)
        )

        # Video yaz
        if video_writer is not None:
            video_writer.write(annotated)

        # Canlı önizleme
        if show_preview:
            # Büyük çözünürlüklerde küçült
            display = annotated
            if frame_w > 1920:
                scale = 1920 / frame_w
                display = cv2.resize(
                    annotated,
                    (int(frame_w * scale), int(frame_h * scale)),
                )
            cv2.imshow("Teknofest - Lokal Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[Bilgi] Kullanıcı tarafından durduruldu.")
                break
            elif key == ord(" "):
                # Boşluk = durakla
                print("[Bilgi] Duraklatıldı. Devam etmek için herhangi bir tuşa basın.")
                cv2.waitKey(0)

        frame_idx += 1

    # Temizlik
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    # JSON kaydet
    if save_json and all_results:
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON sonuçları: {json_path}")

    # Bitiş raporu
    avg_time = total_inference_time / max(1, frame_idx)
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("  Test Tamamlandı!")
    print(f"{'=' * 60}")
    print(f"  İşlenen kare: {frame_idx}")
    print(f"  Toplam süre: {total_inference_time:.1f} saniye")
    print(f"  Ortalama süre/kare: {avg_time * 1000:.1f} ms")
    print(f"  Ortalama FPS: {avg_fps:.1f}")
    if save_video:
        print(f"  Çıktı video: {os.path.join(output_dir, 'annotated_output.mp4')}")
    if save_json:
        print(f"  JSON sonuçları: {os.path.join(output_dir, 'results.json')}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Teknofest Nesne Tespiti - Lokal Video Testi"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Test videosu yolu (örn: video.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.OUTPUT_DIR,
        help="Çıktı dizini (varsayılan: results/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Canlı önizleme göster",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Annotated video kaydetme",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="JSON sonuçlarını kaydetme",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maksimum kare sayısı (test için sınırlama)",
    )
    parser.add_argument(
        "--no-sahi",
        action="store_true",
        help="SAHI pipeline'ını devre dışı bırak (sadece tam frame inference)",
    )

    args = parser.parse_args()

    # --no-sahi flag'i varsa config'i geçici olarak değiştir
    use_sahi = not args.no_sahi
    if args.no_sahi:
        config.SAHI_ENABLED = False

    run_local_test(
        video_path=args.video,
        output_dir=args.output,
        show_preview=args.show,
        save_video=not args.no_video,
        save_json=not args.no_json,
        max_frames=args.max_frames,
        use_sahi=use_sahi,
    )


if __name__ == "__main__":
    main()
