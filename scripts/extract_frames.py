# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Video Frame Çıkarma Aracı
# ==============================================================================
# Bir video dosyasından belirli aralıklarla frame çıkarır.
#
# Kullanım:
#   python scripts/extract_frames.py --video video.mp4
#   python scripts/extract_frames.py --video video.mp4 --interval 10
#   python scripts/extract_frames.py --video video.mp4 --output frames/
# ==============================================================================

import cv2
import os
import argparse


def extract_frames(video_path: str, output_dir: str = ".", interval: int = 5, max_seconds: int = 30):
    """
    Video dosyasından belirli aralıklarla frame çıkarır.

    Args:
        video_path: Video dosyası yolu
        output_dir: Çıktı dizini
        interval: Frame çıkarma aralığı (saniye)
        max_seconds: Maksimum süre (saniye)
    """
    if not os.path.exists(video_path):
        print(f"[HATA] Video bulunamadı: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Toplam kare: {total_frames}, Süre: {duration:.1f}s")
    print(f"Çıkarma aralığı: her {interval} saniye")

    seconds_to_extract = list(range(0, min(int(duration), max_seconds) + 1, interval))

    for sec in seconds_to_extract:
        frame_id = int(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}_frame_{sec}s.jpg")
            cv2.imwrite(output_path, frame)
            print(f"  ✓ Kaydedildi: {output_path}")
        else:
            print(f"  ✗ {sec}. saniyedeki kare okunamadı")

    cap.release()
    print(f"\nToplam {len(seconds_to_extract)} frame çıkarıldı.")


def main():
    parser = argparse.ArgumentParser(
        description="Video dosyasından frame çıkarma aracı"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Video dosyası yolu"
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Çıktı dizini (varsayılan: mevcut dizin)"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Frame çıkarma aralığı - saniye (varsayılan: 5)"
    )
    parser.add_argument(
        "--max-seconds", type=int, default=30,
        help="Maksimum süre - saniye (varsayılan: 30)"
    )

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.interval, args.max_seconds)


if __name__ == "__main__":
    main()
