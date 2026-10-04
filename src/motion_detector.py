# ==============================================================================
# Teknofest Havacılıkta Yapay Zekâ - Hareket Algılama Modülü
# ==============================================================================
# Kamera kompanzasyonlu hareket algılama.
# Drone kamerası sürekli hareket halinde olduğundan, sabit nesneler de
# görüntüde hareketli görünür. Bu modül kamera hareketini çıkararak
# taşıtların gerçek hareketini tespit eder.
#
# Yöntem:
#   1. ORB öznitelikleri ile ardışık kareler arasında eşleşme
#   2. Homography matrisi ile kamera hareketi tahmini
#   3. Her taşıt için: önceki konumu homography ile dönüştürüp beklenen
#      konum ile gerçek konum arasındaki farkı hesaplama
#   4. Hysteresis (çift eşik) ile kararlı karar — flip-flop önleme
#   5. EMA yumuşatma + çok kareli ağırlıklı oylama ile kararlı karar
# ==============================================================================

import cv2
import numpy as np
from collections import defaultdict

import config


class MotionDetector:
    """
    Kamera kompanzasyonlu hareket algılama sınıfı.

    Ardışık kareler arasında homography tabanlı kamera hareketi tahmini yapar
    ve taşıtların gerçek hareketini tespit eder.

    İyileştirmeler:
      - Hysteresis: Çift eşik ile flip-flop önleme
      - EMA: Exponential Moving Average ile displacement yumuşatma
      - Ağırlıklı oylama: Son frame'lere daha fazla ağırlık
    """

    def __init__(self):
        # ORB öznitelik çıkarıcı
        self.orb = cv2.ORB_create(nfeatures=config.ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Önceki kare bilgileri
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Takip geçmişi: track_id → {centers, motion_votes, ema_displacement, last_status, bboxes, displacements}
        self.track_history = defaultdict(lambda: {
            "centers": [],
            "motion_votes": [],
            "ema_displacement": 0.0,
            "last_status": 0,  # Son kararlı durum (hysteresis için)
            "bboxes": [],      # BBox geçmişi (boyut kararlılığı için)
            "displacements": [],  # Ham displacement geçmişi (median filtre için)
        })

        # Son hesaplanan homography
        self.last_homography = None

    def _compute_features(self, gray: np.ndarray, mask: np.ndarray = None):
        """Gri tonlamalı görüntüden ORB özniteliklerini hesaplar."""
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        return keypoints, descriptors

    def _estimate_homography(self, curr_gray: np.ndarray, mask: np.ndarray = None):
        """
        Önceki ve şimdiki kare arasında kamera hareketini temsil eden
        homography matrisini tahmin eder.

        Returns:
            H: 3x3 homography matrisi (None başarısızsa)
            success: bool
        """
        if self.prev_gray is None:
            return None, False

        curr_kp, curr_des = self._compute_features(curr_gray, mask)

        if curr_des is None or self.prev_descriptors is None:
            self.prev_keypoints = curr_kp
            self.prev_descriptors = curr_des
            return None, False

        # Öznitelik eşleşmesi
        try:
            matches = self.bf.match(self.prev_descriptors, curr_des)
        except cv2.error:
            self.prev_keypoints = curr_kp
            self.prev_descriptors = curr_des
            return None, False

        if len(matches) < 10:
            self.prev_keypoints = curr_kp
            self.prev_descriptors = curr_des
            return None, False

        # Eşleşen noktaları çıkar
        src_pts = np.float32(
            [self.prev_keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [curr_kp[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # RANSAC ile homography hesapla
        H, inlier_mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, config.RANSAC_THRESHOLD
        )

        # Önceki bilgileri güncelle
        self.prev_keypoints = curr_kp
        self.prev_descriptors = curr_des

        if H is not None and inlier_mask is not None:
            # Homography kalite kontrolü: inlier oranını kontrol et
            inlier_ratio = inlier_mask.sum() / max(len(inlier_mask), 1)
            if inlier_ratio < 0.4:
                # Çok az inlier → homography güvenilir değil, kullanma
                return None, False
            self.last_homography = H
            return H, True
        return None, False

    def _compute_residual_displacement(
        self, H: np.ndarray, prev_center: tuple, curr_center: tuple
    ) -> float:
        """
        Kamera hareketi kompanzasyonu sonrası artık yer değiştirmeyi hesaplar.

        Args:
            H: Homography matrisi (prev → curr)
            prev_center: Önceki karede nesne merkezi (cx, cy)
            curr_center: Şimdiki karede nesne merkezi (cx, cy)

        Returns:
            Artık yer değiştirme (piksel)
        """
        if H is None:
            return 0.0

        # Önceki merkezi homography ile dönüştür (kamera hareketi uygulanmış beklenen konum)
        prev_pt = np.float32([[prev_center]]).reshape(-1, 1, 2)
        expected_pt = cv2.perspectiveTransform(prev_pt, H)
        expected_cx, expected_cy = expected_pt[0][0]

        # Gerçek konum ile beklenen konum arasındaki fark
        actual_cx, actual_cy = curr_center
        displacement = np.sqrt(
            (actual_cx - expected_cx) ** 2 + (actual_cy - expected_cy) ** 2
        )
        return displacement

    def _apply_hysteresis(self, ema_displacement: float, last_status: int, dyn_high: float, dyn_low: float) -> int:
        """
        Hysteresis (çift eşik) ile hareket kararı verir.

        Flip-flop'u önlemek için:
          - Hareketsiz → Hareketli geçişi: displacement > dyn_high olmalı
          - Hareketli → Hareketsiz geçişi: displacement < dyn_low olmalı
          - Aradaki bölgede önceki durum korunur

        Args:
            ema_displacement: EMA yumuşatılmış displacement
            last_status: Önceki kararlı hareket durumu (0 veya 1)
            dyn_high: Dinamik yüksek eşik
            dyn_low: Dinamik düşük eşik

        Returns:
            Yeni hareket durumu (0 veya 1)
        """
        if last_status == 0:
            # Şu an hareketsiz: hareketli olması için yüksek eşiği geçmeli
            if ema_displacement > dyn_high:
                return 1
            return 0
        else:
            # Şu an hareketli: hareketsiz olması için düşük eşiğin altına düşmeli
            if ema_displacement < dyn_low:
                return 0
            return 1

    def _weighted_vote(self, votes: list) -> float:
        """
        Uniform oylama: tüm frame'lere eşit ağırlık verir.

        Spike'lara dayanıklılık için son frame'lere ekstra ağırlık verilmez.

        Args:
            votes: motion_votes listesi (0 veya 1)

        Returns:
            Ortalama [0, 1]
        """
        if not votes:
            return 0.0
        return sum(votes) / len(votes)

    def _match_detections_to_tracks(self, detections: list, prev_detections: list):
        """
        Track ID'si olmayan tespitler için IoU tabanlı eşleştirme yapar.

        Bu fonksiyon sadece tracking devre dışıyken kullanılır.
        Tracking açıksa model zaten track_id sağlar.
        """
        if not prev_detections:
            return

        for det in detections:
            if det.track_id >= 0:
                continue  # Zaten track_id var
            if det.class_id not in config.VEHICLE_CLASSES:
                continue

            best_iou = 0
            best_prev = None
            for prev in prev_detections:
                if prev.class_id != det.class_id:
                    continue
                iou = self._compute_iou(det.bbox, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = prev

            if best_iou > config.TRACKING_IOU_THRESHOLD and best_prev is not None:
                det.track_id = best_prev.track_id

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

    def update(self, frame: np.ndarray, detections: list) -> list:
        """
        Yeni bir kare ve tespitlerle hareket durumunu günceller.

        Args:
            frame: BGR formatında numpy dizisi.
            detections: Detection nesnelerinin listesi.

        Returns:
            Hareket durumu güncellenmiş Detection listesi.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Maske oluşturma: Araçların bulunduğu alanları siyah (0), diğer yerleri beyaz (255) yap
        # Böylece ORB algoritması sadece hareket etmeyen arka planı (yol, bina vb.) referans alır.
        mask = np.ones_like(gray) * 255
        for det in detections:
            if det.class_id in config.VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, det.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
                # Kutu içini sıfırla ki ORB o bölgeden öznitelik almasın
                mask[y1:y2, x1:x2] = 0

        # İlk kare → tüm taşıtlar hareketsiz
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints, self.prev_descriptors = self._compute_features(gray, mask)
            for det in detections:
                if det.class_id in config.VEHICLE_CLASSES:
                    det.motion_status = 1  # Hareketli (GPU'da trackler düzelince ilk görülenler hareketlidir)
                    # Geçmişe ekle
                    if det.track_id >= 0:
                        self.track_history[det.track_id]["centers"].append(det.center())
                        self.track_history[det.track_id]["motion_votes"].append(1)
                        self.track_history[det.track_id]["bboxes"].append(det.bbox)
            return detections

        # Kamera hareketini tahmin et (Maske ile sadece arka planı kullan)
        H, h_success = self._estimate_homography(gray, mask)

        # Kamera hareket büyüklüğünü hesapla (hızlı dönüşlerde eşikleri esnetmek için)
        cam_speed = 0.0
        if h_success and H is not None:
            h_h, h_w = gray.shape
            center_pt = np.float32([[[h_w / 2, h_h / 2]]])
            new_center_pt = cv2.perspectiveTransform(center_pt, H)
            cam_speed = np.linalg.norm(new_center_pt[0][0] - center_pt[0][0])
            
        # Dinamik eşikler: Kamera ne kadar hızlı hareket ediyorsa, hata payı o kadar artar.
        # Güçlendirilmiş formül: cam_speed=3.7 → speed_penalty=1.85 (eskisi: 0.22)
        speed_penalty = (cam_speed / 5.0) * 1.0
        dyn_high_thresh = config.MOTION_HYSTERESIS_HIGH + speed_penalty
        dyn_low_thresh = config.MOTION_HYSTERESIS_LOW + (speed_penalty * 0.5)

        # Her taşıt için hareket durumunu belirle
        for det in detections:
            if det.class_id not in config.VEHICLE_CLASSES:
                det.motion_status = -1
                continue

            track_id = det.track_id
            curr_center = det.center()

            if track_id < 0 or not h_success:
                # Takip veya homography yoksa hareketsiz varsay
                det.motion_status = 0
                continue

            history = self.track_history[track_id]

            if len(history["centers"]) > 0:
                prev_center = history["centers"][-1]

                # Kamera kompanzasyonlu artık yer değiştirme
                raw_displacement = self._compute_residual_displacement(
                    H, prev_center, curr_center
                )

                # Araç boyutuna göre normalizasyon (iptal edildi, çünkü gürültü sabittir)
                # Gürültü sabit bir piksel hatası (1-3px) oluşturduğu için normalizasyon 
                # büyük araçlarda gerçek hareketi maskeliyordu.
                displacement = raw_displacement

                # Median outlier filtresi: ani spike'ları bastır
                disp_history = history["displacements"]
                if len(disp_history) >= 3:
                    recent_disps = disp_history[-5:]
                    median_disp = float(np.median(recent_disps))
                    # Eğer yeni displacement, medianın 4 katından fazlaysa → outlier
                    if displacement > max(median_disp * 4.0, dyn_high_thresh * 2.0):
                        displacement = median_disp  # Outlier'ı median ile değiştir
                disp_history.append(displacement)

                # EMA ile displacement yumuşatma
                alpha = config.MOTION_EMA_ALPHA
                prev_ema = history["ema_displacement"]
                ema_disp = alpha * displacement + (1 - alpha) * prev_ema
                history["ema_displacement"] = ema_disp

                # Hysteresis ile anlık karar
                last_status = history["last_status"]
                is_moving = self._apply_hysteresis(ema_disp, last_status, dyn_high_thresh, dyn_low_thresh)

                # Geçmişe ekle
                history["motion_votes"].append(is_moving)

                # Son N kare üzerinden oylama (yumuşatma)
                recent_votes = history["motion_votes"][-config.MOTION_MIN_FRAMES:]
                if len(recent_votes) >= config.MOTION_MIN_FRAMES:
                    avg_vote = self._weighted_vote(recent_votes)
                    det.motion_status = 1 if avg_vote > config.MOTION_VOTE_THRESHOLD else 0
                else:
                    det.motion_status = is_moving

                # Kararlı durumu güncelle (hysteresis için)
                history["last_status"] = det.motion_status
            else:
                # İlk görülme → hareketli varsay (kamera açısına yeni giren araç genelde hareketlidir)
                det.motion_status = 1
                history["motion_votes"].append(1)
                # Başlangıç yer değiştirmesini yüksek tutalım ki algoritma hemen hareketsiz yapmasın
                history["displacements"].append(dyn_high_thresh + 1.0) 

            # Merkez ve bbox geçmişini güncelle
            history["centers"].append(curr_center)
            history["bboxes"].append(det.bbox)

            # Geçmiş uzunluğunu sınırla (bellek tasarrufu)
            max_history = config.MOTION_MIN_FRAMES * 3
            if len(history["centers"]) > max_history:
                history["centers"] = history["centers"][-max_history:]
                history["motion_votes"] = history["motion_votes"][-max_history:]
                history["bboxes"] = history["bboxes"][-max_history:]
                history["displacements"] = history["displacements"][-max_history:]

        # Önceki kareyi güncelle
        self.prev_gray = gray

        return detections

    def reset(self):
        """Tüm dahili durumu sıfırlar (yeni video için)."""
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.track_history.clear()
        self.last_homography = None
