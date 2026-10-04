# Teknofest Görev 1: Termal ve RGB Görüntüler İçin Çift Modelli (Dual-Model) Çözüm Boru Hattı

Bu doküman, yarışma sunucusundan karışık olarak gelecek olan renkli (RGB) ve termal (Grayscale/White-Hot) görüntülerin en yüksek doğrulukla ve hızla işlenmesi için tasarlanmış sistem mimarisini ve Colab eğitim kodunu içerir.

## 1. Çift Modelli Pipeline (Boru Hattı) Mantığı

Yarışmada sunucu bize sıradaki karenin termal mi yoksa RGB mi olduğunu söylemeyecektir. Bu yüzden belleğe (RAM/VRAM) iki farklı model yükleyip, gelen karenin renk doygunluğuna bakarak onu doğru modele yönlendireceğiz.

**Akış (Pipeline):**
1. Sunucudan 1 adet Görüntü Karesi (Frame) alınır.
2. Görüntü OpenCV ile analiz edilir (Termal mi, RGB mi?).
3. **Eğer Termal ise:** `best_thermal.pt` modeli çalıştırılır.
4. **Eğer RGB ise:** Mevcut olan `best_rgb.pt` (elinizdeki `best.pt`) modeli çalıştırılır.
5. Her iki durumdan çıkan sonuçlar, Hareket ve İniş Alanı kontrol mantığından geçirilip sunucuya tek bir formatta gönderilir.

### Görüntü Ayırıcı (Router) Fonksiyon Örneği

Termal görüntüler yapıları gereği renksizdir (Gri tonlamalı). R, G, B kanallarının birbirine olan varyansına veya doğrudan renk uzayına bakarak milisaniyeler içinde ayrım yapabiliriz.

```python
import cv2
import numpy as np

def is_thermal(frame_bgr):
    """
    Görüntünün termal (siyah-beyaz/grayscale) olup olmadığını tespit eder.
    """
    # Görüntüyü HSV formatına çeviriyoruz.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    # Saturation (Doygunluk) kanalını alıyoruz
    saturation = hsv[:, :, 1]
    
    # Eğer ortalama doygunluk çok düşükse (örneğin < 5), bu renksiz bir görüntüdür (Termaldir)
    mean_saturation = np.mean(saturation)
    
    return mean_saturation < 5.0

# Pipeline İçindeki Kullanımı
def process_frame(frame):
    if is_thermal(frame):
        # Termal modele yönlendir
        results = model_thermal(frame)
    else:
        # RGB modele yönlendir
        results = model_rgb(frame)
        
    return results
```

---

## 2. Google Colab T4 İçin En Uygun YOLO Eğitim Kodu

Elinizdeki `best.pt` ağırlıklarını başlangıç noktası alarak sadece termal verilerle yeni bir model eğiteceğiz. Colab'da T4 GPU 4 saatlik limitle çalıştığı için kodun donanımdan maksimum verim alması ve termal verilere özgü (renk artırmalarının kapalı olduğu) ayarlarla çalışması gerekir.

Aşağıdaki kodu Colab üzerinde bir hücreye kopyalayıp çalıştırabilirsiniz. (Öncesinde termal verilerinizi ve `data.yaml` dosyanızı Colab'a yüklemiş olmalısınız).

```python
# 1. Gerekli kütüphaneyi kurun
# !pip install ultralytics

from ultralytics import YOLO
import torch

# GPU'nun aktif olduğundan emin olalım
print("Cihaz:", "GPU" if torch.cuda.is_available() else "CPU")

# 2. Mevcut RGB ağırlıklarınızı (best.pt) başlangıç noktası olarak yüklüyoruz
# Bu sayede model araç ve insan şekillerini sıfırdan öğrenmek zorunda kalmaz.
model = YOLO('best.pt') 

# 3. Termal görüntüler için özelleştirilmiş eğitim parametreleri
# DİKKAT: Termal görüntülerde renk olmadığı için hsv_h, hsv_s, hsv_v değerlerini 0.0 yapıyoruz.
results = model.train(
    data='termal_veri_seti/data.yaml',  # Colab'a yüklediğiniz termal veri setinizin yaml dosyası
    epochs=100,                         # T4 GPU ile 100 epoch ortalama 1-2 saat sürecektir.
    imgsz=640,                          # Şartnamedeki Full HD ve 4K görseller için 640 veya 1280 (RAM'e göre)
    batch=16,                           # T4 GPU (16GB VRAM) için genelde 16 idealdir. Memory Error alırsanız 8'e düşürün.
    device=0,                           # T4 GPU'yu seçiyoruz
    project='Teknofest_Termal_Egitim',
    name='thermal_model',
    exist_ok=True,
    
    # --- Veri Artırma (Augmentation) Ayarları ---
    hsv_h=0.0,    # Termalde renk (hue) değişimi olmaz
    hsv_s=0.0,    # Termalde doygunluk (saturation) değişimi olmaz
    hsv_v=0.0,    # Termalde parlaklık değişimi olmaz
    
    degrees=5.0,  # Hafif rotasyon (Drone kamerasının yalpalamasını simüle eder)
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,   # Havacılık görüntüsü olduğu için yukarı/aşağı çevirme risklidir (Araçlar ters uçmaz)
    fliplr=0.5,   # Yatayda çevirme (Sağ/sol) oldukça faydalıdır.
    mosaic=1.0,   # Mozaik veri artırma araç/insan tespitinde çok etkilidir.
    mixup=0.0,
    
    workers=8,    # Colab CPU çekirdeklerine göre veri okuma hızı
    amp=True      # T4 GPU üzerinde eğitimi hızlandıran (Mixed Precision) ayarı
)

print("Eğitim tamamlandı! Yeni termal ağırlık dosyanız oluşturuldu.")
```

**Eğitim Sonrası:**
Eğitim bittiğinde Colab sol menüsünde `Teknofest_Termal_Egitim/thermal_model/weights/best.pt` dosyası oluşacaktır. Bu dosyayı bilgisayarınıza indirip adını `best_thermal.pt` olarak değiştirerek bilgisayarınızdaki Pipeline'da kullanabilirsiniz.
