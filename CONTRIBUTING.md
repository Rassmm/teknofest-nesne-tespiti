# Katkıda Bulunma Rehberi (Contributing Guide)

Bu projeye katkıda bulunmak istediğiniz için teşekkür ederiz! Teknofest Havacılıkta Yapay Zekâ yarışması kapsamında geliştirilen bu nesne tespiti projesine katkı sağlamak için lütfen aşağıdaki adımları ve kuralları takip edin.

---

## 1. Dal (Branch) İsimlendirme Kuralları

Yeni bir geliştirme veya düzeltme yaparken amaca uygun bir dal açınız:

- `feature/<ozellik-adi>`: Yeni bir modül veya özellik eklerken (ör. `feature/sahi-entegrasyonu`)
- `bugfix/<hata-adi>`: Bir hatayı düzeltirken (ör. `bugfix/sunucu-baglanti-hatasi`)
- `refactor/<duzenleme>`: Kod iyileştirme ve yeniden yapılandırmalarda (ör. `refactor/veri-on-isleme`)
- `docs/<konu>`: Dokümantasyon güncellemelerinde (ör. `docs/kurulum-kilavuzu`)
- `test/<test-adi>`: Test senaryoları eklerken (ör. `test/inferans-testleri`)

---

## 2. Fork ve Pull Request (PR) Süreci

1. **Depoyu Fork Edin**: Sağ üst köşedeki **Fork** butonuna tıklayarak projeyi kendi hesabınıza kopyalayın.
2. **Yerel Klon Oluşturun**:
   ```bash
   git clone https://github.com/<kullanici-adiniz>/teknofest-nesne-tespiti.git
   cd teknofest-nesne-tespiti
   ```
3. **Yeni Dal Oluşturun**:
   ```bash
   git checkout -b feature/yeni-ozellik
   ```
4. **Değişikliklerinizi Gerçekleştirin**:
   - Gerekli kod düzenlemelerini yapın ve yerel olarak test edin.
   - Büyük veri setlerini veya video dosyalarını kesinlikle commit'e dahil etmeyin.
5. **Commit Edin**:
   - Commit mesajlarınızın açık, anlaşılır ve eylem odaklı olmasına özen gösterin:
   ```bash
   git add .
   git commit -m "feat: SAHI destekli dilimleme inferans fonksiyonu eklendi"
   ```
6. **Uzak Depoya Gönderin**:
   ```bash
   git push origin feature/yeni-ozellik
   ```
7. **Pull Request Açın**:
   - GitHub arayüzünden `main` dalına yönelik bir PR oluşturun.
   - PR açıklamasında yapılan değişiklikleri, neden yapıldığını ve test sonuçlarını özetleyin.

---

## 3. Kod ve Stil Standartları

- **PEP 8 Uyumluluğu**: Python kodlarında PEP 8 standartlarına özen gösterin.
- **Yorum Satırları**: Kod içi yorum ve açıklamalar **Türkçe**, açık ve amaca yönelik olmalıdır.
- **Tip Belirteçleri (Type Hints)**: Fonksiyon ve sınıf metotlarında parametre ve dönüş tiplerini belirtmeye özen gösterin.
- **Büyük Dosyalar**: Ağırlık modelleri (`.pt`, `.onnx` vb.) için Git LFS kullanılmalı; ham test videoları (`.mp4`, `.avi`) depoya gönderilmemelidir.

---

## 4. Hata Bildirimi (Issues)

Bir hata tespit ettiyseniz veya bir öneriniz varsa:
1. GitHub deposunda **Issues** sekmesine gidin.
2. Daha önce benzer bir bildirimin açılıp açılmadığını kontrol edin.
3. Yeni bir Issue oluştururken:
   - Açıklayıcı bir başlık yazın.
   - Hatayı yeniden üretme adımlarını (reproduction steps) belirtin.
   - Çalışma ortamı bilgilerinizi (İşletim Sistemi, Python sürümü, GPU/CUDA durumu) ekleyin.
   - Varsa konsol çıktıları ve hata loglarını kod bloğu (` ``` `) içinde paylaşın.
