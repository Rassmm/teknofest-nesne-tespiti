# Yarışma Şartları: Nesne Tespiti

Bu doküman, Havacılıkta Yapay Zekâ yarışması kapsamındaki "Birinci Görev: Nesne Tespiti" kurallarını yapay zekaya bağlam olarak vermek için düzenlenmiştir.

## Görevin Amacı

Yarışmacılardan video kareleri üzerinde aşağıdaki nesne türlerini tespit etmeleri beklenmektedir:

1. Taşıt
2. İnsan
3. Uçan Araba Park Alanı (UAP)
4. Uçan Ambulans İniş Alanı (UAİ)

Taşıt sınıfı için nesnenin hareketli veya hareketsiz olduğu da tespit edilmelidir.

UAP ve UAİ sınıfları için alanın inişe uygun olup olmadığı da bildirilmelidir.

## Sınıflar ve ID Değerleri

| Sınıf | Sınıf ID | İniş Durumu Değerleri | Hareket Durumu Değerleri | Açıklama |
|---|---:|---|---|---|
| Taşıt | 0 | -1 | 0, 1 | Motorlu karayolu taşıtları, raylı taşıtlar ve deniz taşıtları |
| İnsan | 1 | -1 | -1 | Ayakta duran veya oturan tüm insanlar |
| Uçan Araba Park Alanı (UAP) | 2 | 0, 1 | -1 | Uçan arabanın park edebileceğini gösteren işaretli alan |
| Uçan Ambulans İniş Alanı (UAİ) | 3 | 0, 1 | -1 | Uçan ambulansın iniş yapabileceğini gösteren işaretli alan |

## Hareket Durumu Değerleri

| Hareket Durum ID | Hareket Durumu |
|---:|---|
| 0 | Hareketsiz |
| 1 | Hareketli |

Taşıt dışındaki sınıflarda hareket durumu `-1` olmalıdır.

## İniş Durumu Değerleri

| İniş Durum ID | İniş Durumu |
|---:|---|
| 0 | Uygun değil |
| 1 | Uygun |
| -1 | İniş alanı değil |

UAP ve UAİ dışındaki sınıflarda iniş durumu `-1` olmalıdır.

## Video ve Görüntü Şartları

- Yarışma videoları hava aracının kalkışını, inişini ve seyrüseferini içerebilir.
- Hava aracının yerden yüksekliği video boyunca değişebilir.
- Her oturumda verilecek video süresi normal şartlarda 5 dakikadır.
- FPS değeri 7.5 olacaktır.
- 5 dakikalık oturumda toplam 2250 görüntü karesi bulunur.
- Takımlardan normal şartlarda 2250 adet sonuç beklenir.
- Uçuş süreleri ve kare sayıları değişkenlik gösterebilir.
- Videolar Full HD veya 4K çözünürlüğünde olabilir.
- Kareler `jpg`, `png` veya başka bir görüntü formatında olabilir.
- Videolar karelere ayrılmış ve sıralı şekilde sunulacaktır.
- Görüntüler günün herhangi bir vaktinde kaydedilmiş olabilir.
- Kar, yağmur ve benzeri hava koşulları bulunabilir.
- Şehir, orman ve deniz üstü uçuş görüntüleri bulunabilir.
- Kamera açısı hava aracının hareketine bağlı olarak 70-90 derece aralığında değişebilir.
- Alt görüş kamerası kaynaklı bulanıklık, ölü piksel, bozulma, tekrar, donma veya görüntü kaybı olabilir.
- Görüntüler RGB kamera veya termal kamera ile elde edilmiş olabilir.

## Taşıt ve İnsan Tespiti Kuralları

- Görüntü karesinin tamamındaki tüm taşıtlar ve insanlar dikkate alınmalıdır.
- Tespit edilen taşıtlar hareketli veya hareketsiz olarak sınıflandırılmalıdır.
- Tamamı görünmeyen taşıt ve insanlar da tespit edilmelidir.
- Görüntü karesinden kısmen çıkmış nesneler de etiketlenmelidir.
- Başka bir objenin arkasında kısmen görünen insanlar ve taşıtlar da tespit edilmelidir.
- Görüntüde tren varsa lokomotif ve her vagon ayrı birer obje olarak tanımlanmalıdır.
- Bisiklet ve motosiklet sürücüleri ayrıca `insan` olarak etiketlenmemelidir.
- Bisiklet veya motosiklet ve sürücüsü bir bütün olarak yalnızca `taşıt` etiketiyle etiketlenmelidir.
- Scooter sürücüsüzse `taşıt` olarak etiketlenmelidir.
- Scooter sürücülü ise `insan` olarak etiketlenmelidir.
- Kamera sürekli hareket halinde olduğundan sabit taşıtlar görüntü üzerinde hareketli gibi görünebilir.
- Algoritma, taşıtın gerçek hareketi ile kamera hareketinden kaynaklanan görüntü hareketini ayırt etmelidir.

## Taşıt Kapsamına Giren Nesneler

Aşağıdaki nesne türleri taşıt olarak değerlendirilmelidir:

- Motorlu karayolu taşıtları
- Otomobiller
- Motosikletler
- Otobüsler
- Kamyonlar
- Traktör, ATV ve benzeri kara araçları
- Raylı taşıtlar
- Trenler
- Lokomotifler
- Vagonlar
- Tramvaylar
- Monoraylar
- Füniküler
- Tüm deniz taşıtları

## İnsan Kapsamına Giren Nesneler

- Ayakta duran insanlar
- Oturan insanlar
- Görüntüde kısmen görünen insanlar
- Başka objeler tarafından kısmen kapanmış insanlar

Bisiklet ve motosiklet sürücüleri ayrıca insan olarak sayılmamalıdır.

## UAP ve UAİ Tespiti Kuralları

- Uçan Araba Park Alanı (UAP) ve Uçan Ambulans İniş Alanı (UAİ), 4,5 metre çapında dairelerle belirtilir.
- UAP ve UAİ görselleri temsilidir; yarışmada kullanılacak asıl örnekler veri seti olarak paylaşılacaktır.
- UAP ve UAİ alanları tespit edildikten sonra iniş durumu da bildirilmelidir.
- Alanın inişe uygunluğu, alan üzerinde herhangi bir cisim bulunup bulunmamasına bağlıdır.
- Alan üzerinde insan, taşıt veya başka herhangi bir nesne varsa iniş durumu `uygun değil` olmalıdır.
- Alan üzerinde nesne tespiti yapılan veya nesne tespiti yapılamayan herhangi bir cisim varsa iniş için uygun değildir.
- UAP ve UAİ alanlarının bir kısmının görüntü karesinde olması, alanın nesne olarak tespiti için yeterlidir.
- Ancak iniş durumunun `uygun` olabilmesi için UAP veya UAİ alanının tamamı görüntü karesi içinde bulunmalıdır.
- Alan kısmen görüntü dışındaysa nesne tespit edilmeli ancak iniş durumu `uygun değil` verilmelidir.
- UAP veya UAİ alanlarının üzerinde insan veya taşıt varsa, bu insan ve taşıtlar ayrıca kendi sınıflarıyla tespit edilmelidir.
- Çekim açısı nedeniyle alana yakın cisimler alanın üzerindeymiş gibi görünebilir.
- Bu yanıltıcı durumda iniş durumu yine `uygun değil` olmalıdır.

## Algoritma Çalışma Şartları

- Yarışmacılar sunucuya istek gönderdiğinde bir adet görüntü karesi alacaktır.
- Her görüntü karesinde tespit edilen nesnelerin bilgisi istenen formatta sunucuya gönderilecektir.
- Video kareleri sırayla gönderilecektir.
- Bir kareye sonuç gönderilmeden sıradaki kare alınamayacaktır.
- Tüm görüntü karelerinin toplu olarak indirilmesi mümkün değildir.
- Her görüntü karesi için yalnızca 1 adet sonuç gönderilmelidir.
- Aynı kare için birden çok sonuç gönderilirse ilk gönderilen sonuç değerlendirmeye alınacaktır.
- Bir görüntü karesi için belirlenen limitten fazla tahmin gönderilirse takımın ilgili oturumda sonuç gönderme kabiliyeti geçici olarak engellenebilir.
- Her kare için gönderilen tahmin sayısı takip edilmelidir.

## Yapay Zeka İçin Kritik Uygulama Notları

- Model farklı çözünürlüklerde çalışabilmelidir: Full HD ve 4K.
- Model farklı görüntü formatlarını desteklemelidir.
- Model düşük ışık, gündüz, gece, yağmur, kar ve farklı hava koşullarına dayanıklı olmalıdır.
- Model şehir, orman ve deniz üstü gibi farklı arka planlara genellenebilmelidir.
- Model RGB ve termal görüntülerle çalışabilecek şekilde tasarlanmalıdır.
- Model kısmi görünüm, kapanma, görüntü dışına taşma ve bozuk kare durumlarını dikkate almalıdır.
- Hareket durumu için yalnızca optik akış veya görüntüdeki pozisyon değişimi yeterli olmayabilir; kamera hareketi mutlaka hesaba katılmalıdır.
- UAP ve UAİ iniş uygunluğu değerlendirilirken alanın tamamının kare içinde olup olmadığı ve alan üzerinde herhangi bir cisim bulunup bulunmadığı kontrol edilmelidir.
- Sunucu akışı sıralı olduğu için algoritma gerçek zamanlı veya kare bazlı çalışmaya uygun olmalıdır.
- Toplu veri indirme varsayımı yapılmamalıdır.
