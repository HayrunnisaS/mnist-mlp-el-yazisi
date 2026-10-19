# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

## 1. Projenin Amacı

Bu projenin amacı, MNIST el yazısı rakam veri setini kullanarak Multi-Layer Perceptron (MLP) tabanlı bir yapay sinir ağı geliştirmek ve eğitilen modeli kendi el yazımla hazırladığım rakam görselleri üzerinde test etmektir.

## 2. Kullanılan Teknolojiler

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- OpenCV

## 3. Model Mimarisi

Model, 28 × 28 boyutundaki görüntüleri giriş olarak alır. Görüntüler düzleştirilerek 784 giriş değerine dönüştürülür.

- Giriş: 784 özellik
- Birinci gizli katman: 256 nöron, ReLU
- İkinci gizli katman: 128 nöron, ReLU
- Çıkış: 10 nöron (0–9 rakamları)

Bu projede CNN yerine MLP mimarisi kullanılmıştır.

## 4. Veri Seti ve Eğitim

MNIST veri seti eğitim, doğrulama ve test kümelerine ayrılmıştır.

- Eğitim kümesi: 55.000 görüntü
- Doğrulama kümesi: 5.000 görüntü
- Test kümesi: 10.000 görüntü

Model 10 epoch boyunca eğitilmiştir. Eğitim sırasında Adam optimizasyon algoritması kullanılmıştır.

## 5. Deney Sonuçları

Bir eğitim çalışmasında elde edilen sonuçlar:

- Test doğruluğu: %97,77
- Test kaybı: 0,0862
- Eğitim süresi: yaklaşık 89,21 saniye

Sonuçlar kullanılan model ve deney çalışmasına aittir. Farklı çalıştırmalarda küçük farklılıklar oluşabilir.

## 6. Kendi El Yazımla Yaptığım Test

Eğitilen model, `el_yazisi` klasöründeki kendi el yazımla hazırladığım öğrenci numarası görseli üzerinde test edilmiştir.

Bir test çalışmasında 11 görüntü parçası tespit edilmiş ve modelin tahminleri gerçek numarayla tamamen eşleşmemiştir. Bu sonuç, MNIST test kümesindeki yüksek doğruluğun gerçek el yazısı görsellerinde aynı başarıyı garanti etmediğini göstermektedir.

Görseldeki rakamların ayrılması, boyutlandırılması, renk düzeni ve yazım biçimi tahmin sonuçlarını etkileyebilir.

## 7. Genel Değerlendirme

Deney sonucunda MLP modelinin MNIST test kümesindeki rakamları yüksek doğrulukla sınıflandırabildiği görülmüştür. Ancak kendi el yazımla hazırladığım görselde bazı rakamlar yanlış tahmin edilmiştir.

Bunun olası nedenleri, el yazısının MNIST örneklerinden farklı olması, görseldeki rakamların ayrıştırılması ve görüntü ön işleme adımlarıdır. Gelecek çalışmalarda görüntü ön işleme yöntemleri iyileştirilerek ve farklı el yazısı örnekleri kullanılarak sonuçlar geliştirilebilir.

## 8. Projeyi Çalıştırma

Gerekli kütüphaneleri kurun:

```bash
pip install -r requirements.txt
```

Modeli eğitmek için:

```bash
python main.py
```

El yazısı testini çalıştırmak için:

```bash
python test_el_yazim.py
```

Not: Test kodunun kullandığı görsel dosyasının `el_yazisi` klasöründe bulunduğundan emin olun.
