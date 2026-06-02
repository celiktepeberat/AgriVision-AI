# 🌍 AgriVision AI (AgroSeg) - Tarımsal Semantik Segmentasyon Paneli

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-brightgreen?logo=qt&logoColor=white)
![Rasterio](https://img.shields.io/badge/Rasterio-GIS-lightgrey)
![TÜBİTAK](https://img.shields.io/badge/TÜBİTAK-2209--A-red)

**AgriVision AI**, çok zamanlı Sentinel-2 uydu görüntüleri ve Derin Öğrenme (Semantik Segmentasyon) modelleri kullanarak tarımsal ürünlerin (Mısır, Fındık, Pirinç/Çeltik, Kavak vb.) otomatik tespitini, haritalanmasını ve akademik doğruluk analizini yapan profesyonel bir masaüstü yazılımıdır. 

Bu yazılım, **TÜBİTAK 2209-A** "Çok Zamanlı Sentinel-2 Görüntüleri ve Semantik Segmentasyon Modelleri ile Tarımsal Ürünlerin ve Kavak Plantasyonlarının Haritalanması" projesi kapsamında geliştirilmiştir.

## ⚠️ ÖNEMLİ: Model ve Veri Dosyaları
GitHub'ın dosya boyutu sınırları (100 MB) nedeniyle, projenin çalışması için gereken **eğitilmiş Yapay Zeka Modelleri (.h5)** ve **Örnek Uydu Görüntüleri (.tif)** bu depoya dahil edilmemiştir. 

Sistemi test etmek için gerekli veri setini ve modelleri aşağıdaki bağlantıdan indirebilirsiniz:
👉 **[Modelleri ve Örnek Verileri İndirmek İçin Tıklayın](https://drive.google.com/drive/folders/1nblVOvMnT-rXZBUYuiPRIdi093Iqn-ND?usp=drive_link)**

*(İndirdiğiniz `.h5` dosyalarını proje klasörü içindeki `modeller` klasörüne kopyalamanız yeterlidir.)*

## ✨ Öne Çıkan Özellikler

* **🧠 Çoklu Derin Öğrenme Modeli Desteği:** U-Net, Attention U-Net, U-Net++ ve U2-Net mimarileri ile anında çıkarım (inference) yeteneği.
* **🗺️ İkili Ekran (Split-Screen) & Senkronize Harita:** Aynı anda RGB, Kızılötesi (False Color), NDVI ve AI Segmentasyon katmanlarını senkronize kaydırma ve yakınlaştırma (zoom) ile kıyaslama.
* **🖱️ Sürükle ve Bırak (Drag & Drop) UX:** Katmanları harita panelleri üzerine sürükleyerek anında görselleştirme.
* **📊 Sınıf Bazlı Akademik Metrikler:** Arazideki sınıf dengesizliklerini önleyen Sınıf Bazlı F1-Skoru, Kullanıcı Doğruluğu (UA), Üretici Doğruluğu (PA) ve Cohen's Kappa hesaplamaları.
* **📈 Canlı İstatistikler & Dinamik Pasta Grafik:** Tespit edilen alanların hektar (Ha) cinsinden hesaplanması ve `PyQtChart` ile interaktif görselleştirilmesi.
* **💾 CBS Entegrasyonu:** Üretilen yapay zeka segmentasyon maskelerini veya NDVI indekslerini orijinal koordinat (CRS) ve transform verileriyle `.tif` formatında dışa aktarma.

## 🚀 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/celiktepeberat/AgriVision-AI.git](https://github.com/celiktepeberat/AgriVision-AI.git)
    cd AgriVision-AI
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install numpy tensorflow opencv-python rasterio scikit-learn PyQt5 PyQtChart
    ```

3.  Uygulamayı başlatın:
    ```bash
    python proje.py
    ```

## 🛠️ Kullanım Rehberi

1.  **STEP 1:** `10 Bantlı TIF Yükle` butonu ile Sentinel-2 (L2A) uydu görüntünüzü yükleyin. Sistem otomatik olarak aynı klasördeki referans maskesini bulup eşleştirecektir.
2.  **STEP 2:** Açılır menüden analiz yapmak istediğiniz yapay zeka modelini seçin.
3.  **STEP 3:** `Modeli Çalıştır` butonuna basarak yapay zekanın pikselleri sınıflandırmasını bekleyin.
4.  **Analiz:** Sağ panelde oluşan harita üzerinden katmanları inceleyin, alt panelden sınıf bazlı doğruluk oranlarını analiz edin ve dilerseniz `Haritayı İndir` butonu ile sonuçları dışa aktarın.

## 🤝 Teşekkür & Bildirim

Bu proje, **Gebze Teknik Üniversitesi (GTÜ) Harita Mühendisliği Bölümü** bünyesinde ve **TÜBİTAK 2209-A** Üniversite Öğrencileri Araştırma Projeleri Destekleme Programı kapsamında gerçekleştirilmiştir.

Projedeki değerli katkıları ve vizyoner yönlendirmeleri için akademik danışmanlarım **Doç. Dr. İsmail Çölkesen**'e ve **Arş. Gör. Muhammed Yusuf Öztürk**'e sonsuz teşekkürlerimi sunarım.

---
*Geliştirici: Berat Çeliktepe - GTÜ Harita Mühendisliği (2026)*
