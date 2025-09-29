# Kısa Özet
Python ve YOLOv8 kullanarak gerçek zamanlı nesne tespiti modülü geliştirdim. Canlı kamera/RTSP akışlarını işleyip düşük gecikme ile tespitler üretir; GridFormer ile hava koşullarına bağlı bozulmaları iyileştirir ve ROS 2 ile navigasyon yığınına entegre olur.

# Otonom Robotlar için Perception Modülü: YOLOv8 Nesne Tespiti

Bu proje, otonom robotlarda düşük görüş koşullarında dahi güvenilir algılama yapabilmek için GridFormer tabanlı görüntü iyileştirme ve YOLOv8 tabanlı nesne tespitini bir araya getirir. Amaç; canlı kamera/RTSP akışlarını işleyip düşük gecikme ile tespitler üretmek ve ROS 2 ile navigasyon yığınına entegre olmaktır.

## Özellikler
- Hava koşullarına dayanıklı algı (GridFormer ile iyileştirme)
- YOLOv8 ile gerçek zamanlı nesne tespiti (kutu, palet vb.)
- ONNX Runtime / TensorRT desteği (GPU hızlandırma, CPU geri dönüşüm)
- ROS 2 Nav2 ile entegrasyon (maliyet haritası ve planlama)
- Testler ve örnek betikler ile kolay doğrulama

## Mimari (Özet)
- Girdi: Kamera/RTSP görüntüsü
- Görüntü iyileştirme: GridFormer
- Nesne tespiti: YOLOv8 (ONNX/PyTorch)
- Çıktılar: Tespit görselleştirmesi, ROS 2 mesajları, maliyet haritası

## Sistem Gereksinimleri
- İşletim Sistemi: Ubuntu 20.04+ veya Windows 10+
- Python: 3.8+
- (Opsiyonel) GPU: NVIDIA GTX 1650+ ve güncel sürücüler
- ROS 2: Humble (Linux’ta tavsiye edilir)

## Kurulum
1. Depoyu klonlayın
   ```bash
   git clone <repo-url>
   cd staj-2-
   ```
2. Python bağımlılıklarını kurun
   ```bash
   pip install -r requirements.txt
   ```
3. (Linux/ROS 2) Gerekli paketler
   ```bash
   sudo apt install ros-humble-nav2-bringup ros-humble-rviz2
   ```

## Modelleri İndirme
Ağır model dosyaları depoya dahil edilmez. Klasörleri hazırlamak ve isterseniz doğrudan URL’den indirmek için:
```bash
python scripts/download_models.py \
  --yolo-url https://…/yolov8s.onnx \
  --gridformer-url https://…/gridformer.onnx
```
Veya Ultralytics ile otomatik indirme yapıp dosyayı `models/yolo/` altına taşıyabilirsiniz.

## Hızlı Başlangıç
- Basit doğrulama ve hızlı boru hattı testi:
  ```bash
  python scripts/quick_pipeline_test.py
  ```
- Performans/GPU durumu kontrolü:
  ```bash
  python scripts/check_gpu_providers.py
  python scripts/simple_monitor.py
  ```

## ROS 2 Demo (Linux)
1. Derleme ve ortamı yükleme
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```
2. Demo başlatma
   ```bash
   ros2 launch launch/warehouse_demo.launch.py \
     enable_gridformer:=true \
     enable_yolo:=true \
     enable_nav2:=true \
     enable_rviz:=true
   ```

## Windows için Demo
- Sunum/demoyu üretmek için PowerShell betiği:
  ```powershell
  .\scripts\run_presentation.ps1
  ```
- Hızlı test:
  ```powershell
  .\scripts\run_quick_test.ps1
  ```

## Testler
- Tüm testleri çalıştırma:
  ```bash
  python scripts/run_tests.py --type all
  ```
- Entegrasyon testi (boru hattı):
  ```bash
  python scripts/test_pipeline.py
  ```

## Model Eğitimi ve Dışa Aktarım
- Örnek eğitim betikleri:
  ```bash
  python train_gridformer.py
  python train_yolo.py
  ```
- Optimizasyon ve dışa aktarma:
  ```bash
  python scripts/optimize_models.py
  python scripts/export_models.py
  ```

## Proje Yapısı
```
staj-2-/
├── models/                 # Ağır model dosyaları (izlenmez)
├── data/                   # Veri kümeleri / sentetik veri (izlenmez)
├── perception/             # GridFormer & YOLO ROS 2 düğümleri
├── navigation/             # Maliyet haritası ve navigasyon düğümleri
├── launch/                 # ROS 2 launch dosyaları
├── scripts/                # Yardımcı ve demo betikleri
├── tests/                  # Testler
├── README.md               # Bu dosya
└── requirements.txt        # Python bağımlılıkları
```

## Notlar
- `.gitignore` dosyası; modelleri, veri kümelerini, geçici dosyaları ve büyük medya dosyalarını hariç tutar. Depo boyutunu küçük tutmak için modelleri `scripts/download_models.py` ile indirip `models/` klasörüne yerleştirin.
- ONNX Runtime ve TensorRT ile GPU hızlandırma desteklenir; GPU bulunmadığında CPU ile çalışır.

## Lisans
MIT Lisansı (bkz. `LICENSE`).
