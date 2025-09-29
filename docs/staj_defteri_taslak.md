# Staj Defteri Taslağı — Weather-Adaptive Autonomous Warehouse AGV

Bu dosya, staj defterinizi hızlı ve eksiksiz doldurmanız için hazır bir şablondur. Kısa metinler hazırdır; kendi katkılarınızı/ekran görüntülerinizi ilgili yerlere ekleyin.

## 1) Proje Özeti
Depolama alanında çalışan otonom AGV’nin olumsuz hava koşullarında (sis, yağmur, kar, fırtına) güvenli ve hızlı navigasyon yapabilmesi için GridFormer ile görüntü iyileştirme ve YOLOv8 ile nesne algılama birleştirilmiştir. Algılama çıktıları maliyet haritasına (costmap) dönüştürülerek ROS 2 Nav2 planlayıcıya beslenir. Sistem CPU üzerinde fonksiyonel, GPU ile gerçek-zaman hedeflerine ulaşacak şekilde tasarlanmıştır.

## 2) Amaç ve Kapsam
- Amaç: Görüşün bozulduğu durumlarda bile engelleri doğru algılayıp güvenli rota planlamak.
- Kapsam: Veri üretimi, modellerin eğitimi/optimizasyonu (PyTorch→ONNX→TensorRT), ROS 2 düğümleri, entegrasyon testleri, demo ve performans ölçümleri.

## 3) Kurum/Proje Bilgisi
- Proje: Autonomous Warehouse AGV with Weather-Adaptive Perception
- Dönem: 2025 Yaz Sprint 3-4
- Lisans: MIT

## 4) Kullanılan Teknolojiler
- Python 3.8+, PyTorch 2.x, ONNX, ONNX Runtime, TensorRT (FP16)
- YOLOv8s (Ultralytics), GridFormer
- ROS 2 Humble, Nav2, RViz
- MLflow, Optuna, W&B
- Docker, GitHub Actions, pytest

## 5) Sistem Mimarisi
Giriş kamera akışı → GridFormer (hava koşulu iyileştirme) → YOLOv8 (nesne algılama) → bbox→pointcloud → costmap → Nav2 planlayıcı → AGV navigasyonu. Metrikler MLflow’a loglanır.

Öneri: README’deki mimari diyagramını ve `docs/figures/demo_frames/` örnek karelerini bu bölüme ekleyin.

## 6) Veri ve Modeller
- data/synthetic: 5 hava koşulunda sentetik depo görüntüleri
- data/yolo_dataset: YOLO formatında train/val/test
- models/gridformer, models/yolo, models/optimized: PyTorch, ONNX ve (varsa) TensorRT çıktıları

## 7) Yöntem ve Çalışma Adımları (Haftalık/Günlük Kayıt)

### Hafta 1 — Temel Kurulum ve Hızlı Testler
- Geliştirme ortamı kurulumu (PowerShell `setup_dev_env.ps1` / WSL)
- Hızlı testlerin çalıştırılması (`scripts/run_quick_test.ps1` / `.sh`)
- PyBullet sahne kontrolü (`env.py --render`), FPS ve kamera açıları

### Hafta 2 — GridFormer Entegrasyonu ve Ölçümler
- GridFormer modelinin hazırlanması (ONNX, opsiyonel TensorRT)
- `gridformer_node.py` düğümünün test edilmesi, PSNR ölçümleri
- MLflow ile metrik loglama akışının doğrulanması

### Hafta 3 — YOLOv8 Entegrasyonu ve Boru Hattı
- YOLOv8s’ in bağlanması, sınıf konfigurasyonu
- `gridformer_yolo_pipeline.py` ile Uçtan Uca (E2E) akışın doğrulanması
- Degrade→Restore→Detect ardışık düzen çıktılarının görselleştirilmesi

### Hafta 4 — ROS 2 ve Navigasyon
- `/bbox_cloud` yayını, costmap güncellemeleri ve RViz doğrulaması
- `navigation/src/bbox2costmap_node.cpp` ile costmap entegrasyonu
- 2D Nav Goal ile başarı oranı ölçümü (Nav2)

### Hafta 5 — Optimizasyon ve MLOps
- ONNX→TensorRT (FP16) dönüşümü, giriş çözünürlüğü/çalışma alanı ince ayarı
- `scripts/optimize_models.py`, `mlops/mlflow_utils.py` ile izleme
- GitHub Actions/pytest ile sürekli entegrasyon

### Hafta 6 — Demo, Dokümantasyon ve Son Rötuşlar
- Demo video/GIF üretimi (`scripts/run_presentation.ps1`, `create_demo_gif.py`)
- README/FINAL_SUMMARY güncellemeleri ve çıktıların toplanması
- Staj defteri ve teslim evraklarının hazırlanması

Not: Günlük girdi için her başlığın altına 3–5 madde ekleyin (yapılan iş, karşılaşılan sorun, çözüm, öğrenilenler, kanıt ekran görüntüsü yolu).

## 8) Deneyler ve Sonuçlar (Özet Metrikler)

### Pipeline Bileşenleri (CPU Testleri)
| Bileşen | Durum | Gecikme | Çıktı |
|---|---|---|---|
| GridFormer | Çalışıyor | ~1457ms | `demo_restored.jpg` |
| YOLOv8 | Çalışıyor | ~1046ms | `demo_detected.jpg` |
| Costmap | Çalışıyor | ~30ms | `demo_costmap.jpg` |
| Yol Planlama | Çalışıyor | ~20ms | RViz/NAV2 |
| Uçtan Uca | Çalışıyor | ~2534ms | Tam entegrasyon |

### Hedef vs Gerçek (GPU Odaklı)
- Hedef E2E gecikme: < 350ms (gerçek-zaman)
- Ölçülen/raporlanan (GTX 1650): ~295ms E2E, 6.8 FPS, ~2.8GB VRAM
- CPU ile fonksiyonel fakat yavaş; GPU (TensorRT FP16) ile gerçek-zaman mümkün

Kaynaklar: `README.md` “Results/Performance”, `FINAL_DEMO_RESULTS.md`, `FINAL_SUMMARY.md`, `performance_analysis.md`.

## 9) Test ve Doğrulama
- Birim/entegrasyon testleri: `pytest` (markers: unit, integration, ros, gpu)
- Hızlı koşum: `python scripts/run_tests.py --type quick`
- ROS testleri: `python scripts/run_tests.py --ros --type integration`
- Tam boru hattı: `python scripts/test_pipeline.py`

## 10) Karşılaşılan Sorunlar ve Çözümler
- TensorRT derlemesi: Sürüm/bağımlılık uyumu, FP16 ve çalışma alanı (`--workspace`) ayarları
- ROS 2 varlığı: Windows’ta WSL2 kullanımı ve ortamın source edilmesi
- VRAM sınırı (4GB): Giriş boyutu 384–448 ve FP16 kullanımı

## 11) Sonuç ve Değerlendirme
Sistem mimarisi ve entegrasyonu başarıyla kanıtlanmış, demo çıktıları üretilmiştir. CPU’da fonksiyonel, GPU hızlandırma ile hedeflenen <350ms E2E gecikme düzeyine ulaşılabilir. Dokümantasyon, CI ve MLOps bileşenleri tamamlanmıştır.

## 12) Gelecek Çalışmalar
- Çoklu robot koordinasyonu, gelişmiş hava koşulları (kum fırtınası vb.)
- Gerçek robot platformuna dağıtım ve canlı kamera akışı
- INT8 kuantizasyon ve daha ileri optimizasyonlar

## 13) Ekler (Ekran Görüntüleri/Çıktılar)
- `docs/figures/demo_nav.mp4`, `docs/figures/demo_frames/`
- `demo_original.jpg`, `demo_restored.jpg`, `demo_detected.jpg`, `demo_costmap.jpg`
- `pipeline_*_comparison.jpg`, `yolo_test_*`, `dummy_test_*`

## 14) Yeniden Üretim (Kısa Komutlar)
- Hızlı test: `./scripts/run_quick_test.ps1` (Windows) veya `bash scripts/run_quick_test.sh`
- Demo: `./scripts/run_presentation.ps1` (Windows) veya `bash scripts/run_ros2_demo.sh`
- Testler: `python scripts/run_tests.py --type all`

---

Not: Bu şablonu Word/PDF’e dönüştürmeden önce görselleri ekleyip kendi yorumlarınızla zenginleştirmeniz önerilir.


