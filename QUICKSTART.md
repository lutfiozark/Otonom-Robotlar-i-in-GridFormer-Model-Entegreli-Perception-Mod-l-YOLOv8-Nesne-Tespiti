# GridFormer Robot - Hızlı Başlangıç Rehberi

> **Windows kullanıcıları için özel uyarlamalar içerir**

## 🚀 1. Hızlı Kurulum (5 dakika)

### Windows (Native)
```powershell
# PowerShell'i yönetici olarak açın
.\setup_dev_env.ps1

# Test edin
.\scripts\run_quick_test.ps1
```

### WSL2/Linux
```bash
# Geliştirme ortamını kurun
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Test edin
bash scripts/run_quick_test.sh
```

## 🎯 2. Adım Adım Test Süreci

### Adım 1: Sağlık Kontrolü ✅
```bash
# Windows
.\scripts\run_quick_test.ps1

# Linux/WSL
bash scripts/run_quick_test.sh
```
**Beklenen Çıktı**: Tüm testler ✅ olmalı

### Adım 2: PyBullet Ortamını Test Et 🎮
```bash
python env.py --render
```
**Kontrol Listesi**:
- [x] Masa görünüyor mu?
- [x] 3 renkli küp (kırmızı, yeşil, mavi) var mı?
- [x] Kamera açısı uygun mu?
- [x] FPS > 20 mi? (değilse pencereyi küçültün)

### Adım 3: GridFormer Model Hazırlığı 🤖
```bash
# Önce models klasörünü oluşturun
mkdir models

# GridFormer modelinizi models/gridformer.onnx olarak yerleştirin
# Sonra TensorRT'ye dönüştürün:

# Linux/WSL
bash scripts/onnx_to_trt.sh models/gridformer.onnx

# Windows (WSL gerekli)
wsl bash scripts/onnx_to_trt.sh models/gridformer.onnx
```

**GTX 1650 İçin Optimizasyonlar**:
- `--fp16` zaten aktif
- VRAM yetersiz derse: `--workspace 2048` kullanın
- Model boyutu > 1GB ise normal

### Adım 4: ROS 2 Workspace (WSL/Linux gerekli) 🤖
```bash
# ROS 2 Humble kurulu olmalı
source /opt/ros/humble/setup.bash

# Workspace'i derleyin
colcon build --symlink-install
source install/setup.bash

# Launch dosyasını test edin
ros2 launch launch/warehouse_demo.launch.py
```

### Adım 5: Docker Alternatifi 🐳
```bash
# GPU destekli build
docker compose build

# Sistemi başlatın
docker compose up

# Veya sadece ana servis
docker compose up gridformer-robot
```

## 🔧 Yaygın Sorunlar ve Çözümler

| Sorun | Çözüm |
|-------|-------|
| `PyBullet import error` | `pip install pybullet` |
| `CUDA not found` | NVIDIA sürücüleri güncelleyin |
| `TensorRT not found` | Docker kullanın veya TensorRT yükleyin |
| `ROS 2 not found` | WSL2 + ROS 2 Humble kurun |
| `Permission denied (scripts)` | Windows'ta PowerShell execution policy ayarlayın |

## 📊 Sprint Takip Sistemi

### Hafta 1 - Temel Sistem ✅
- [x] PyBullet ortamı çalışıyor
- [x] Kamera görüntüsü alınıyor
- [x] FPS benchmark yapılıyor
- [ ] GridFormer TRT engine oluşturuluyor

### Hafta 2 - Görüntü İşleme
- [ ] GridFormer node çalışıyor
- [ ] PSNR metrikleri ölçülüyor
- [ ] MLflow logging aktif
- [ ] Restored images kaydediliyor

### Hafta 3 - Nesne Algılama
- [ ] YOLOv8 node bağlı
- [ ] Detection sonuçları görselleştiriliyor
- [ ] mAP metrikleri hesaplanıyor
- [ ] Custom classes tanımlanıyor

### Hafta 4 - Navigasyon
- [ ] bbox2costmap çalışıyor
- [ ] Nav2 ile entegrasyon
- [ ] RViz'de rota planlama
- [ ] TF transforms doğru

### Hafta 5 - RL (Opsiyonel)
- [ ] PPO agent eğitimi
- [ ] Gymnasium environment
- [ ] Policy optimization
- [ ] Success rate tracking

### Hafta 6 - Finalizasyon
- [ ] Docker production ready
- [ ] Benchmark raporu
- [ ] Demo video/GIF
- [ ] Staj dokumentasyonu

## 📈 Performans Hedefleri (GTX 1650)

| Metrik | Hedef | Mevcut |
|--------|-------|--------|
| PyBullet FPS | >30 | Test edilecek |
| GridFormer Latency | <100ms | Test edilecek |
| YOLO FPS | >15 | Test edilecek |
| End-to-End Latency | <200ms | Test edilecek |

## 🆘 Yardım Alın

1. **Hızlı Test**: Her değişiklikten sonra `run_quick_test` çalıştırın
2. **Verbose Logging**: Sorun tespit için `python run_test.py --quick`
3. **GPU Monitoring**: `nvidia-smi` ile VRAM kullanımını takip edin
4. **Docker Logs**: `docker compose logs -f` ile runtime logları görün

---

**💡 İpucu**: İlk kez çalıştırıyorsanız Adım 1'den başlayın ve her adımı tek tek doğrulayın. Bir önceki adım çalışmadan bir sonrakine geçmeyin! 