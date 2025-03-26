## Klasifikasi Kualitas Susu Berdasarkan Lama Penyimpanan Menggunakan Computer Vision

Dataset yang digunakan terdiri dari foto susu di dalam petri dish yang diambil dari atas setiap 6 jam, mulai dari 0 jam hingga 48 jam. Total terdapat 9 kelas susu berdasarkan waktu penyimpanan.

Data asli terdiri dari 6 foto per kelas. Untuk meningkatkan jumlah data, dilakukan augmentasi hingga menghasilkan sekitar 100 foto per kelas. Proses augmentasi dilakukan menggunakan library Augmentor dengan teknik flip, rotate, dan shear. Tidak digunakan augmentasi yang memengaruhi warna atau tekstur gambar agar karakteristik visual tetap terjaga.

Seluruh model dilatih menggunakan dataset yang sama.

| Model              | Best Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|--------------------|------------|------------|----------------|----------|--------------|
| **ResNet18**       | 9          | 0.0123     | 99.88%         | 0.0005   | **100.00%**  |
| **MobileNetV3-Large** | 10      | 0.3179     | 95.00%         | 0.2580   | 99.00%       |
| **EfficientNetB0** | 14         | 0.0523     | 99.50%         | 0.0692   | 99.65%       |
| **YOLOv11-Classification** | 10 | **0.0014** | **100.00%**    | -        | **100.00%**  |

Recap hasil training tiap model

**1. YOLOv11-classification**
<img src="https://github.com/0bry/milk_length/blob/cf0a5e9c3487ed11d106564ce3c6b09f94dd0b6d/YOLOv11-classification/runs/classify/train3/results.png" alt="App Screenshot" width="50%" /><br>

**2. MobileNetV3**
<img src="https://github.com/0bry/milk_length/blob/d5925c69f75fd2955eaaa74c1a333b84689df529/MobileNetV3/train2/accuracy_curve.png" alt="App Screenshot" width="50%" /><img src="https://github.com/0bry/milk_length/blob/d5925c69f75fd2955eaaa74c1a333b84689df529/MobileNetV3/train2/loss_curve.png" alt="App Screenshot" width="50%" /><br>

**3. EfficientNetB0**
<img src="https://github.com/0bry/milk_length/blob/cf0a5e9c3487ed11d106564ce3c6b09f94dd0b6d/EfficientNetB0/Figure_1.png" alt="App Screenshot" width="80%" /><br>

**4. ResNet18**
<img src="https://github.com/0bry/milk_length/blob/cf0a5e9c3487ed11d106564ce3c6b09f94dd0b6d/ResNet18/train4/training_summary.png" alt="App Screenshot" width="80%" /><br>

Sekian dari hasil pengerjaan saya terima kasih banyak 🙏🙏🙏
