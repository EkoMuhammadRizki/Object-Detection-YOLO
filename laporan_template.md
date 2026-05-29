# Laporan Object Detection Menggunakan YOLO

## 1. Pendahuluan

Tuliskan latar belakang penggunaan computer vision dan object detection. Jelaskan alasan memilih objek target selain manusia, misalnya kendaraan, botol, laptop, kursi, atau objek lain yang relevan.

## 2. Tujuan

Tujuan project ini adalah membangun sistem object detection sederhana menggunakan model YOLO untuk mendeteksi objek tertentu dari kamera, video, atau gambar.

## 3. Dasar Teori

Jelaskan secara singkat:

- Object detection.
- Model YOLO.
- Dataset COCO atau dataset lain yang digunakan.
- Bounding box, confidence score, dan class label.

## 4. Alat dan Bahan

- Python.
- Visual Studio Code atau Google Colab.
- Library `ultralytics`.
- Library `opencv-python`.
- Kamera, video, atau gambar uji.
- Model YOLO, misalnya `yolov8n.pt`.

## 5. Metodologi

Langkah pengerjaan:

1. Menyiapkan folder project.
2. Menginstal library yang diperlukan.
3. Memuat model YOLO.
4. Mengambil input dari kamera, video, atau gambar.
5. Melakukan deteksi objek selain `person`.
6. Menampilkan bounding box, label objek, confidence score, dan jumlah objek.
7. Menyimpan hasil deteksi dalam bentuk gambar atau video.

## 6. Implementasi Program

Jelaskan fungsi utama pada `kamera.py`, terutama:

- Pemilihan model YOLO.
- Pemilihan kelas objek target.
- Proses membaca frame.
- Proses deteksi objek.
- Proses membuat anotasi hasil deteksi.
- Proses menyimpan hasil output.

## 7. Hasil dan Pembahasan

Masukkan tangkapan layar atau frame hasil deteksi. Jelaskan objek yang berhasil terdeteksi, jumlah objek, serta kondisi yang memengaruhi akurasi seperti pencahayaan, jarak objek, dan sudut kamera.

## 8. Kesimpulan

Tuliskan kesimpulan dari hasil project. Jelaskan apakah sistem berhasil mendeteksi objek target dan sebutkan keterbatasan yang ditemukan.

## 9. Saran

Tuliskan pengembangan yang dapat dilakukan, misalnya menggunakan model YOLO yang lebih besar, menambah dataset custom, atau meningkatkan kualitas kamera.

## 10. Daftar Pustaka

Cantumkan referensi, misalnya dokumentasi Ultralytics YOLO, OpenCV, dan dataset COCO.
