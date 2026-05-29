# Object Detection YOLO

Project ini dibuat berdasarkan ketentuan tugas pada PPTX: membuat object detection dengan YOLO menggunakan COCO atau dataset lain, selain people counting.

## Isi Project

- `kamera.py`: script utama object detection.
- `requirements.txt`: daftar library Python yang diperlukan.
- `laporan_template.md`: kerangka laporan akademik formal maksimal 20 halaman.
- `outputs/`: folder hasil video atau gambar deteksi.

## Instalasi

Pastikan Python sudah terpasang, lalu jalankan:

```powershell
pip install -r requirements.txt
```

Saat pertama kali dijalankan, Ultralytics akan mengunduh model `yolov8n.pt` jika file model belum tersedia.

## Menjalankan Kamera

Deteksi semua objek COCO selain `person`:

```powershell
python kamera.py
```

Deteksi kelas tertentu, misalnya botol, laptop, dan kursi:

```powershell
python kamera.py --classes bottle,laptop,chair
```

Jika memakai kamera eksternal:

```powershell
python kamera.py --source 1 --classes bottle,laptop,chair
```

## Menjalankan Video atau Gambar

Video:

```powershell
python kamera.py --source "video_uji.mp4" --classes car,motorcycle,bus
```

Gambar:

```powershell
python kamera.py --source "gambar_uji.jpg" --classes "bottle,cup,cell phone"
```

## Catatan Tugas

Default script ini mengecualikan kelas `person`, sehingga project tidak menjadi people counting. Hasil deteksi otomatis disimpan ke folder `outputs`.
