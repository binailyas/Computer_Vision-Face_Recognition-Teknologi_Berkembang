# Face Recognition Real-Time — YuNet + SFace

Sistem **deteksi dan pengenalan wajah secara real-time** menggunakan Python dan OpenCV. Project ini menggunakan **YuNet** untuk mendeteksi wajah serta **SFace** untuk membuat *face embedding* dan mengenali identitas berdasarkan *cosine similarity*.

Project dikembangkan untuk praktikum mata kuliah **Teknologi Berkembang**, Program Studi Sistem Informasi, Institut Teknologi Sepuluh Nopember (ITS).

**Repository:** [Computer_Vision-Face_Recognition-Teknologi_Berkembang](https://github.com/binailyas/Computer_Vision-Face_Recognition-Teknologi_Berkembang)

## Fitur

- **Deteksi wajah real-time** dari webcam menggunakan YuNet.
- **Pengenalan beberapa wajah dalam satu frame** dengan SFace.
- Pengumpulan dataset berdasarkan nama anggota, sesi, dan variasi kondisi.
- Pemeriksaan kualitas saat koleksi: hanya satu wajah, ukuran memadai, dan gambar tidak terlalu buram.
- *Data augmentation* berupa pembalikan horizontal dan penyesuaian pencahayaan.
- Pembentukan database *face embedding* tanpa melatih ulang bobot model SFace.
- Evaluasi **leave-one-condition-out**, *confusion matrix*, serta kalibrasi ambang kemiripan (*threshold*).
- Pengenalan menggunakan **cosine similarity**, **threshold**, dan **margin**.
- *Voting* antar-frame untuk mengurangi perubahan label secara tiba-tiba.
- Label **`Anonymous`** jika wajah tidak memenuhi syarat pengenalan.

## Teknologi

| Komponen | Fungsi |
| --- | --- |
| Python 3 | Bahasa pemrograman utama |
| `opencv-contrib-python` | Pengolahan citra, akses webcam, detektor YuNet, dan pengenal SFace |
| `numpy` | Operasi numerik dan pengolahan embedding |
| **YuNet** (`face_detection_yunet_2023mar.onnx`) | Deteksi wajah dan landmark |
| **SFace** (`face_recognition_sface_2021dec.onnx`) | Penyelarasan wajah dan ekstraksi embedding 128 dimensi |
| Jupyter Notebook / VS Code | Menjalankan notebook per tahap |

> **Catatan:** Implementasi pada `face_recognition_deep.ipynb` menggunakan **YuNet + SFace**, **bukan** Haar Cascade + LBPH. File `haarcascade_frontalface_default.xml`, bila ada di repository, **tidak diperlukan** untuk alur notebook ini.

## Alur kerja

```text
Webcam / Foto
     |
     v
YuNet: deteksi wajah + landmark
     |
     v
SFace: penyelarasan dan ekstraksi embedding
     |
     v
Bandingkan dengan database menggunakan cosine similarity
     |
     v
Cek threshold dan margin kandidat terdekat
     |
     v
Voting identitas antar-frame
     |
     +--> Dikenali  : nama + persentase tampilan + bounding box
     |
     +--> Ditolak   : Anonymous
```

## Persiapan

### 1. Clone repository

```bash
git clone https://github.com/binailyas/Computer_Vision-Face_Recognition-Teknologi_Berkembang.git
cd Computer_Vision-Face_Recognition-Teknologi_Berkembang
```

### 2. Siapkan Python dan library

Pastikan Python 3, `pip`, dan webcam tersedia. Sebaiknya gunakan environment virtual agar dependensi terpisah dari project lain.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy notebook
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy notebook
```

> Instalasi paket OpenCV dan NumPy juga tersedia pada **Cell 1** notebook. Paket `notebook` diperlukan jika menjalankan lewat Jupyter lokal; pengguna VS Code dapat memakai ekstensi Jupyter.

### 3. Buka notebook

```bash
jupyter notebook face_recognition_deep.ipynb
```

Alternatif: buka `face_recognition_deep.ipynb` langsung melalui VS Code, pilih interpreter/kernel Python yang sudah terpasang dependensinya.

**Disarankan menjalankan notebook secara lokal** (Windows/Linux/macOS) karena proses koleksi dan pengenalan menggunakan jendela webcam OpenCV (`cv2.imshow`). Notebook ini tidak dirancang untuk menjalankan akses webcam dengan mekanisme Google Colab.

## Cara menjalankan project (6 tahap)

Jalankan cell secara berurutan dari atas ke bawah.

### Cell 1 — Instalasi dan persiapan model

Instal dependensi, lalu unduh dua model ONNX dari [OpenCV Zoo](https://github.com/opencv/opencv_zoo) ke folder `models/` bila file belum tersedia.

- `models/yunet.onnx` — model deteksi wajah.
- `models/sface.onnx` — model pengenalan wajah.

Proses pengunduhan memerlukan koneksi internet pada penggunaan pertama.

### Cell 2 — Konfigurasi dan fungsi inti

Atur nama anggota serta parameter sistem. Pengaturan awal notebook:

```python
MEMBERS = ["Devin", "Osa", "Abe"]

DATASET_DIR = "dataset_dl"
DB_PATH = "face_db.npz"
CAM_INDEX = 0
DET_SCORE = 0.85
MIN_FACE = 80
TOP_K = 3
MARGIN = 0.03
DEFAULT_THRESHOLD = 0.40
```

Jika webcam tidak terbuka, coba ubah `CAM_INDEX` menjadi `1` atau nomor kamera lain. Nama di `MEMBERS` harus sama persis dengan nama folder dataset masing-masing orang.

### Cell 3 — Pengumpulan dataset

Ambil foto setiap anggota dengan webcam. Contoh pemanggilan:

```python
collect_dataset("Devin", session="rumah")
collect_dataset("Osa", session="rumah")
collect_dataset("Abe", session="rumah")
```

**Penting:** Pada notebook yang disertakan, baris pengumpulan **Devin** aktif secara default, sedangkan **Osa** dan **Abe** masih diberi komentar (`#`). Jalankan **satu orang bergantian** dengan mengaktifkan baris yang sesuai agar kamera tidak memulai tiga sesi berturut-turut tanpa persiapan.

Tiap sesi penuh memiliki **9 kondisi** dan target **200 foto per orang**:

| Kondisi | Target foto |
| --- | ---: |
| Terang | 25 |
| Remang | 25 |
| Backlight | 25 |
| Miring kiri | 20 |
| Miring kanan | 20 |
| Menunduk | 20 |
| Menengadah | 20 |
| Kacamata | 25 |
| Variasi ekspresi | 20 |
| **Total per sesi** | **200** |

Pada jendela koleksi dataset:

- **`Space`** — mulai pengambilan foto pada kondisi yang sedang ditampilkan.
- **`Q`** — hentikan sesi koleksi.

Gambar disimpan otomatis dengan interval minimal **0,25 detik** jika hanya ada satu wajah, lebarnya minimal **80 px**, dan ketajamannya memenuhi nilai batas. Disarankan mengambil sesi tambahan di tempat/waktu berbeda, misalnya `rumah`, `kampus`, dan `luar`.

Contoh hasil penamaan file:

```text
dataset_dl/
├── Devin/
│   ├── Devin_rumah-terang_000.jpg
│   └── Devin_rumah-remang_000.jpg
├── Osa/
│   └── Osa_rumah-terang_000.jpg
└── Abe/
    └── Abe_rumah-terang_000.jpg
```

### Cell 4 — Membuat database wajah

```python
db = build_gallery()
```

Fungsi ini membaca foto pada direktori tiap anggota, menjalankan deteksi serta ekstraksi embedding, dan membuat variasi gambar asli, *mirror*, serta dua tingkat gamma. Hasilnya disimpan sebagai **`face_db.npz`**.

**Minimal dibutuhkan dataset dari dua identitas** agar database dapat dibentuk. Untuk project ini, gunakan ketiga anggota.

> Tahap ini merupakan **pembentukan galeri embedding**, bukan *training* ulang arsitektur atau bobot neural network YuNet/SFace.

### Cell 5 — Evaluasi dan kalibrasi threshold

```python
THRESHOLD = evaluate(db)
```

Evaluasi menggunakan metode **leave-one-condition-out** untuk mengukur kemampuan mengenali kondisi wajah yang disisihkan dari galeri pembanding. Output mencakup:

- Akurasi identifikasi pada data uji dan jumlah *query*.
- *Confusion matrix* identitas asli dan prediksi.
- Statistik skor kemiripan benar dan salah.
- Nilai threshold hasil kalibrasi serta indikator penerimaan/penolakan.

Jika tersedia data wajah orang yang **bukan** anggota, simpan foto dengan persetujuan pemiliknya di `dataset_dl/_outsiders/` agar dapat digunakan saat kalibrasi. Threshold hasil evaluasi disimpan kembali ke `face_db.npz`.

### Cell 6 — Pengenalan wajah real-time

```python
run_realtime()
```

Program membuka webcam, mendeteksi wajah, mencari kecocokan terhadap database, dan menggambar kotak wajah beserta label. Identitas yang tidak memenuhi batas kemiripan atau margin akan diberi label **`Anonymous`**.

**Kontrol saat pengenalan:**

| Tombol | Fungsi |
| --- | --- |
| `Q` | Keluar dari program |
| `+` / `=` | Menaikkan threshold sebesar 0,02 |
| `-` / `_` | Menurunkan threshold sebesar 0,02 |
| `D` | Mengaktifkan/menonaktifkan mode debug |

Penyesuaian threshold melalui tombol berlaku untuk sesi berjalan; kode saat ini **tidak menyimpan otomatis** nilai threshold yang diubah secara manual.

## Contoh hasil

Pada dokumentasi pengujian kelompok, sistem memperlihatkan pengenalan wajah dengan label:

| Identitas | Bounding box | Contoh skor tampilan |
| --- | --- | ---: |
| Abe | Biru | 76% |
| Osa | Merah | 85% |
| Devin | Hijau | 72% |
| Wajah tak dikenali | Abu-abu | `Anonymous` |

**Penting:** Persentase pada jendela webcam adalah **skor kemiripan yang dipetakan ke skala tampilan**, bukan probabilitas identitas benar dan **bukan akurasi model**. Nilai dapat berubah tergantung frame dan threshold yang dipakai.

## Struktur file

Struktur berikut menjelaskan file utama dan artefak yang dibuat saat notebook dijalankan. Folder/file hasil runtime mungkin belum tersedia sebelum tahap terkait dijalankan.

```text
Computer_Vision-Face_Recognition-Teknologi_Berkembang/
├── face_recognition_deep.ipynb  # Notebook utama
├── README.md                    # Dokumentasi project ini
├── models/                      # Dibuat otomatis pada Cell 1
│   ├── yunet.onnx
│   └── sface.onnx
├── dataset_dl/                  # Dibuat saat koleksi dataset
│   ├── Devin/
│   ├── Osa/
│   ├── Abe/
│   └── _outsiders/             # Opsional; dibuat manual jika diperlukan
└── face_db.npz                 # Dibuat pada Cell 4, dikalibrasi pada Cell 5
```

## Troubleshooting

| Kendala | Langkah yang disarankan |
| --- | --- |
| `No module named cv2` | Instal `opencv-contrib-python` di environment/kernel yang aktif. |
| `FaceDetectorYN` / `FaceRecognizerSF` tidak tersedia | Perbarui OpenCV: `python -m pip install -U opencv-contrib-python`. |
| Webcam tidak terbuka | Tutup aplikasi yang menggunakan kamera, periksa izin kamera, atau ubah `CAM_INDEX`. |
| Model gagal diunduh | Periksa koneksi internet dan akses ke OpenCV Zoo; ulangi Cell 1. |
| Database gagal dibuat | Pastikan Cell 3 telah menyimpan dataset minimal dua anggota dan nama folder sesuai `MEMBERS`. |
| Anggota sering dikenali sebagai `Anonymous` | Perbaiki cahaya dan variasi foto; periksa nilai threshold/margin, lalu evaluasi ulang. |
| Orang luar dikenali sebagai anggota | Tambahkan contoh orang luar untuk kalibrasi dan pertimbangkan threshold yang lebih ketat. |
| Label berubah-ubah | Dekatkan wajah ke kamera, hindari gerakan cepat, dan gunakan pencahayaan merata. |

## Privasi dataset

Foto wajah dan *face embedding* merupakan data biometrik. Ambil dan gunakan foto **hanya dengan persetujuan pemiliknya**, simpan secara aman, dan hindari mempublikasikan dataset pribadi ke repository publik.

Contoh isi `.gitignore` yang disarankan:

```gitignore
.venv/
.ipynb_checkpoints/
__pycache__/
dataset_dl/
face_db.npz
```

Model `.onnx` akan diunduh otomatis oleh notebook, sehingga tidak wajib dimasukkan ke GitHub. Jika ingin mengabaikannya juga, tambahkan `models/` ke `.gitignore`.

## Anggota kelompok

| Nama | NRP |
| --- | --- |
| Axellino Berwin Parlindungan Sinaga | 5026241094 |
| Devin Alif Fansyah Yahya | 5026241042 |
| Bina Ilyas Santosa | 5026241074 |

**Mata kuliah:** Teknologi Berkembang  
**Program studi:** Sistem Informasi, Institut Teknologi Sepuluh Nopember  
**Tahun:** 2026

## Referensi

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- [OpenCV Face Detection & Face Recognition dengan YuNet dan SFace](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)

---

*Dokumentasi ini disusun berdasarkan implementasi `face_recognition_deep.ipynb` yang digunakan dalam praktikum.*
