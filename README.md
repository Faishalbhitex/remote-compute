# remote-compute

Menjalankan inferensi model kecil di GPU gratis (Colab T4) dari VPS lewat Colab CLI.
VPS hanya mengirim script dan mengambil hasil. Model tidak diunduh ke VPS.

## Setup (VPS Ubuntu 24.04, Python 3.12)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    colab sessions

Perintah pertama memicu login OAuth. Config OAuth ada di
~/.colab-cli-oauth-config.json (di luar repo, jangan di-commit).

## Alur

    colab usage
    colab new -s slm --gpu T4
    colab exec -s slm -f colab/check_gpu.py
    colab exec -s slm -f colab/qwen/lib.py
    colab exec -s slm --timeout 900 <<'PY'
    load("Qwen/Qwen3.5-0.8B")
    PY
    colab exec -s slm --timeout 300 <<'PY'
    invoke("Apa itu API?")
    PY
    colab stop -s slm

Ganti model tanpa restart kernel: panggil purge() lalu load("repo/model-baru").
Selalu tunggu "Session terminated" setelah colab stop.
colab exec punya timeout bawaan 30 detik, jadi pakai --timeout untuk download model.

## Tutorial: unduh model dan proses satu gambar

Aktifkan environment dari direktori proyek terlebih dahulu. `load()` mengunduh
model langsung dari Hugging Face ke `/content/models` pada VM Colab; model tidak
disalin ke VPS. Model publik dapat diunduh tanpa token, tetapi `hf auth whoami`
berguna untuk memastikan login dan menghindari rate limit rendah.

    cd ~/remote-compute
    source .venv/bin/activate
    hf auth whoami
    colab sessions

Buat sesi bernama jelas, lalu cek GPU sebelum mengunduh bobot model.

    colab new -s qwen-nota --gpu T4
    colab exec -s qwen-nota -f colab/check_gpu.py

File pada VPS harus diunggah terlebih dahulu; path VPS tidak dapat diberikan ke
`image=`. Buat direktori tujuan sekali per sesi, kemudian upload. Folder `img/`
untuk input lokal diabaikan Git agar nota tidak ikut ter-push ke GitHub.

    colab exec -s qwen-nota <<'PY'
    import os
    os.makedirs("/content/input", exist_ok=True)
    PY
    colab upload -s qwen-nota img/1000137282.jpg /content/input/1000137282.jpg

Muat helper dan jalankan tes nota. Download awal untuk `Qwen3.5-2B` bisa cukup
lama, jadi gunakan timeout panjang. Download berikutnya dalam sesi yang sama
menggunakan cache `/content/models`.

    colab exec -s qwen-nota -f colab/qwen/lib.py
    colab exec -s qwen-nota --timeout 900 -f colab/qwen/test_nota.py

Atau jalankan sendiri dengan model yang lebih kecil:

    colab exec -s qwen-nota --timeout 900 <<'PY'
    load("Qwen/Qwen3.5-0.8B")
    invoke("Baca total dan daftar barang pada nota", n=256,
           image="/content/input/1000137282.jpg")
    purge()
    PY

Selalu lepaskan VM setelah selesai, bahkan bila inferensi gagal.

    colab stop -s qwen-nota
    colab sessions

Jika upload memberi HTTP 500, pastikan `/content/input` sudah dibuat. Jika
`colab exec` terputus tetapi sesi masih `BUSY`, jangan langsung membuat sesi
baru: periksa `colab status -s qwen-nota` dan `colab log -s qwen-nota -n 40`,
lalu `colab stop -s qwen-nota` sebelum mencoba lagi.

## Bantuan di kernel Qwen

Setelah `colab/qwen/lib.py` dieksekusi, panggil `help()` dari `colab exec` untuk
melihat alur dan semua fungsi. Topik singkat tersedia, misalnya `help("image")`,
`help("stream")`, atau `help("lifecycle")`.

    colab exec -s slm <<'PY'
    help()
    PY

## Gambar + teks

`invoke` dan `stream` menerima argumen `image` berupa path gambar **di VM
Colab**, sedangkan `batch` menerima `images`, sebuah list path sejajar dengan
list pertanyaan. Model yang dimuat harus vision-language (misalnya
`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3-VL-2B-Instruct`, atau
`Qwen/Qwen2.5-VL-3B-Instruct`).

    invoke("Jelaskan isi gambar ini", image="/content/input/foto.jpg")
    stream("Baca teks pada nota", image="/content/input/nota.png")

Path VPS atau komputer lokal tidak dapat diberikan langsung: file perlu berada
lebih dahulu di Colab. Untuk file sesekali, paling sederhana unggah dari VPS
ke `/content/input/` dengan `colab upload`. Untuk kumpulan file yang akan
dipakai berulang, mount Google Drive secara interaktif dengan
`colab drivemount -s slm`, lalu gunakan path `/content/drive/...`. Jangan
menyimpan gambar input di `/content/models`, karena `purge()` dapat
membersihkan direktori itu.

## Struktur

- colab/check_gpu.py : tes GPU dan PyTorch
- colab/qwen/lib.py  : download, load, purge, invoke, stream, batch, usage

## Catatan Termux

Colab CLI gagal dipasang di Termux (Android): salah satu dependensinya (rpds-py)
tidak punya wheel siap pakai untuk Android, jadi pip mencoba membangunnya dari
sumber dengan Rust/maturin dan gagal. Di Linux x86_64 (VPS) wheel siap pakai
tersedia, jadi pemasangan langsung berhasil. Termux tetap berguna untuk
mengedit dan push ke git, sedangkan CLI dijalankan di VPS.

## Rencana

Qwen3.5 (0.8B, 2B) sudah dites. Berikutnya Gemma.
