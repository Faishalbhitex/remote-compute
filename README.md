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
