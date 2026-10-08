"""Uji image understanding Qwen pada satu nota yang telah diunggah ke Colab."""

IMAGE = "/content/input/1000137282.jpg"
MODEL = "Qwen/Qwen3.5-2B"

if not os.path.isfile(IMAGE):
    raise FileNotFoundError(
        f"{IMAGE} belum ada. Unggah dulu dengan colab upload sebelum menjalankan tes."
    )

help("image")
load(MODEL)
try:
    invoke(
        """Baca nota ini dengan teliti. Jawab dalam Bahasa Indonesia dengan format:
penjual, tanggal, alamat, daftar barang (qty | nama | harga satuan | jumlah),
dan total. Jika teks atau angka tidak terbaca, tulis 'tidak terbaca'; jangan
menebak.""",
        n=320,
        image=IMAGE,
    )
finally:
    purge(MODEL)
