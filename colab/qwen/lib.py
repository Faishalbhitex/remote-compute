import gc, os, shutil, threading, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForMultimodalLM, TextIteratorStreamer

ROOT = "/content/models"
S = {}
GB = 1024**3

_HELP = """\
Qwen Colab helper

Mulai cepat:
  load("Qwen/Qwen3.5-0.8B")
  invoke("Apa itu API?")
  purge()

Fungsi:
  help([topik])                  tampilkan panduan; topik: lifecycle, image, load,
                                 invoke, stream, batch, purge, usage
  load(repo, dtype=torch.float16) unduh (bila perlu) dan muat satu model ke GPU
  invoke(q, n=64, image=None)    jawab sekali dan cetak metrik kecepatan
  stream(q, n=64, image=None)    cetak jawaban token demi token
  batch(qs, n=64, images=None)   jawab beberapa pertanyaan sekaligus
  usage()                         tampilkan pemakaian VRAM dan disk
  purge([repo])                   lepas model dari VRAM; hapus cache model bila perlu

Gambar:
  image harus berupa path file yang SUDAH ada di VM Colab, misalnya
  invoke("Jelaskan gambar ini", image="/content/input/foto.jpg")
  stream("Baca semua teks", image="/content/input/nota.png")
  batch(["Apa isinya?", "Ringkas"], images=["/content/a.jpg", "/content/b.jpg"])

Siklus aman:
  load(...) -> inferensi (invoke/stream/batch) -> purge() -> colab stop
"""

_TOPICS = {
    "lifecycle": "load(...) -> invoke()/stream()/batch() -> purge() -> colab stop",
    "load": "load(repo, dtype=torch.float16): download repo ke /content/models lalu muat ke GPU.",
    "invoke": "invoke(q, n=64, image=None): jawaban lengkap sekali jalan; n adalah token keluaran maksimum.",
    "stream": "stream(q, n=64, image=None): seperti invoke, tetapi keluaran dicetak bertahap.",
    "batch": "batch(qs, n=64, images=None): qs adalah list pertanyaan; images opsional list path dengan panjang sama.",
    "image": "Path gambar harus berada di VM Colab. Gunakan image=... pada invoke/stream, atau images=[...] pada batch.",
    "purge": "purge() membuang model dari VRAM dan seluruh cache /content/models; purge(repo) hanya menghapus repo itu.",
    "usage": "usage() menampilkan VRAM teralokasi/reserved, disk bebas, dan total cache model.",
}

def help(topic=None):
    """Cetak panduan penggunaan helper Qwen; panggil help('image') untuk topik spesifik."""
    if topic is None:
        print(_HELP)
        return
    topic = str(topic).lower()
    if topic not in _TOPICS:
        print(f"Topik tidak dikenal: {topic}. Pilih: {', '.join(_TOPICS)}")
        return
    print(_TOPICS[topic])

def _path(repo):
    return f"{ROOT}/{repo.replace('/', '__')}"

def usage():
    print(f"VRAM alokasi {torch.cuda.memory_allocated()/GB:.2f} GB | reserved {torch.cuda.memory_reserved()/GB:.2f} GB")
    print(f"Disk bebas {shutil.disk_usage('/content').free/GB:.1f} GB")
    print(os.popen(f"du -sh {ROOT} 2>/dev/null").read().strip() or "(belum ada model di disk)")

def load(repo, dtype=torch.float16):
    path = _path(repo)
    if not os.path.isdir(path):
        snapshot_download(repo, local_dir=path)
    S["proc"] = AutoProcessor.from_pretrained(path)
    S["model"] = AutoModelForMultimodalLM.from_pretrained(path, device_map="auto", dtype=dtype).eval()
    usage()

def purge(repo=None):
    S.pop("model", None)
    S.pop("proc", None)
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(_path(repo) if repo else ROOT, ignore_errors=True)
    usage()

def _ready():
    if "proc" not in S or "model" not in S:
        raise RuntimeError("Belum ada model. Jalankan load('Qwen/Qwen3.5-0.8B') lebih dulu; lihat help('load').")
    return S["proc"], S["model"]

def _msg(q, image=None):
    content = []
    if image is not None:
        image = os.fspath(image)
        if not os.path.isfile(image):
            raise FileNotFoundError(f"File gambar tidak ditemukan di VM Colab: {image}")
        content.append({"type": "image", "path": image})
    content.append({"type": "text", "text": q})
    return [{"role": "user", "content": content}]

def _inputs(q, image=None):
    p, m = _ready()
    x = p.apply_chat_template(
        _msg(q, image), add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(m.device)
    return p, m, x

def invoke(q, n=64, image=None):
    p, m, x = _inputs(q, image)
    t = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(**x, max_new_tokens=n, do_sample=False)
    dt = time.perf_counter() - t
    new = out[0][x["input_ids"].shape[-1]:]
    print(p.decode(new, skip_special_tokens=True))
    print(f"[{len(new)} token, {dt:.2f}s, {len(new)/dt:.1f} tok/s]")

def stream(q, n=64, image=None):
    p, m, x = _inputs(q, image)
    st = TextIteratorStreamer(p.tokenizer, skip_prompt=True, skip_special_tokens=True)
    th = threading.Thread(target=m.generate, kwargs=dict(**x, streamer=st, max_new_tokens=n, do_sample=False))
    th.start()
    for piece in st:
        print(piece, end="", flush=True)
    th.join()
    print()

def batch(qs, n=64, images=None):
    p, m = _ready()
    if images is None:
        images = [None] * len(qs)
    if len(images) != len(qs):
        raise ValueError("images harus None atau memiliki panjang yang sama dengan qs")
    p.tokenizer.padding_side = "left"
    messages = [_msg(q, image) for q, image in zip(qs, images)]
    x = p.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        padding=True, return_dict=True, return_tensors="pt",
    ).to(m.device)
    t = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(**x, max_new_tokens=n, do_sample=False)
    dt = time.perf_counter() - t
    for q, o in zip(qs, out):
        print("Q:", q)
        print("A:", p.decode(o[x["input_ids"].shape[-1]:], skip_special_tokens=True), "\n")
    print(f"[batch {len(qs)}: {dt:.2f}s]")
