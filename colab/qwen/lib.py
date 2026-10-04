import gc, os, shutil, threading, time
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForMultimodalLM, TextIteratorStreamer

ROOT = "/content/models"
S = {}
GB = 1024**3

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

def _msg(q):
    return [{"role": "user", "content": [{"type": "text", "text": q}]}]

def invoke(q, n=64):
    p, m = S["proc"], S["model"]
    x = p.apply_chat_template(_msg(q), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(m.device)
    t = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(**x, max_new_tokens=n, do_sample=False)
    dt = time.perf_counter() - t
    new = out[0][x["input_ids"].shape[-1]:]
    print(p.decode(new, skip_special_tokens=True))
    print(f"[{len(new)} token, {dt:.2f}s, {len(new)/dt:.1f} tok/s]")

def stream(q, n=64):
    p, m = S["proc"], S["model"]
    x = p.apply_chat_template(_msg(q), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(m.device)
    st = TextIteratorStreamer(p.tokenizer, skip_prompt=True, skip_special_tokens=True)
    th = threading.Thread(target=m.generate, kwargs=dict(**x, streamer=st, max_new_tokens=n, do_sample=False))
    th.start()
    for piece in st:
        print(piece, end="", flush=True)
    th.join()
    print()

def batch(qs, n=64):
    p, m = S["proc"], S["model"]
    p.tokenizer.padding_side = "left"
    texts = [p.apply_chat_template(_msg(q), add_generation_prompt=True, tokenize=False) for q in qs]
    x = p(text=texts, padding=True, return_tensors="pt").to(m.device)
    t = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(**x, max_new_tokens=n, do_sample=False)
    dt = time.perf_counter() - t
    for q, o in zip(qs, out):
        print("Q:", q)
        print("A:", p.decode(o[x["input_ids"].shape[-1]:], skip_special_tokens=True), "\n")
    print(f"[batch {len(qs)}: {dt:.2f}s]")
