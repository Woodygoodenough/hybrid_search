# chunked_approx_uniform_150k.py
from datasets import load_dataset
import csv, json, random, time
from datetime import datetime
from settings import N, SEED, CHUNK_FETCH, LOG_EVERY, OUT_CSV, COLUMNS, REPO_ID

def norm(key,v):
    if v is None: return ""
    if key=="categories" and isinstance(v,(list,tuple)): return "; ".join(map(str,v))
    if isinstance(v,(list,dict)): return json.dumps(v, ensure_ascii=False, separators=(",",":"))
    if isinstance(v,str): return v
    return str(v)

def now(): return datetime.now().isoformat(timespec="seconds")

def main():
    print(f"[{now()}] starting… chunked stream shuffle target={N} chunk={CHUNK_FETCH}")
    rng = random.Random(SEED)
    ds = load_dataset(REPO_ID, split="train", streaming=True)
    it = iter(ds)
    n = 0
    t0 = time.time()
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()

        while n < N:
            block = []
            # fetch a block
            for _ in range(min(CHUNK_FETCH, N - n)):
                try:
                    block.append(next(it))
                except StopIteration:
                    break
            if not block:
                break

            rng.shuffle(block)  # local shuffle = good mixing, instant yield

            # write the (shuffled) block
            for row in block:
                w.writerow({c: norm(c, row.get(c, "")) for c in COLUMNS})
                n += 1
                if n % LOG_EVERY == 0:
                    rate = n / max(time.time()-t0, 1e-9)
                    eta = (N - n) / max(rate, 1e-9)
                    print(f"[{now()}] wrote {n}/{N} | {rate:.1f}/s | eta ~ {eta/60:.1f} min", flush=True)
                if n >= N:
                    break

    print(f"[{now()}] done. wrote {n} rows → {OUT_CSV}")

if __name__ == "__main__":
    main()
