# %%
from datasets import load_dataset
import csv, random, json
from datetime import datetime, date

# ---- Config ----
REPO_ID = "DragonLLM/Clean-Wikipedia-English-Articles"
SAMPLE_SIZE = 150_000
SEED = 6400
OUT_CSV = "wikipedia_sample_150k.csv"

COLUMNS = ["id", "title", "text", "categories", "url", "revdate", "token_count", "entity"]

rng = random.Random(SEED)

def normalize_value(key, val):
    if val is None:
        return ""
    if key == "categories":
        if isinstance(val, (list, tuple)):
            return "; ".join(map(str, val))
        return str(val)
    # ensure dates/datetimes serialize nicely
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    return str(val)

def main():
    # ---- Load (streaming) ----
    # IMPORTANT: select the split; otherwise you get a DatasetDict and iterating yields strings
    ds = load_dataset(REPO_ID, split="train", streaming=True)

    # ---- Reservoir sampling ----
    reservoir = []
    n_seen = 0

    for row in ds:
        # Defensive: skip anything weird
        if not isinstance(row, dict):
            continue

        n_seen += 1
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(row)
        else:
            j = rng.randint(1, n_seen)  # inclusive
            if j <= SAMPLE_SIZE:
                reservoir[j - 1] = row

    # ---- Write CSV ----
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in reservoir:
            out_row = {c: normalize_value(c, r.get(c, "")) for c in COLUMNS}
            writer.writerow(out_row)

    print(f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"Wrote {len(reservoir)} rows to {OUT_CSV} (seen {n_seen} total)")

if __name__ == "__main__":
    main()
