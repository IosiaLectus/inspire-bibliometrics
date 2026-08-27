import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API = "https://inspirehep.net/api/authors"
HERE = os.path.dirname(os.path.abspath(__file__))
ELIGIBLE_PATH = os.path.join(HERE, "eligible_authors_v2.json")
OUTPUT_PATH = os.path.join(HERE, "author_positions_v2.json")
MAX_WORKERS = 16
CHECKPOINT_EVERY = 100


def fetch_positions(bai, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(API, params={"q": f"ids.value:{bai}", "size": 1}, timeout=20)
            r.raise_for_status()
            hits = r.json().get("hits", {}).get("hits", [])
            if not hits:
                return bai, None
            return bai, hits[0]["metadata"].get("positions", [])
        except Exception:
            if attempt == retries - 1:
                return bai, "ERROR"
    return bai, "ERROR"


def main():
    with open(ELIGIBLE_PATH) as f:
        eligible = json.load(f)
    all_bais = [p["bai"] for p in eligible]

    try:
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    needed = [b for b in all_bais if b not in results]
    print(f"{len(needed)} of {len(all_bais)} still needed", flush=True)

    lock = threading.Lock()
    done = [0]

    def checkpoint():
        with lock:
            if done[0] % CHECKPOINT_EVERY == 0:
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(results, f)
                print(f"[{done[0]}/{len(needed)}] checkpoint saved", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_positions, b): b for b in needed}
        for fut in as_completed(futs):
            bai, positions = fut.result()
            with lock:
                results[bai] = positions
                done[0] += 1
            checkpoint()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)
    n_err = sum(1 for v in results.values() if v == "ERROR")
    print(f"DONE: {len(results)} cached ({n_err} errors)", flush=True)


if __name__ == "__main__":
    main()
