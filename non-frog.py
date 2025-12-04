import os
import hashlib
from PIL import Image
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# ==============================
# USER CONFIG
# ==============================
NONFROG_ROOT = "/Volumes/HDB"   # external drive root
TARGET_FOLDERS = [
    "WB_05_19_25",
    "WB_05_22_25",
    "WB_05_25_25",
    "WB_05_28_25",
    "WB_05_31_25"
]

OUTPUT_LIST = "selected_nonfrog_images.txt"
HASH_SIZE = 16
MAX_PER_CLUSTER = 60
MAX_TOTAL_IMAGES = 15000
SCAN_STEP = 20   # ✅ every 20th image
# ==============================

def list_images_limited(root_folder):
    image_paths = []
    for folder in TARGET_FOLDERS:
        target_path = os.path.join(root_folder, folder)
        if not os.path.exists(target_path):
            continue
        for root, dirs, files in os.walk(target_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, f))
    return image_paths[::SCAN_STEP]

def compute_background_hash(image_path):
    try:
        img = Image.open(image_path).convert("L")
        img = img.resize((HASH_SIZE, HASH_SIZE))
        return (hashlib.md5(img.tobytes()).hexdigest(), image_path)
    except Exception:
        return (None, None)

# ==============================
# MAIN ENTRY POINT (REQUIRED)
# ==============================
if __name__ == "__main__":
    # STEP 1 — Gather candidates
    nonfrog_images = list_images_limited(NONFROG_ROOT)
    print(f"✅ Scanning every {SCAN_STEP}th file")
    print(f"✅ Found {len(nonfrog_images)} candidate images\n")

    # STEP 2 — Parallel hashing
    print(f"🧠 Using {cpu_count()} CPU cores...")
    print("🔍 Computing background hashes...\n")

    with Pool(cpu_count()) as pool:
        hash_results = pool.map(compute_background_hash, nonfrog_images)

    hash_results = [(h, p) for (h, p) in hash_results if h]

    # STEP 3 — Cluster images
    clusters = defaultdict(list)
    for h, path in hash_results:
        clusters[h].append(path)

    print(f"✅ Background clusters detected: {len(clusters)}\n")

    # STEP 4 — Select balanced sample
    selected = []
    for _, imgs in clusters.items():
        selected.extend(imgs[:MAX_PER_CLUSTER])

    selected = selected[:MAX_TOTAL_IMAGES]

    # STEP 5 — Save text list
    with open(OUTPUT_LIST, "w") as f:
        for path in selected:
            f.write(path + "\n")

    print("🎉 DONE!")
    print(f"🚫 Non-frog selected: {len(selected)}")
    print(f"📝 Output written to: {OUTPUT_LIST}")
    print("\n✅ No files modified, moved, or deleted.")
