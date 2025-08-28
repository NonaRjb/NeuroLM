#!/usr/bin/env python3
import argparse
from pathlib import Path
import webdataset as wds
from tqdm import tqdm
import hashlib

# --------- helpers ---------
EXTMAP = {
    ".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".bmp": "bmp", ".tiff": "tiff",
    ".npy": "npy", ".npz": "npz",
    ".json": "json", ".jsonl": "jsonl", ".txt": "txt", ".csv": "csv", ".tsv": "tsv",
    ".pkl": "pkl", ".pt": "pt", ".pth": "pth",
    ".edf": "edf", ".fif": "fif", ".mat": "mat", ".h5": "h5",
}
def field_for(path: Path) -> str:
    return EXTMAP.get(path.suffix.lower(), path.suffix.lower().lstrip("."))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

# --------- main ---------
def build_shards(
    root: Path,
    out: Path,
    top_dirs=None,
    max_bytes=5_000_000_000,   # ~5 GB
    max_count=20_000,          # or cap by sample count
    group_by_stem=False,
    follow_symlinks=False,
):
    out.mkdir(parents=True, exist_ok=True)

    # autodetect top-level dirs if not provided
    if top_dirs is None:
        top_dirs = [p.name for p in sorted(root.iterdir()) if p.is_dir()]

    for top in top_dirs:
        src = root / top
        if not src.exists():
            print(f"[warn] skip missing: {src}")
            continue

        # One shard-stream per top dir (e.g., captions, images, processed)
        pattern = str(out / f"{root.name}-{top}-%06d.tar")
        wrote = 0

        # Collect files deterministically
        files = [p for p in (src.rglob("*") if follow_symlinks else src.glob("**/*")) if p.is_file()]
        files = sorted(files)

        print(f"Sharding '{top}' from {src} → {pattern}")
        with wds.ShardWriter(pattern, maxsize=max_bytes, maxcount=max_count) as sink:
            if not group_by_stem:
                # Each file is its own sample; key is its relative path without extension
                for p in tqdm(files, unit="file"):
                    field = field_for(p)
                    rel_under_top = p.relative_to(src)              # keep hierarchy under this top
                    key = str(rel_under_top.with_suffix(""))        # e.g. "train/s01/run3/eeg_0001"
                    sample = {"__key__": key, field: p.read_bytes()}
                    sink.write(sample)
                    wrote += 1
            else:
                # Group all files that share the same stem (relative path minus extension)
                groups = {}
                for p in files:
                    rel = p.relative_to(src)
                    stem_key = str(rel.with_suffix(""))
                    groups.setdefault(stem_key, []).append(p)

                for key in tqdm(sorted(groups.keys()), unit="sample"):
                    sample = {"__key__": key}
                    for p in groups[key]:
                        field = field_for(p)
                        sample[field] = p.read_bytes()
                    sink.write(sample)
                    wrote += 1

        print(f"[{top}] wrote {wrote} {'samples' if group_by_stem else 'files'} into shards at {pattern}")

    print("Done. Shards in:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create sharded WebDataset tars while preserving folder hierarchy.")
    ap.add_argument("--root", required=True, type=Path,
                    help="Dataset root (e.g., data/things_eeg_2)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output dir for .tar shards")
    ap.add_argument("--tops", nargs="*", default=None,
                    help="Top-level subfolders to shard (default: autodetect). Example: captions images processed")
    ap.add_argument("--max-bytes", type=int, default=5_000_000_000,
                    help="Max bytes per shard")
    ap.add_argument("--max-count", type=int, default=20_000,
                    help="Max samples per shard")
    ap.add_argument("--group-by-stem", action="store_true",
                    help="Group files with the same relative stem into a single sample")
    ap.add_argument("--follow-symlinks", action="store_true",
                    help="Use rglob and follow symlinks (slower)")
    args = ap.parse_args()

    build_shards(
        root=args.root,
        out=args.out,
        top_dirs=args.tops,
        max_bytes=args.max_bytes,
        max_count=args.max_count,
        group_by_stem=args.group_by_stem,
        follow_symlinks=args.follow_symlinks,
    )
