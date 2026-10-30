"""Download the Noisy AudioMNIST dataset (Kaggle) into data/.

The dataset is a single ~2.4 GB zip on Kaggle. Two modes are supported, both without a Kaggle account
or API token (public dataset download link):

    python scripts/download_data.py              # full dataset (~3.7 GB unpacked)
    python scripts/download_data.py --sample 300  # 300 noisy/clean pairs from 6 speakers (~40 MB)

The sample mode reads only the needed files from the remote zip with HTTP range requests, about one file
per second. It is meant for a quick smoke test of the notebook (QUICK_RUN=1), not for reproducing the
reported results. Use at least 180 pairs so that the test split fills the batches the notebook plots;
the smallest possible sample is 60 pairs (one per speaker and digit).

Result layout (what the notebook expects):

    data/noisy-audio-mnist/
    ├── audioMNIST_meta.txt
    └── <speaker>/<digit>_<speaker>_<index>.wav and <digit>_<speaker>_<index>_noisy.wav
"""
import argparse
import collections
import io
import random
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/mahdiahmadi81/noisy-audio-mnist"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "noisy-audio-mnist"
META_FILE = "audioMNIST_meta.txt"
SAMPLE_SPEAKERS = 6


class HttpRangeFile(io.RawIOBase):
    """Seekable, read-only view of a remote file using HTTP range requests."""

    def __init__(self, url):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()  # signed storage URL after the redirect
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def safe_extract(zf, name, out_dir):
    target = (out_dir / name).resolve()
    if out_dir.resolve() not in target.parents:
        raise RuntimeError(f"refusing to extract unsafe path: {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(name) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)


def choose_sample(names, n_pairs, seed):
    """Pick about n_pairs noisy/clean pairs, balanced over digits, from a few random speakers."""
    available = set(names)
    per_speaker_digit = collections.defaultdict(list)
    for name in names:
        if name.endswith(".wav") and "_noisy" not in name and name.replace(".wav", "_noisy.wav") in available:
            speaker, filename = name.split("/")
            per_speaker_digit[speaker, filename.split("_")[0]].append(name)

    rng = random.Random(seed)
    speakers = rng.sample(sorted({speaker for speaker, _ in per_speaker_digit}), SAMPLE_SPEAKERS)
    per_group = max(1, n_pairs // (SAMPLE_SPEAKERS * 10))
    pairs = []
    for (speaker, _), group in sorted(per_speaker_digit.items()):
        if speaker in speakers:
            pairs += rng.sample(sorted(group), min(per_group, len(group)))
    return pairs


def download_sample(out_dir, n_pairs, seed):
    # A small read buffer keeps each request close to the size of one recording (~40 KB).
    zf = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 16))
    safe_extract(zf, META_FILE, out_dir)
    pairs = choose_sample(zf.namelist(), n_pairs, seed)
    for k, clean in enumerate(pairs, 1):
        safe_extract(zf, clean, out_dir)
        safe_extract(zf, clean.replace(".wav", "_noisy.wav"), out_dir)
        print(f"\r[{k}/{len(pairs)}] {clean}", end="", flush=True)
    print(f"\nSample ready: {len(pairs)} pairs in {out_dir}")


def download_full(out_dir):
    zip_path = out_dir.parent / "noisy-audio-mnist.zip"
    print(f"Downloading full dataset (~2.4 GB) to {zip_path} ...")
    with urllib.request.urlopen(URL) as r, open(zip_path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print("\nExtracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith("/"):
                safe_extract(zf, name, out_dir)
    zip_path.unlink()
    print(f"Dataset ready in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, metavar="N", help="download about N noisy/clean pairs only")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the sample selection")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.sample:
            download_sample(args.out, args.sample, args.seed)
        else:
            download_full(args.out)
    except urllib.error.HTTPError as err:
        sys.exit(f"Download failed ({err}). Download the zip manually from "
                 "https://www.kaggle.com/datasets/mahdiahmadi81/noisy-audio-mnist "
                 f"and extract it into {args.out}")


if __name__ == "__main__":
    main()
