"""Download and extract the ARC dataset (Easy/Challenge questions + ARC Corpus).

Source: AI2's official release, referenced from the ARC-Solvers repo's
download_data.sh (https://github.com/allenai/ARC-Solvers). This single zip
contains both the question sets and the retrieval corpus.
"""
import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ARC_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.rename(dest)


def extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in tqdm(names, desc=f"extracting {zip_path.name}"):
            zf.extract(name, dest_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Directory to download/extract into")
    parser.add_argument("--keep-zip", action="store_true", help="Don't delete the zip after extracting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zip_path = data_dir / "ARC-V1-Feb2018.zip"

    print(f"Downloading {ARC_URL}")
    download(ARC_URL, zip_path)

    print(f"Extracting to {data_dir}")
    extract(zip_path, data_dir)

    if not args.keep_zip:
        zip_path.unlink()

    print("Done. Contents:")
    for p in sorted(data_dir.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(data_dir))


if __name__ == "__main__":
    main()
