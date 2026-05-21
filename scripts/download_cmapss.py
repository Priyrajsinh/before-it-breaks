"""Download NASA CMAPSS FD001 raw files and verify SHA-256 checksums.

Usage (local or CI):
    python scripts/download_cmapss.py

Skips files that already exist and pass checksum verification.
Set env var CMAPSS_MIRROR_URL to override the default download base URL.

The dataset is the NASA Turbofan Engine Degradation Simulation Dataset:
  Saxena & Goebel (2008), NASA Ames Prognostics Data Repository.
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/
  pcoe/pcoe-data-set-repository/
"""

import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

# Known SHA-256 digests — not secrets, these are public data-integrity hashes.
# pragma: allowlist secret
EXPECTED = {
    "train_FD001.txt": (  # pragma: allowlist secret
        "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
    ),
    "test_FD001.txt": (  # pragma: allowlist secret
        "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851"
    ),
    "RUL_FD001.txt": (  # pragma: allowlist secret
        "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca"
    ),
}

# Public mirror that serves the original NASA zip (FD001 subset).
# Override with env var CMAPSS_MIRROR_URL if this URL becomes stale.
_DEFAULT_URL = (
    "https://github.com/Priyrajsinh/before-it-breaks"
    "/releases/download/v0.1-data/CMAPSSData-FD001.zip"
)


def _sha256(path: Path) -> str:
    """Return hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _already_ok(dest_dir: Path) -> bool:
    """Return True if all three files exist and match expected checksums."""
    for name, digest in EXPECTED.items():
        p = dest_dir / name
        if not p.exists() or _sha256(p) != digest:
            return False
    return True


def _download_and_extract(url: str, dest_dir: Path) -> None:
    """Download a zip from url and extract FD001 files into dest_dir."""
    print(f"Downloading CMAPSS data from {url} …", flush=True)
    with urlopen(url, timeout=120) as resp:  # nosec B310
        data = resp.read()
    print(f"  {len(data) / 1024:.0f} KB received. Extracting …", flush=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in EXPECTED:
            matches = [m for m in zf.namelist() if m.endswith(name)]
            if not matches:
                raise FileNotFoundError(f"{name} not found in zip from {url}")
            with zf.open(matches[0]) as src, open(dest_dir / name, "wb") as dst:
                dst.write(src.read())


def _verify(dest_dir: Path) -> None:
    """Raise ValueError if any file fails its checksum."""
    for name, expected in EXPECTED.items():
        actual = _sha256(dest_dir / name)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {name}\n"
                f"  expected: {expected}\n"
                f"  got:      {actual}"
            )
        print(f"  {name}: OK ({actual[:12]}…)", flush=True)


def main() -> None:
    """Entry point — skip if files already present and valid."""
    dest_dir = Path("data/raw")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _already_ok(dest_dir):
        print("CMAPSS FD001 files already present and checksums match — skip.")
        return

    url = os.environ.get("CMAPSS_MIRROR_URL", _DEFAULT_URL)
    try:
        _download_and_extract(url, dest_dir)
    except Exception as exc:
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        print(
            "Manual fallback:\n"
            "  1. Download CMAPSSData.zip from the NASA Prognostics Center:\n"
            "     https://www.nasa.gov/.../pcoe-data-set-repository/\n"
            "  2. Extract train_FD001.txt, test_FD001.txt, RUL_FD001.txt"
            " to data/raw/\n"
            "  3. Re-run this script to verify checksums.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Verifying checksums …", flush=True)
    _verify(dest_dir)
    print("All CMAPSS FD001 files downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
