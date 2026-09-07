#!/usr/bin/env python3
"""
TEP-VOID Centralized Data Downloader
=====================================
Provides reproducible, checksum-verified downloads of all external
datasets used in the TEP-VOID analysis pipeline.

Features:
  - SHA-256 checksum verification for every downloaded file
  - Automatic retry with exponential backoff
  - Skip-if-exists-and-verified (idempotent re-runs)
  - Machine-readable checksum manifest (data/raw/checksums.json)
  - Support for HTTP/HTTPS, VizieR TAP, and Playwright (CDS) downloads
  - Centralized registry of all dataset URLs and metadata

Usage:
    from scripts.utils.data_downloader import DataDownloader
    dl = DataDownloader()
    dl.download_file(url, target_path, expected_sha256=None, description="...")
    dl.download_vizier_table(catalog="J/MNRAS/487/2061", target=...)
    dl.verify_all()
    dl.write_manifest()
"""

import hashlib
import json
import os
import sys
import time
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlretrieve, urlopen, Request
from urllib.error import URLError, HTTPError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataDownloader:
    """Centralized download utility with checksum verification."""

    # Downloads that failed within this cooldown period (seconds) are not
    # retried, preventing wasted time on servers that are down.
    FAILURE_COOLDOWN_SECONDS = 3600  # 1 hour

    def __init__(self, data_raw=None, manifest_path=None):
        self.data_raw = data_raw or (PROJECT_ROOT / "data" / "raw")
        self.data_ext = self.data_raw / "external"
        self.manifest_path = manifest_path or (self.data_raw / "checksums.json")
        self.manifest = self._load_manifest()
        self.results = []

    # ------------------------------------------------------------------ #
    # Manifest management
    # ------------------------------------------------------------------ #

    def _load_manifest(self):
        if self.manifest_path.exists():
            try:
                m = json.loads(self.manifest_path.read_text())
                m.setdefault("failures", {})
                m.setdefault("files", {})
                return m
            except (json.JSONDecodeError, IOError):
                pass
        return {"files": {}, "failures": {}, "generated": None, "pipeline": "TEP-VOID"}

    def _save_manifest(self):
        self.manifest["generated"] = datetime.now(timezone.utc).isoformat()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def _record(self, path, sha256, size, url, source, status, notes=""):
        rel = str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() else str(path)
        entry = {
            "sha256": sha256,
            "size_bytes": size,
            "url": url,
            "source": source,
            "status": status,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        self.manifest["files"][rel] = entry
        self.results.append({"path": rel, "status": status, "sha256": sha256[:16]})

    def _record_failure(self, url, error_msg):
        """Record a download failure so it can be skipped within the cooldown period."""
        self.manifest["failures"][url] = {
            "error": str(error_msg),
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _is_in_failure_cooldown(self, url):
        """Check if a URL failed recently and is still within the cooldown period."""
        entry = self.manifest.get("failures", {}).get(url)
        if not entry:
            return False
        try:
            failed_at = datetime.fromisoformat(entry["failed_at"])
            elapsed = (datetime.now(timezone.utc) - failed_at).total_seconds()
            return elapsed < self.FAILURE_COOLDOWN_SECONDS
        except (KeyError, ValueError, TypeError):
            return False

    def _clear_failure(self, url):
        """Remove a URL from the failure cache (e.g. after a successful download)."""
        self.manifest.get("failures", {}).pop(url, None)

    @staticmethod
    def _is_votable_error(filepath):
        """Check if a VOTable file is actually a VizieR error response."""
        try:
            content = filepath.read_text(errors="replace")
            return 'name="Error"' in content or 'ID="Error"' in content
        except (IOError, OSError):
            return False

    # ------------------------------------------------------------------ #
    # Checksum utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_checksum(self, filepath, expected_sha256):
        if expected_sha256 is None:
            return True
        actual = self.compute_sha256(filepath)
        if actual != expected_sha256:
            print(f"  CHECKSUM MISMATCH: {filepath}")
            print(f"    expected: {expected_sha256}")
            print(f"    actual:   {actual}")
            return False
        return True

    # ------------------------------------------------------------------ #
    # HTTP/HTTPS download
    # ------------------------------------------------------------------ #

    def download_file(self, url, target, expected_sha256=None,
                      description="", source="", max_retries=3,
                      decompress_gz=False, timeout=120, min_size=0,
                      force=False):
        """Download a file with retry, checksum verification, and manifest recording.

        Args:
            url: URL to download
            target: Path to save the file
            expected_sha256: Expected SHA-256 hash (None to skip verification)
            description: Human-readable description
            source: Data source name (e.g. "Pantheon+SH0ES")
            max_retries: Maximum retry attempts
            decompress_gz: If True, decompress .gz after download
            timeout: Request timeout in seconds
            min_size: Minimum expected file size in bytes; files smaller than
                      this are treated as invalid and re-downloaded
            force: If True, bypass skip logic and always download

        Returns:
            True if download succeeded and checksum verified, False otherwise.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Skip if file exists, is large enough, and checksum matches
        if not force and target.exists() and target.stat().st_size >= max(min_size, 1):
            actual_sha = self.compute_sha256(target)
            if expected_sha256 is None or actual_sha == expected_sha256:
                desc = f" ({description})" if description else ""
                print(f"  [SKIP] {target.name}{desc} — already present "
                      f"({target.stat().st_size:,} bytes)")
                self._record(target, actual_sha, target.stat().st_size,
                             url, source, "skip_exists")
                return True
            else:
                print(f"  [REPLACE] {target.name} — checksum mismatch, re-downloading")
        elif not force and target.exists() and target.stat().st_size < max(min_size, 1):
            print(f"  [REPLACE] {target.name} — file too small "
                  f"({target.stat().st_size} < {min_size} bytes), re-downloading")

        # Skip if this URL failed recently (within cooldown period)
        if not force and self._is_in_failure_cooldown(url):
            print(f"  [SKIP-FAIL] {target.name} — server failed recently, "
                  f"skipping for {self.FAILURE_COOLDOWN_SECONDS // 60} min cooldown")
            self._record(target, "", 0, url, source, "skip_failure_cooldown",
                         "Skipped due to recent failure")
            return False

        # Download with retries
        for attempt in range(1, max_retries + 1):
            try:
                print(f"  [DOWNLOAD] {target.name} — attempt {attempt}/{max_retries}")
                req = Request(url, headers={"User-Agent": "TEP-VOID-Pipeline/1.0"})
                with urlopen(req, timeout=timeout) as resp:
                    with open(target, "wb") as f:
                        shutil.copyfileobj(resp, f)

                size = target.stat().st_size
                if size == 0:
                    raise IOError("Downloaded file is empty")
                if min_size and size < min_size:
                    raise IOError(f"Downloaded file too small ({size} < {min_size} bytes)")

                # Decompress if requested
                if decompress_gz and target.suffix == ".gz":
                    decompressed = target.with_suffix("")
                    with gzip.open(target, "rb") as gz:
                        with open(decompressed, "wb") as out:
                            shutil.copyfileobj(gz, out)
                    print(f"  [DECOMP] {decompressed.name}")

                # Verify checksum
                actual_sha = self.compute_sha256(target)
                if expected_sha256 and not self.verify_checksum(target, expected_sha256):
                    self._record(target, actual_sha, size, url, source, "checksum_mismatch")
                    return False

                self._clear_failure(url)
                self._record(target, actual_sha, size, url, source, "downloaded")
                desc = f" ({description})" if description else ""
                print(f"  [OK] {target.name}{desc} — {size:,} bytes")
                return True

            except (URLError, HTTPError, IOError, OSError) as e:
                wait = 2 ** (attempt - 1) * 5
                print(f"  [RETRY] {target.name} — {e}, waiting {wait}s")
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    print(f"  [FAIL] {target.name} — {e}")
                    self._record_failure(url, e)
                    self._record(target, "", 0, url, source, "failed", str(e))
                    return False

        return False

    # ------------------------------------------------------------------ #
    # VizieR download (via TAP/VOTable)
    # ------------------------------------------------------------------ #

    def download_vizier(self, catalog, target, description="",
                        source="", max_retries=3, timeout=60, min_size=0,
                        force=False):
        """Download a VizieR catalog table as VOTable XML.

        Args:
            catalog: VizieR catalog ID (e.g. "J/MNRAS/487/2061")
            target: Path to save the VOTable XML
            description: Human-readable description
            source: Data source name
            max_retries: Maximum retry attempts
            timeout: Request timeout
            min_size: Minimum expected file size in bytes; smaller files are
                      treated as invalid (e.g. VizieR error responses)
            force: If True, bypass skip logic and always download

        Returns:
            True if download succeeded, False otherwise.
        """
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        # VizieR VOTable endpoint
        url = f"http://vizier.u-strasbg.fr/viz-bin/votable?-source={catalog}&-out.all=1&-out.max=999999"

        # Skip if file exists, is large enough, and is not a VOTable error response
        if not force and target.exists() and target.stat().st_size >= max(min_size, 1):
            if self._is_votable_error(target):
                print(f"  [REPLACE] {target.name} — existing file is a VizieR error response")
            else:
                actual_sha = self.compute_sha256(target)
                print(f"  [SKIP] {target.name} ({description}) — already present "
                      f"({target.stat().st_size:,} bytes)")
                self._record(target, actual_sha, target.stat().st_size,
                             url, source, "skip_exists")
                return True
        elif not force and target.exists() and target.stat().st_size < max(min_size, 1):
            print(f"  [REPLACE] {target.name} — file too small "
                  f"({target.stat().st_size} < {min_size} bytes), re-downloading")

        # Skip if this URL failed recently (within cooldown period)
        if not force and self._is_in_failure_cooldown(url):
            print(f"  [SKIP-FAIL] {target.name} — VizieR failed recently, "
                  f"skipping for {self.FAILURE_COOLDOWN_SECONDS // 60} min cooldown")
            self._record(target, "", 0, url, source, "skip_failure_cooldown",
                         "Skipped due to recent failure")
            return False

        for attempt in range(1, max_retries + 1):
            try:
                print(f"  [VIZIER] {catalog} -> {target.name} — attempt {attempt}/{max_retries}")
                req = Request(url, headers={"User-Agent": "TEP-VOID-Pipeline/1.0"})
                with urlopen(req, timeout=timeout) as resp:
                    content = resp.read()

                if len(content) < 100:
                    raise IOError(f"VizieR response too small ({len(content)} bytes)")

                # Check for VOTable error response before saving
                text = content.decode("utf-8", errors="replace")
                if 'name="Error"' in text or 'ID="Error"' in text:
                    raise IOError(f"VizieR returned an error response for {catalog}")

                target.write_bytes(content)
                size = target.stat().st_size
                if min_size and size < min_size:
                    raise IOError(f"Downloaded file too small ({size} < {min_size} bytes)")

                actual_sha = self.compute_sha256(target)
                self._clear_failure(url)
                self._record(target, actual_sha, size, url, source, "downloaded")
                print(f"  [OK] {target.name} — {size:,} bytes")
                return True

            except (URLError, HTTPError, IOError, OSError) as e:
                wait = 2 ** (attempt - 1) * 5
                print(f"  [RETRY] {catalog} — {e}, waiting {wait}s")
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    print(f"  [FAIL] {catalog} — {e}")
                    self._record_failure(url, e)
                    self._record(target, "", 0, url, source, "failed", str(e))
                    return False

        return False

    # ------------------------------------------------------------------ #
    # GitHub raw file download
    # ------------------------------------------------------------------ #

    def download_github_raw(self, repo, path_in_repo, target,
                            branch="main", description="", source="",
                            expected_sha256=None, max_retries=3, min_size=0,
                            force=False):
        """Download a file from a GitHub repository's raw URL.

        Args:
            repo: GitHub repo as "owner/repo"
            path_in_repo: Path within the repo
            target: Local save path
            branch: Git branch (default "main")
            description: Human-readable description
            source: Data source name
            expected_sha256: Expected checksum
            max_retries: Retry count
            min_size: Minimum expected file size in bytes
            force: If True, bypass skip logic
        """
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path_in_repo}"
        return self.download_file(url, target, expected_sha256,
                                  description, source, max_retries,
                                  min_size=min_size, force=force)

    # ------------------------------------------------------------------ #
    # Verification and manifest
    # ------------------------------------------------------------------ #

    def verify_all(self):
        """Verify checksums for all files in the manifest."""
        print("\n=== Verifying all downloaded files ===")
        all_ok = True
        for rel_path, entry in self.manifest["files"].items():
            filepath = PROJECT_ROOT / rel_path
            if not filepath.exists():
                print(f"  [MISSING] {rel_path}")
                all_ok = False
                continue
            if entry.get("sha256"):
                actual = self.compute_sha256(filepath)
                if actual != entry["sha256"]:
                    print(f"  [MISMATCH] {rel_path}")
                    all_ok = False
                else:
                    print(f"  [OK] {rel_path}")
            else:
                print(f"  [PRESENT] {rel_path} (no checksum)")
        return all_ok

    def write_manifest(self):
        """Write the checksum manifest to disk."""
        self._save_manifest()
        print(f"\n=== Checksum manifest written to {self.manifest_path} ===")
        print(f"  {len(self.manifest['files'])} files recorded")

    def summary(self):
        """Print a download summary."""
        total = len(self.results)
        downloaded = sum(1 for r in self.results if r["status"] == "downloaded")
        skipped = sum(1 for r in self.results if r["status"] == "skip_exists")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        mismatch = sum(1 for r in self.results if r["status"] == "checksum_mismatch")
        cooldown = sum(1 for r in self.results if r["status"] == "skip_failure_cooldown")
        print(f"\n=== Download Summary ===")
        print(f"  Total:   {total}")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (exists): {skipped}")
        print(f"  Skipped (cooldown): {cooldown}")
        print(f"  Failed:  {failed}")
        print(f"  Checksum mismatch: {mismatch}")
        return {"total": total, "downloaded": downloaded,
                "skipped": skipped, "skipped_cooldown": cooldown,
                "failed": failed, "mismatch": mismatch}


# ---------------------------------------------------------------------- #
# Dataset registry — all external data sources in one place
# ---------------------------------------------------------------------- #

DATASET_REGISTRY = {
    # Pantheon+ SH0ES
    "pantheon_shoes_dat": {
        "url": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES.dat",
        "target": "data/raw/Pantheon+SH0ES.dat",
        "source": "Pantheon+SH0ES (Scolnic et al. 2022, Riess et al. 2022)",
        "description": "Cepheid + SN Ia distance ladder data",
        "min_size": 500_000,
    },
    "pantheon_stat_sys_cov": {
        "url": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES_STAT%2BSYS.cov",
        "target": "data/raw/Pantheon+SH0ES_STAT+SYS.cov",
        "source": "Pantheon+SH0ES (Scolnic et al. 2022)",
        "description": "Pantheon+ statistical + systematic covariance matrix",
        "min_size": 30_000_000,
    },
    "pantheon_statonly_cov": {
        "url": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2BSH0ES_STATONLY.cov",
        "target": "data/raw/Pantheon+SH0ES_STATONLY.cov",
        "source": "Pantheon+SH0ES (Scolnic et al. 2022)",
        "description": "Pantheon+ statistical-only covariance matrix",
        "min_size": 30_000_000,
    },

    # CosmicFlows-4 (CDS — requires Playwright for Anubis bypass)
    "cf4_table2": {
        "url": "http://cdsarc.u-strasbg.fr/ftp/J/ApJ/944/94/table2.dat.gz",
        "target": "data/raw/external/cf4_table2.dat.gz",
        "source": "CosmicFlows-4 (Tully et al. 2023, ApJ 944 94)",
        "description": "55,877 individual galaxy distances (all methodologies)",
        "decompress_gz": True,
        "requires_playwright": True,
        "min_size": 2_000_000,
    },
    "cf4_table4": {
        "url": "http://cdsarc.u-strasbg.fr/ftp/J/ApJ/944/94/table4.dat.gz",
        "target": "data/raw/external/cf4_table4.dat.gz",
        "source": "CosmicFlows-4 (Tully et al. 2023, ApJ 944 94)",
        "description": "38,053 galaxy groups with peculiar velocities",
        "decompress_gz": True,
        "requires_playwright": True,
        "min_size": 2_000_000,
    },
    "cf4_readme": {
        "url": "http://cdsarc.u-strasbg.fr/ftp/J/ApJ/944/94/ReadMe",
        "target": "data/raw/external/cf4_readme.txt",
        "source": "CDS ReadMe for J/ApJ/944/94",
        "description": "Column format documentation for CF4 tables",
        "requires_playwright": True,
        "min_size": 15_000,
    },

    # SPARC (Lelli et al. 2016)
    "sparc_table1": {
        "url": "http://astroweb.cwru.edu/SPARC/Table1.mrt",
        "target": "data/raw/external/sparc_table1.mrt",
        "source": "SPARC (Lelli et al. 2016, ApJL 826 L17)",
        "description": "SPARC galaxy mass models (175 late-type galaxies)",
        "min_size": 50_000,
    },
    "sparc_table2": {
        "url": "http://astroweb.cwru.edu/SPARC/Table2.mrt",
        "target": "data/raw/external/sparc_table2.mrt",
        "source": "SPARC (Lelli et al. 2016)",
        "description": "SPARC photometric and kinematic parameters",
        "min_size": 30_000,
    },
    "sparc_rotmod": {
        "url": "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
        "target": "data/raw/external/Rotmod_LTG.zip",
        "source": "SPARC (Lelli et al. 2016)",
        "description": "Rotation curve files for 175 SPARC galaxies",
        "min_size": 1_000_000,
    },

    # 2MTF (Hong et al. 2019)
    "2mtf_vizier": {
        "vizier_catalog": "J/MNRAS/487/2061",
        "target": "data/raw/external/2mtf_catalog.vot",
        "source": "2MTF (Hong et al. 2019, MNRAS 487 2061)",
        "description": "2MASS Tully-Fisher catalog (2,062 spirals)",
        "min_size": 1_000_000,
    },

    # SFI++ (Springob et al. 2007)
    "sfipp_vizier": {
        "vizier_catalog": "J/ApJS/182/474",
        "target": "data/raw/external/sfipp_catalog.vot",
        "source": "SFI++ (Springob et al. 2007, ApJS 182 474)",
        "description": "SFI++ I-band Tully-Fisher catalog (4,861 galaxies)",
        "min_size": 500_000,
    },

    # Jia et al. (2023) H(z) data
    "jia_hz": {
        "url": "https://raw.githubusercontent.com/JoJo20221003/Hz-Code/main/Data/Hz%20data.txt",
        "target": "data/raw/external/jia_hz_data.txt",
        "source": "Jia et al. 2023, A&A 674 A45",
        "description": "60 H(z) measurements (cosmic chronographs + BAO)",
        "min_size": 500,
    },

    # ALFALFA A40 (Haynes et al. 2018)
    "alfalfa_a40_vizier": {
        "vizier_catalog": "J/AJ/154/78/table2",
        "target": "data/raw/external/alfalfa_a40.vot",
        "source": "ALFALFA A40 (Haynes et al. 2018, AJ 154 78)",
        "description": "ALFALFA alpha.40 HI source catalog",
        "min_size": 500_000,
    },

    # 6dFGS peculiar velocities (Springob et al. 2014)
    "6dfgs_vizier": {
        "vizier_catalog": "J/MNRAS/445/2677/table1",
        "target": "data/raw/external/6dfgs_pecvel.vot",
        "source": "6dFGS (Springob et al. 2014, MNRAS 445 2677)",
        "description": "6dFGS peculiar velocity catalog",
        "min_size": 1_000_000,
    },
}
