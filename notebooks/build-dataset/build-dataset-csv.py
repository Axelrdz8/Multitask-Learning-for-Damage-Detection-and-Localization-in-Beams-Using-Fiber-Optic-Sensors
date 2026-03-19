from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re as _re_idx

# ---------------- Configuration ---------------- #
N_PEAKS = 5

CLASS_MAP = {
    "healthy": 0,
    "crack_slight": 1, "crack_moderate": 2, "crack_severe": 3,
    "wear_slight": 4, "wear_moderate": 5, "wear_severe": 6,
    # 7..11 are assigned based on the folder inside 'simultaneous'
}

# Mapping of subfolders inside /simultaneous to class and fixed distances
SIMULTANEOUS_MAP = {
    "severecrack_slightwear":     {"cls": 7,  "crack_cm": 6.5, "wear_cm": 3.2},
    "severecrack_moderatewear":   {"cls": 8,  "crack_cm": 6.5, "wear_cm": 3.2},
    "severecrack_severewear":     {"cls": 11, "crack_cm": 6.5, "wear_cm": 3.2},
    "severewear_slightcrack":     {"cls": 9,  "crack_cm": 2.2, "wear_cm": 6.5},
    "severewear_moderatecrack":   {"cls": 10, "crack_cm": 2.2, "wear_cm": 6.5},
    "severewear_severecrack":     {"cls": 11, "crack_cm": 2.2, "wear_cm": 6.5},
}

RE_CM = re.compile(r"(\d+(?:[.,]\d+)?)\s*cm", re.I)
RE_BEAM_CM = re.compile(r"beam\s*(\d+(?:[.,]\d+)?)\s*cm", re.I)

# === Fill values to avoid NaNs during training ===
FILL_DISTANCE_FOR_HEALTHY = 0.0
FILL_SEVERITY_FOR_HEALTHY = "none"
FILL_NUMERIC_NA = 0.0

_RE_ALL_IDX = _re_idx.compile(r"all\s*(\d{4})", _re_idx.I)

def _extract_all_index(all_folder_name: str) -> int | None:
    m = _RE_ALL_IDX.search(all_folder_name)
    return int(m.group(1)) if m else None

# ---------------- D/E Reading (Tektronix) ---------------- #
def _read_signal_csv_de(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, header=None, engine="python", sep=None)
    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.shape[1] < 5:
        raise RuntimeError(f"{csv_path.name}: D/E columns not found (>=5 columns required).")
    t_col, y_col = 3, 4
    mask = df_num[t_col].notna() & df_num[y_col].notna()
    sub = df_num.loc[mask, [t_col, y_col]]
    if sub.empty:
        raise RuntimeError(f"{csv_path.name}: D/E columns do not contain numeric data.")
    t = sub[t_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    return y, t

# ---------------- FFT and Features ---------------- #
def _compute_fft(y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float).flatten()
    y = y - np.nanmean(y)
    n = len(y)
    if n < 4:
        return np.array([]), np.array([])
    dt = float(np.nanmedian(np.diff(t)))
    fs = 1.0 / dt if np.isfinite(dt) and dt > 0 else 1.0
    win = np.hanning(n)
    Y = np.fft.rfft(y * win, n=n)
    mag = np.abs(Y) / (np.sum(win) / 2.0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag

def _top_peaks(freqs: np.ndarray, mag: np.ndarray, k: int) -> List[Tuple[float, float]]:
    if freqs.size == 0:
        return []
    low_cut = 15.0 if freqs[-1] > 2 else 0.01
    mask = freqs >= low_cut
    f, m = freqs[mask], mag[mask]
    if f.size == 0:
        return []
    k = min(k, f.size)
    idx = np.argpartition(m, -k)[-k:]
    idx = idx[np.argsort(m[idx])[::-1]]
    return [(float(f[i]), float(m[i])) for i in idx]

def _spectral_features(freqs: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
    if freqs.size == 0:
        return {k: math.nan for k in ["FC_Hz", "FRMS_Hz", "FRVF_Hz", "m0", "m1", "m2"]}
    power = mag ** 2
    m0 = float(np.sum(power))
    if m0 == 0 or not np.isfinite(m0):
        return {k: math.nan for k in ["FC_Hz", "FRMS_Hz", "FRVF_Hz", "m0", "m1", "m2"]}
    m1 = float(np.sum(freqs * power))
    m2 = float(np.sum((freqs ** 2) * power))
    FC = m1 / m0
    FRMS = float(np.sqrt(m2 / m0))
    FRVF = float(np.sqrt(np.sum(((freqs - FC) ** 2) * power) / m0))
    return {
        "FC_Hz": float(FC),
        "FRMS_Hz": float(FRMS),
        "FRVF_Hz": FRVF,
        "m0": m0,
        "m1": m1,
        "m2": m2
    }

# ---------------- Metadata Parsers ---------------- #
def _parse_simultaneous_folder(folder_name: str) -> Optional[Dict[str, float | int]]:
    key = folder_name.strip().lower()
    if key in SIMULTANEOUS_MAP:
        info = SIMULTANEOUS_MAP[key]
        return {
            "target_class": info["cls"],
            "distance_crack_cm": float(info["crack_cm"]),
            "distance_wear_cm": float(info["wear_cm"]),
        }
    return None

def _parse_metadata_from_path(path: Path) -> Dict[str, Optional[str | float]]:
    parts = [p.lower() for p in path.parts]

    # mechanism
    if "simultaneous" in parts:
        mechanism = "simultaneous"
    elif "healthy" in parts:
        mechanism = "healthy"
    elif "crack" in parts:
        mechanism = "crack"
    elif "wear" in parts:
        mechanism = "wear"
    else:
        mechanism = None

    # severity
    severity = None
    for sev in ("slight", "moderate", "severe"):
        if sev in parts:
            severity = sev
            break

    # Also check compound names such as slightdamage_1.5cm_beam10cm
    if severity is None:
        joined = " ".join(parts)
        if "slightdamage" in joined or "slightcrack" in joined or "slightwear" in joined:
            severity = "slight"
        elif "moderatedamage" in joined or "moderatecrack" in joined or "moderatewear" in joined:
            severity = "moderate"
        elif "severedamage" in joined or "severecrack" in joined or "severewear" in joined:
            severity = "severe"

    # distance
    distance_cm = None
    for p in parts:
        m = RE_CM.search(p)
        if m:
            distance_cm = float(m.group(1).replace(",", "."))
            break

    # beam length
    beam_cm = None
    for p in parts:
        m = RE_BEAM_CM.search(p)
        if m:
            beam_cm = float(m.group(1).replace(",", "."))
            break

    return {
        "mechanism": mechanism,
        "severity": severity,
        "distance_cm": distance_cm,
        "beam_cm": beam_cm
    }

def _class_from_meta(mechanism: Optional[str], severity: Optional[str]) -> int:
    if mechanism == "healthy":
        return CLASS_MAP["healthy"]
    if mechanism in ("crack", "wear") and severity in ("slight", "moderate", "severe"):
        return CLASS_MAP[f"{mechanism}_{severity}"]
    return -1  # simultaneous or other

def _preclass_for_dir(adir: Path) -> int:
    """
    Pre-classifies the directory to sort processing order:
    0 first (healthy), then 1..11. Unknown classes go last (99).
    """
    parts_lower = [p.lower() for p in adir.parts]

    if "simultaneous" in parts_lower:
        try:
            idx = parts_lower.index("simultaneous")
            sim_sub = parts_lower[idx + 1] if idx + 1 < len(parts_lower) else ""
            info = _parse_simultaneous_folder(sim_sub)
            if info:
                return int(info["target_class"])
        except Exception:
            pass
        return 99

    meta = _parse_metadata_from_path(adir)
    cls = _class_from_meta(meta["mechanism"], meta["severity"])
    return cls if cls >= 0 else 99

# ---------------- Segmentation ---------------- #
def _split_segments_by_type(y: np.ndarray, t: np.ndarray,
                            mechanism: Optional[str],
                            all_index: Optional[int]) -> list[tuple[int, int]]:
    """
    - 'crack' / 'wear' / 'simultaneous' -> 2 segments (halves).
    - 'healthy':
        * ALL0000..ALL0049 -> 1 segment
        * ALL0050..ALL0099 -> 2 segments
    """
    n = len(y)
    if n < 8:
        return [(0, n)]

    if mechanism in ("crack", "wear", "simultaneous"):
        mid = n // 2
        return [(0, mid), (mid, n)]

    if all_index is not None and all_index >= 50:
        mid = n // 2
        return [(0, mid), (mid, n)]
    return [(0, n)]

# ---------------- Pipeline ---------------- #
def process_dataset(root_dir: Path, out_csv: Path, n_peaks: int = N_PEAKS) -> pd.DataFrame:
    rows = []

    all_dirs = sorted([p for p in root_dir.rglob("*") if p.is_dir() and p.name.lower().startswith("all")])
    if not all_dirs:
        all_csvs = sorted([p for p in root_dir.rglob("*.csv")])
        all_dirs = list({csv.parent for csv in all_csvs})

    all_dirs_sorted = sorted(all_dirs, key=lambda d: (_preclass_for_dir(d), d.as_posix()))

    for adir in all_dirs_sorted:
        meta = _parse_metadata_from_path(adir)
        all_index = _extract_all_index(adir.name)

        if meta["beam_cm"] is None or (isinstance(meta["beam_cm"], float) and math.isnan(meta["beam_cm"])):
            meta["beam_cm"] = 10.0

        parts_lower = [p.lower() for p in adir.parts]
        sim_info = None
        if "simultaneous" in parts_lower:
            try:
                sim_idx = parts_lower.index("simultaneous")
                sim_sub = parts_lower[sim_idx + 1] if sim_idx + 1 < len(parts_lower) else ""
                sim_info = _parse_simultaneous_folder(sim_sub)
            except Exception:
                sim_info = None

        if sim_info is not None:
            target_class = int(sim_info["target_class"])
            distance_crack_cm = float(sim_info["distance_crack_cm"])
            distance_wear_cm = float(sim_info["distance_wear_cm"])
            crack_mask, wear_mask = 1, 1
            meta_mechanism = "simultaneous"
        else:
            target_class = _class_from_meta(meta["mechanism"], meta["severity"])
            meta_mechanism = meta["mechanism"]

            if target_class == CLASS_MAP["healthy"]:
                meta["severity"] = FILL_SEVERITY_FOR_HEALTHY
                distance_crack_cm = FILL_DISTANCE_FOR_HEALTHY
                distance_wear_cm = FILL_DISTANCE_FOR_HEALTHY
                crack_mask, wear_mask = 0, 0
            else:
                if meta_mechanism == "crack":
                    distance_crack_cm = float(meta["distance_cm"]) if meta["distance_cm"] is not None else math.nan
                    distance_wear_cm = FILL_DISTANCE_FOR_HEALTHY
                    crack_mask, wear_mask = 1, 0
                elif meta_mechanism == "wear":
                    distance_wear_cm = float(meta["distance_cm"]) if meta["distance_cm"] is not None else math.nan
                    distance_crack_cm = FILL_DISTANCE_FOR_HEALTHY
                    crack_mask, wear_mask = 0, 1
                else:
                    distance_crack_cm = math.nan
                    distance_wear_cm = math.nan
                    crack_mask, wear_mask = math.nan, math.nan

        csv_files = sorted({p.resolve() for p in adir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"})

        for csv_path in csv_files:
            try:
                y, t = _read_signal_csv_de(csv_path)
                segments = _split_segments_by_type(y, t, meta_mechanism, all_index)

                for seg_idx, (i0, i1) in enumerate(segments, start=1):
                    yy = y[i0:i1]
                    tt = t[i0:i1]
                    if len(yy) < 8:
                        continue

                    freqs, mag = _compute_fft(yy, tt)
                    feats = _spectral_features(freqs, mag)
                    peaks = _top_peaks(freqs, mag, n_peaks)

                    rec: Dict[str, object] = {
                        "file_path": str(csv_path),
                        "folder_all": adir.name,
                        "all_index": all_index,
                        "segment_id": seg_idx,
                        "t0": float(tt[0]) if len(tt) else FILL_NUMERIC_NA,
                        "t1": float(tt[-1]) if len(tt) else FILL_NUMERIC_NA,
                        "mechanism": meta_mechanism,
                        "severity": meta["severity"],
                        "beam_cm": meta["beam_cm"],
                        "target_class": target_class,
                        "distance_crack_cm": distance_crack_cm,
                        "distance_wear_cm": distance_wear_cm,
                        "crack_mask": crack_mask,
                        "wear_mask": wear_mask,
                        **feats
                    }

                    for i in range(n_peaks):
                        rec[f"f{i+1}_Hz"] = float(peaks[i][0]) if i < len(peaks) else FILL_NUMERIC_NA
                        rec[f"a{i+1}"] = float(peaks[i][1]) if i < len(peaks) else FILL_NUMERIC_NA

                    rows.append(rec)

            except Exception as e:
                rows.append({
                    "file_path": str(csv_path),
                    "folder_all": adir.name,
                    "all_index": all_index,
                    "mechanism": meta_mechanism,
                    "severity": meta.get("severity"),
                    "beam_cm": meta["beam_cm"] if meta.get("beam_cm") is not None else 10.0,
                    "target_class": target_class if target_class != -1 else -1,
                    "distance_crack_cm": FILL_NUMERIC_NA,
                    "distance_wear_cm": FILL_NUMERIC_NA,
                    "crack_mask": 0,
                    "wear_mask": 0,
                    "error": str(e)
                })

    df = pd.DataFrame(rows)

    if "severity" in df.columns:
        df["severity"] = df["severity"].fillna(FILL_SEVERITY_FOR_HEALTHY)

    if "target_class" in df.columns:
        healthy_mask = df["target_class"] == CLASS_MAP["healthy"]
        df.loc[healthy_mask, ["distance_crack_cm", "distance_wear_cm"]] = FILL_DISTANCE_FOR_HEALTHY
        df.loc[healthy_mask, ["crack_mask", "wear_mask"]] = 0

    numeric_feature_regex = re.compile(
        r"^(f\d+_Hz|a\d+|FC_Hz|FRMS_Hz|FRVF_Hz|m[0-2]|"
        r"distance_crack_cm|distance_wear_cm|crack_mask|wear_mask)$"
    )
    num_cols = [c for c in df.columns if numeric_feature_regex.match(c)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(FILL_NUMERIC_NA)

    base_cols = [
        "mechanism", "severity", "beam_cm", "target_class",
        "distance_crack_cm", "distance_wear_cm", "crack_mask", "wear_mask",
        "FC_Hz", "FRMS_Hz", "FRVF_Hz", "m0", "m1", "m2"
    ]
    peak_cols = [c for c in df.columns if re.match(r"f\d+_Hz|a\d+$", c)]
    other_cols = [c for c in df.columns if c not in set(base_cols + peak_cols)]
    ordered = base_cols + sorted(peak_cols, key=lambda x: (x[0], int(re.findall(r"\d+", x)[0]))) + other_cols
    df = df.reindex(columns=[c for c in ordered if c in df.columns])

    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df

# ---------------- CLI ---------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Extracts peak frequencies and spectral features, and generates multitask targets."
    )
    ap.add_argument("--root", required=True, help="Root dataset folder")
    ap.add_argument("--out", default="beam_vibration_dataset.csv", help="Output CSV file")
    ap.add_argument("--peaks", type=int, default=N_PEAKS, help="Number of peaks to extract")
    args = ap.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    if not root_dir.exists():
        raise SystemExit(f"Path does not exist: {root_dir}")

    df = process_dataset(root_dir, out_csv, n_peaks=args.peaks)
    print(f"Done. Records: {len(df)}")
    print(f"Saved to: {out_csv}")

if __name__ == "__main__":
    main()