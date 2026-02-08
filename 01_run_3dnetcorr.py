#!/usr/bin/env python3
"""
Calculate network connectivity for multiple subjects using AFNI's 3dNetCorr
and the Yeo 2011 17-network atlas. Runs the analysis in parallel.
"""

import os
import glob
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---

YEO_ATLAS = "./Yeo_JNeurophysiol11_MNI152/Yeo2011_17Networks_MNI152_FreeSurferConformed2mm_LiberalMask.nii.gz"
MASK_FILE = "mask_8.nii.gz"
OUTPUT_DIR = "./Yeo17"
MAX_JOBS = 40
SUBJECT_GLOB = "sub*.nii.gz"


def run_netcorr(
    subject_file: str,
    atlas: str,
    mask: str,
    output_dir: str,
) -> Tuple[str, bool, Optional[str]]:
    """
    Run 3dNetCorr for a single subject.
    Returns (base_name, success, error_message).
    """
    subject_file = os.path.abspath(subject_file)
    base = Path(subject_file).name.replace("_rest_preproc_mni.nii.gz", "")

    print(f"Starting job for subject: {base}")

    cmd = [
        "3dNetCorr",
        "-inset", subject_file,
        "-in_rois", atlas,
        "-mask", mask,
        "-fish_z",
        "-ts_out",
        "-prefix", os.path.join(output_dir, base),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"Finished job for subject: {base} (FAILED)")
            return base, False, result.stderr or result.stdout
        print(f"Finished job for subject: {base}")
        return base, True, None
    except subprocess.TimeoutExpired:
        print(f"Finished job for subject: {base} (TIMEOUT)")
        return base, False, "Timeout"
    except Exception as e:
        print(f"Finished job for subject: {base} (ERROR)")
        return base, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run 3dNetCorr (Yeo 17-network) on subject NIfTI files in parallel."
    )
    parser.add_argument(
        "--atlas", default=YEO_ATLAS,
        help=f"Path to Yeo 17-network atlas (default: {YEO_ATLAS})",
    )
    parser.add_argument(
        "--mask", default=MASK_FILE,
        help=f"Path to whole-brain mask (default: {MASK_FILE})",
    )
    parser.add_argument(
        "-o", "--output-dir", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=MAX_JOBS,
        help=f"Max parallel jobs (default: {MAX_JOBS})",
    )
    parser.add_argument(
        "--pattern", default=SUBJECT_GLOB,
        help=f"Glob for subject files (default: {SUBJECT_GLOB})",
    )
    parser.add_argument(
        "--work-dir", default=".",
        help="Working directory where subject files and mask live (default: current)",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    atlas = os.path.abspath(os.path.join(work_dir, args.atlas))
    mask = os.path.abspath(os.path.join(work_dir, args.mask))
    output_dir = os.path.abspath(os.path.join(work_dir, args.output_dir))

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(atlas):
        raise FileNotFoundError(f"Yeo atlas not found: {atlas}")
    if not os.path.isfile(mask):
        raise FileNotFoundError(f"Mask file not found: {mask}")

    subject_files = sorted(glob.glob(os.path.join(work_dir, args.pattern)))
    if not subject_files:
        print(f"No files matching '{args.pattern}' in {work_dir}")
        return

    print(f"Starting network correlation analysis with up to {args.jobs} parallel jobs...")

    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_netcorr, f, atlas, mask, output_dir): f
            for f in subject_files
        }
        for future in as_completed(futures):
            base, success, err = future.result()
            if not success:
                failed.append((base, err))

    if failed:
        print(f"\n{len(failed)} job(s) failed:")
        for base, err in failed:
            print(f"  {base}: {err}")
    else:
        print(f"\nAnalysis complete. All output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
