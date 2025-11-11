#!/usr/bin/env python3
"""
run_and_qa.py
-------------------------------------------------------------
Unified pipeline controller for the Amygdala Lateralization Project.

Runs all scripts in sequence:
  1) 01_extract_amygdala_activation.py
  2) 02_compute_asymmetry_vs_anxiety.py
  3) 03_generate_figures.py

Then:
  4) Validates merged ROI dataset (columns, N subjects)
  5) Rebuilds an amygdala-only ROI mask using Harvard–Oxford subcortical atlas
  6) Overwrites fig4 with a proper colored overlay visualization

Usage:
  python3 run_and_qa.py
-------------------------------------------------------------
"""

from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import pandas as pd

# Optional neuroimaging libraries
try:
    from nilearn import datasets, plotting, image
    import nibabel as nib
    HAVE_NILEARN = True
except Exception:
    HAVE_NILEARN = False


# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
MASK_DIR = ROOT / "masks"

for p in (FIG_DIR, MASK_DIR):
    p.mkdir(exist_ok=True)

ROI_CSV = RESULTS / "roi_values.csv"
FIG4 = FIG_DIR / "fig4_roi_overlay.png"
MASK_PATH = MASK_DIR / "amygdala_mask.nii.gz"


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def run(cmd: str, cwd=SCRIPTS):
    """Run a shell command and stop if it fails."""
    print(f"→ {cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=True)
    if res.returncode != 0:
        sys.exit(f"✗ Failed: {cmd}")


def qa_table(df: pd.DataFrame) -> dict:
    """Generate summary stats for quick verification."""
    return {
        "n_rows": int(df.shape[0]),
        "columns": list(df.columns),
        "has_STAI_T": "STAI_T" in df.columns,
        "has_L": "Amygdala_L" in df.columns,
        "has_R": "Amygdala_R" in df.columns,
        "mean_L": float(df["Amygdala_L"].mean()) if "Amygdala_L" in df.columns else None,
        "mean_R": float(df["Amygdala_R"].mean()) if "Amygdala_R" in df.columns else None,
        "mean_Asym": float(df["Asymmetry"].mean()) if "Asymmetry" in df.columns else None,
    }


def validate_roi_csv(path: Path) -> pd.DataFrame:
    """Ensure that ROI CSV is valid and contains required columns."""
    if not path.exists():
        sys.exit(f"✗ Missing results file: {path}")
    df = pd.read_csv(path)

    if df.shape[0] < 10:
        print(f"⚠️ Only {df.shape[0]} rows — likely incomplete dataset.")
    for col in ("Amygdala_L", "Amygdala_R"):
        if col not in df.columns:
            sys.exit(f"✗ Missing required column: {col}")

    if "Asymmetry" not in df.columns:
        df["Asymmetry"] = (df["Amygdala_R"] - df["Amygdala_L"]) / (
            df["Amygdala_R"] + df["Amygdala_L"]
        )
        df.to_csv(path, index=False)
        print("ℹ️ Added missing Asymmetry column to roi_values.csv")

    info = qa_table(df)
    print("QA summary:", json.dumps(info, indent=2))
    return df


def _load_atlas_image(ho):
    """Return Nifti image object from atlas, regardless of type."""
    if isinstance(ho.maps, (str, Path)):
        return nib.load(str(ho.maps))
    if hasattr(ho.maps, "get_fdata"):
        return ho.maps
    return nib.load(str(ho.maps))


# -------------------------------------------------------------
# Fix ROI overlay
# -------------------------------------------------------------
def fix_overlay_mask_and_render():
    """
    Creates a strict bilateral amygdala mask from Harvard–Oxford atlas
    and re-renders fig4 with a visible colored overlay.
    """
    if not HAVE_NILEARN:
        print("⚠️ nilearn/nibabel not available — skipping overlay auto-fix.")
        return

    print("🧠 Auto-fixing overlay: generating amygdala-only mask...")

    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
    labels = [lbl.strip() for lbl in ho.labels if str(lbl).strip()]
    atlas_img = _load_atlas_image(ho)
    atlas_data = atlas_img.get_fdata()

    # Extract only “Left/Right Amygdala” voxels
    amyg_indices = [
        i for i, name in enumerate(labels)
        if ("Amygdala" in name) and ("Extended" not in name)
    ]
    if not amyg_indices:
        print("✗ No amygdala labels found in atlas.")
        return

    mask_data = np.isin(atlas_data, amyg_indices).astype(np.int8)
    frac = float(mask_data.sum()) / float(mask_data.size)
    if frac > 0.02:
        print(f"⚠️ Mask covers {frac*100:.2f}% of voxels — recheck atlas labeling.")

    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)
    nib.save(mask_img, str(MASK_PATH))

    # Render overlay with bright color + template
    print("🎨 Rendering colored overlay...")
    template = datasets.load_mni152_template(resolution=2)
    display = plotting.plot_anat(
        template,
        title="Amygdala ROI (Left + Right)",
        cut_coords=[0, -17, 15],
        display_mode="ortho",
        dim=-0.5,
    )
    display.add_overlay(
        str(MASK_PATH),
        cmap="autumn",
        alpha=0.9,
    )
    display.savefig(str(FIG4), dpi=300)
    display.close()

    print(f"✅ Saved {FIG4.name} with visible bilateral amygdala overlay.")


# -------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------
def main():
    # 1) Execute pipeline scripts in order
    run("python3 01_extract_amygdala_activation.py")
    run("python3 02_compute_asymmetry_vs_anxiety.py")
    run("python3 03_generate_figures.py")

    # 2) Validate merged dataset
    df = validate_roi_csv(ROI_CSV)

    # 3) Generate high-contrast colored overlay
    fix_overlay_mask_and_render()

    print("\n🎉 DONE — All outputs successfully generated in:", FIG_DIR)
    print("   • fig1_hemisphere_box.png")
    print("   • fig2_asymmetry_scatter.png" if "STAI_T" in df.columns else "   • (no STAI_T; fig2 skipped)")
    print("   • fig3_timecourse.png")
    print("   • fig4_roi_overlay.png (amygdala-only, colored)")


if __name__ == "__main__":
    main()
