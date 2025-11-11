# -------------------------------------------------------------
# 01_extract_amygdala_activation.py
# Extracts left/right amygdala activation values and saves CSV
# -------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FMRI_RESULTS = ROOT / "data" / "fmri" / "results"
FMRI_RESULTS.mkdir(parents=True, exist_ok=True)

# Mock extraction (replace this block with actual BIDS/NIfTI parsing later)
# -------------------------------------------------------------------------
timepoints = np.arange(0, 290)
subjects = ["sub-0006", "sub-0189", "sub-0924"]
rng = np.random.default_rng(42)
data = {
    "timepoint": timepoints,
    "sub-0006": rng.normal(135000, 500, len(timepoints)),
    "sub-0189": rng.normal(325, 1, len(timepoints)),
    "sub-0924": rng.normal(258, 2, len(timepoints)),
}
df = pd.DataFrame(data)

# Save file
out_path = FMRI_RESULTS / "amygdala_activation.csv"
df.to_csv(out_path, index=False)
print(f"✅ Saved amygdala activation matrix to: {out_path}")
