import pandas as pd
import requests
import os

# --- Load your filtered high-anxiety list ---
high_anx = pd.read_csv("high_anxiety_subjects.csv")

# --- Base URL for the ds003097 dataset on GitHub ---
BASE_URL = "https://github.com/OpenNeuroDatasets/ds003097/raw/master/"

# --- Create local directories ---
fmri_dir = os.path.join("..", "data", "fmri")
os.makedirs(fmri_dir, exist_ok=True)

# --- Loop through each high-anxiety participant ---
for sub in high_anx['participant_id']:
    fmri_file = f"{sub}_task-moviewatching_bold.nii.gz"
    events_file = f"{sub}_task-moviewatching_recording-respcardiac_physio.tsv.gz"

    fmri_url = f"{BASE_URL}{sub}/func/{fmri_file}"
    events_url = f"{BASE_URL}{sub}/func/{events_file}"

    fmri_path = os.path.join(fmri_dir, fmri_file)
    events_path = os.path.join(fmri_dir, events_file)

    print(f"Downloading {sub}...")

    # --- Download the .nii.gz ---
    fmri_response = requests.get(fmri_url)
    if fmri_response.status_code == 200:
        with open(fmri_path, "wb") as f:
            f.write(fmri_response.content)
        print(f"✓ Saved {fmri_file}")
    else:
        print(f"⚠️ Could not find {fmri_file} (status {fmri_response.status_code})")

    # --- Download the events/physio file ---
    events_response = requests.get(events_url)
    if events_response.status_code == 200:
        with open(events_path, "wb") as f:
            f.write(events_response.content)
        print(f"✓ Saved {events_file}")
    else:
        print(f"⚠️ Could not find {events_file} (status {events_response.status_code})")

print("✅ Download complete.")
