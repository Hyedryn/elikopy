import sys
sys.path.append(r"C:\Users\quent\Documents\elikopy")

import elikopy
from pathlib import Path

# Initialize study with BIDS dataset
study = elikopy.ElikopyStudy(r"C:\Users\quent\Documents\elikopy\bids_example", derivatives_name="elikopy")

study.setup_from_qsiprep(r"C:\Users\quent\Documents\elikopy\bids_example\derivatives\qsiprep")

print("Study step 1 done")

