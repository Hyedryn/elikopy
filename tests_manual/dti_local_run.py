import sys
sys.path.append(r"C:\Users\quent\Documents\elikopy")

import elikopy
from pathlib import Path

config = elikopy.ElikopyConfig.from_file(r"C:\Users\quent\Documents\elikopy\tests_manual\test_dti_config.yaml")
# Initialize study with BIDS dataset
study = elikopy.ElikopyStudy(r"C:\Users\quent\Documents\elikopy\tests_manual\elikopy_output", derivatives_name="elikopy", config=config)

study.setup_from_qsiprep(r"C:\Users\quent\Documents\elikopy\bids_example\derivatives\qsiprep")
study.get_study_summary()

print("Study step 1 done")

processor = study.create_processor(processing_type="dti")

print("Processor created.")

