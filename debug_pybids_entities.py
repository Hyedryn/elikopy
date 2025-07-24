"""
Debug script to check what entities are available in the qsiprep data
"""

from pathlib import Path
from bids.layout import BIDSLayout

# Initialize layout
bids_example_dir = Path("bids_example")
qsiprep_dir = bids_example_dir / "derivatives" / "qsiprep"

layout = BIDSLayout(qsiprep_dir, validate=False)

print("Available entities:")
entities = layout.get_entities()
for entity_name, entity_obj in entities.items():
    print(f"  {entity_name}: {entity_obj}")

print("\nAll files in layout:")
all_files = layout.get()
for file in all_files:
    print(f"  {file.path}")
    print(f"    Entities: {file.get_entities()}")

print("\nDWI files specifically:")
dwi_files = layout.get(datatype='dwi', suffix='dwi', extension='.nii.gz')
for file in dwi_files:
    print(f"  {file.path}")
    print(f"    Entities: {file.get_entities()}")