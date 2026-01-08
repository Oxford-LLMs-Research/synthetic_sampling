import sys
sys.path.insert(0, 'src')

from synthetic_sampling import DataPaths
from synthetic_sampling.loaders import SurveyLoader

paths = DataPaths(
    raw_data_dir='C:/Users/murrn/cursor/synthetic_sampling/data',
    metadata_dir='C:/Users/murrn/cursor/synthetic_sampling/synthetic_sampling/src/synthetic_sampling/profiles/metadata',
    output_dir='./outputs'
)

loader = SurveyLoader(paths)
df, metadata = loader.load_survey('afrobarometer')

# Check metadata structure
print(f"Metadata type: {type(metadata)}")
print(f"Top-level keys: {list(metadata.keys())[:5]}")

# Check first section
first_key = list(metadata.keys())[0]
print(f"\nFirst key: '{first_key}'")
print(f"Type of metadata['{first_key}']: {type(metadata[first_key])}")

# If it's a dict, show its structure
if isinstance(metadata[first_key], dict):
    sub_keys = list(metadata[first_key].keys())[:3]
    print(f"Sub-keys: {sub_keys}")
    if sub_keys:
        print(f"Type of first sub-value: {type(metadata[first_key][sub_keys[0]])}")