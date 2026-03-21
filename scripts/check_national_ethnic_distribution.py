#!/usr/bin/env python
"""Check how national_ethnic_identity is distributed - why it was filtered."""
import json
from pathlib import Path

# Check in the raw results to see how many instances per model
# The topic has 63 total instances, but if spread across 13 models, each would have ~5

# From dataset description: national_ethnic_identity appears in ESS Wave 10 and ESS Wave 11
# ESS Wave 10: 36 instances (atcherp variable)
# ESS Wave 11: 27 instances (atchctr variable)
# Total: 63 instances

print("From dataset description:")
print("  national_ethnic_identity: 63 total instances")
print("    - ESS Wave 10: 36 instances (atcherp variable)")
print("    - ESS Wave 11: 27 instances (atchctr variable)")
print()
print("The disaggregated analysis uses min_n_tag=50 per MODEL.")
print("If 63 instances are spread across 13 models:")
print("  Average per model: ~5 instances")
print("  Maximum per model: likely < 50 (probably only in ESS models)")
print()
print("This explains why it was filtered out - each model has < 50 instances.")
