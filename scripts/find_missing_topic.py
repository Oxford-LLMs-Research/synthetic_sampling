#!/usr/bin/env python
"""Find which topic is missing and why."""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

data_file = Path(__file__).parent.parent / 'analysis' / 'disaggregated' / 'by_topic_tag.json'
data = json.load(open(data_file))

# Get all unique topics across all models
all_topics = set()
for model, topics in data.items():
    all_topics.update(topics.keys())

all_topics.discard('Unknown')
print(f"Total unique topics found: {len(all_topics)}")
print(f"Topics: {sorted(all_topics)}\n")

# Check each topic using the same logic as aggregate_by_dimension
dim_values = defaultdict(list)
dim_n = defaultdict(int)

for model, topics in data.items():
    for topic, metrics in topics.items():
        if topic == 'Unknown':
            continue
        # Skip topics with no data or invalid metrics
        n_obs = metrics.get('n', 0)
        acc_value = metrics.get('accuracy', 0)
        
        # Only include if we have valid data (n > 0 and valid accuracy)
        if n_obs > 0 and not np.isnan(acc_value) and acc_value is not None:
            dim_values[topic].append(acc_value)
            dim_n[topic] = n_obs

print(f"Topics with valid data: {len(dim_values)}")
print(f"Valid topics: {sorted(dim_values.keys())}\n")

# Find missing topics
missing_topics = all_topics - set(dim_values.keys())
print(f"Missing topics: {missing_topics}\n")

# Check why they're missing
for topic in missing_topics:
    print(f"Checking {topic}:")
    models_with_topic = [m for m in data if topic in data[m]]
    print(f"  Found in {len(models_with_topic)} models")
    
    for model in models_with_topic[:3]:  # Check first 3 models
        metrics = data[model][topic]
        n_obs = metrics.get('n', 0)
        acc_value = metrics.get('accuracy', None)
        
        print(f"    {model}:")
        print(f"      n={n_obs} (valid: {n_obs > 0})")
        print(f"      accuracy={acc_value} (type: {type(acc_value)})")
        if acc_value is not None:
            print(f"      is_nan: {np.isnan(acc_value) if isinstance(acc_value, (int, float)) else 'N/A'}")
        
        # Check why it's invalid
        if n_obs == 0:
            print(f"      ❌ REASON: n=0")
        elif acc_value is None:
            print(f"      ❌ REASON: accuracy is None")
        elif isinstance(acc_value, (int, float)) and np.isnan(acc_value):
            print(f"      ❌ REASON: accuracy is NaN")
        else:
            print(f"      ✅ Should be valid!")
    print()
