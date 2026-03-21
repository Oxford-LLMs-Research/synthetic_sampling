#!/usr/bin/env python
"""Quick diagnostic to check why certain topics might be missing."""
import json
from pathlib import Path

data_file = Path(__file__).parent.parent / 'analysis' / 'disaggregated' / 'by_topic_tag.json'
data = json.load(open(data_file))

topics_to_check = ['voting', 'partisanship', 'service_delivery', 'political_priorities']

print("Checking topics in by_topic_tag.json:\n")
for topic in topics_to_check:
    print(f"{topic}:")
    models_with_topic = [m for m in data if topic in data[m]]
    print(f"  Found in {len(models_with_topic)}/{len(data)} models")
    
    if models_with_topic:
        for model in models_with_topic[:3]:  # Show first 3 models
            metrics = data[model][topic]
            print(f"    {model}: n={metrics.get('n', 'N/A')}, accuracy={metrics.get('accuracy', 'N/A')}")
        
        # Check if all have valid data
        all_valid = all(
            data[m][topic].get('n', 0) > 0 
            and not (data[m][topic].get('accuracy') is None or str(data[m][topic].get('accuracy', '')).lower() == 'nan')
            for m in models_with_topic
        )
        print(f"  All models have valid data: {all_valid}")
    print()
