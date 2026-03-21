#!/usr/bin/env python
"""Check if any topics appear in only some models."""
import json
from pathlib import Path
from collections import defaultdict

data_file = Path(__file__).parent.parent / 'analysis' / 'disaggregated' / 'by_topic_tag.json'
data = json.load(open(data_file))

# Count how many models each topic appears in
topic_model_count = defaultdict(int)
all_topics = set()

for model, topics in data.items():
    for topic in topics.keys():
        if topic != 'Unknown':
            topic_model_count[topic] += 1
            all_topics.add(topic)

print(f"Total unique topics: {len(all_topics)}")
print(f"Total models: {len(data)}\n")

# Check topics that don't appear in all models
print("Topics by model coverage:")
for topic in sorted(all_topics):
    count = topic_model_count[topic]
    if count < len(data):
        print(f"  {topic}: {count}/{len(data)} models [INCOMPLETE]")
    else:
        print(f"  {topic}: {count}/{len(data)} models [OK]")

# Check if there are any topics that appear in fewer than all models
incomplete_topics = {t: c for t, c in topic_model_count.items() if c < len(data)}
if incomplete_topics:
    print(f"\n[WARNING] Topics not in all models: {incomplete_topics}")
else:
    print("\n[OK] All topics appear in all models")
