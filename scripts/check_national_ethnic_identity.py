#!/usr/bin/env python
"""Check national_ethnic_identity topic."""
import json
from pathlib import Path

data_file = Path(__file__).parent.parent / 'analysis' / 'disaggregated' / 'by_topic_tag.json'
data = json.load(open(data_file))

topic = 'national_ethnic_identity'
found = any(topic in m_data for m_data in data.values())

print(f'Topic "{topic}" found in disaggregated data: {found}')

if found:
    print('\nFound in models:')
    for model in data:
        if topic in data[model]:
            n = data[model][topic].get('n', 0)
            acc = data[model][topic].get('accuracy', 'N/A')
            print(f'  {model}: n={n}, accuracy={acc}')
else:
    print('\nNot found in any model')
    print('This topic likely has n < min_n_tag (50) threshold and was filtered out during aggregation.')
