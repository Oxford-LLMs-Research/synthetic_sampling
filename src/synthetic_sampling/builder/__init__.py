"""
Dataset builder module for synthetic sampling pipeline.

Example usage:
    from synthetic_sampling.config import DataPaths, DatasetConfig, GeneratorConfig
    from synthetic_sampling.builder import DatasetBuilder
    
    paths = DataPaths(
        raw_data_dir='~/data/surveys',
        metadata_dir='./src/synthetic_sampling/profiles/metadata',
        output_dir='./outputs'
    )
    
    builder = DatasetBuilder(paths, DatasetConfig(), GeneratorConfig())
    instances = builder.build_dataset(['wvs', 'ess_wave_10'])
    builder.save_jsonl(instances, 'dataset.jsonl')
"""

from .builder import DatasetBuilder

__all__ = ['DatasetBuilder']