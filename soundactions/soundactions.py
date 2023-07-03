import sys
sys.path.append("..")

import logging

from everything_at_once.data_loader import FeatureDataloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    dataloader = FeatureDataloader(
        "SoundActions",
        dataset_kwargs={
            "word2vec_path": "/home/jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/GoogleNews/GoogleNews-vectors-negative300.bin",
            "video_feature_path": "/home/jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/video_features",
            "audio_feature_path": "/home/jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/audio_features",
            "use_2D": True,
            "use_3D": False,
        }
        )

    for data in dataloader:
        print(data.keys())
        print(f"video shape: f{data['video'].shape}")
        print(f"audio shape: f{data['audio'].shape}")
        print(f"text shape: f{data['text'].shape}")