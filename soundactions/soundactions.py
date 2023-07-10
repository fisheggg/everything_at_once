import sys

sys.path.append("..")

import logging
import argparse

from everything_at_once import data_loader
from parse_config import ConfigParser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # logging.debug("Directly loading dataloader")
    # dataloader = FeatureDataloader(
    #     "SoundActions",
    #     dataset_kwargs={
    #         "word2vec_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/GoogleNews/GoogleNews-vectors-negative300.bin",
    #         "video_feature_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/video_features",
    #         "audio_feature_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/audio_features",
    #         "use_2D": True,
    #         "use_3D": False,
    #         "debug": True,
    #     },
    # )

    # logging.debug(f"dataloader size: {len(dataloader)}")
    # for data in dataloader:
    #     print(data.keys())
    #     print(f"video shape: {data['video'].shape}")
    #     print(f"audio shape: {data['audio'].shape}")
    #     print(f"text shape: {data['text'].shape}")

    logging.debug("Loading dataloader from config")

    args = argparse.ArgumentParser(
        description="extract embeddings for sound action datasets"
    )
    args.add_argument(
        "-r",
        "--resume",
        type=str,
        default="../pretrained_models/everything_at_once_tva/latest_model.pth",
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-c", "--config",
        type=str,
        default="./config.yaml",
        help="config file path"
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser(args, test=True)
    print(config['data_loader'])

    dataloader = config.initialize("data_loader", data_loader)
    logging.debug(f"dataloader size: {len(dataloader)}")
    for data in dataloader:
        print(data.keys())
        print(f"video shape: {data['video'].shape}")
        print(f"audio shape: {data['audio'].shape}")
        print(f"text shape: {data['text'].shape}")
