"""extract embeddings for sound action datasets
modified from test.py
"""
import os
import sys
import tqdm
import torch
import logging
import argparse
import collections

sys.path.append("..")

from sacred import Experiment

from everything_at_once import data_loader as module_data
from everything_at_once import model as module_arch

from everything_at_once.metric import RetrievalMetric
from everything_at_once.trainer import eval
from everything_at_once.trainer.clip_utils import _apply_clip_text_model
from everything_at_once.trainer.utils import (
    short_verbose,
    verbose,
    format_dataloader_output,
    _move_to_device,
)
from everything_at_once.utils.util import state_dict_data_parallel_fix

from parse_config import ConfigParser


ex = Experiment("extract soundactions")


@ex.main
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    if config["trainer"].get("use_clip_text_model", False):
        import clip

        clip_text_model, _ = clip.load("ViT-B/32", device=device)
        clip_text_model.eval()
    else:
        clip_text_model = None
    model = config.initialize("arch", module_arch)

    # setup data_loader instances
    # logging.debug(f"config: {config['data_loader']}")
    # data_loader = config.initialize("data_loader", module_data)
    data_loader = module_data.FeatureDataloader(
        "SoundActions",
        batch_size=1,
        dataset_kwargs={
            "word2vec_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/GoogleNews/GoogleNews-vectors-negative300.bin",
            "video_feature_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/video_features",
            "audio_feature_path": "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/audio_features",
            "use_2D": True,
            "use_3D": True,
            "debug": False,
        },
    )
    logging.debug(f"dataloader size: {len(data_loader)}")

    logging.info(f"Loading model")
    checkpoint = torch.load(config.resume)
    epoch = checkpoint["epoch"]
    state_dict = checkpoint["state_dict"]
    new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
    model.load_state_dict(new_state_dict, strict=True)
    logging.info(f"Model loaded")

    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model
    model = model.to(device)
    model.eval()

    # run through data to get embeddings
    # embed_arr = collections.defaultdict(lambda: [])
    logging.info(f"Extracting embeddings")
    output_dir = "/fp/homes01/u01/ec-jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions/embeddings"
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for data in tqdm.tqdm(data_loader):
            data = format_dataloader_output(data)

            if clip_text_model is not None:
                data = _apply_clip_text_model(clip_text_model, data, device)

            for field in [
                "text",
                "text_mask",
                "video",
                "video_mask",
                "audio",
                "audio_mask",
                "audio_STFT_nframes",
                "caption",
                "image",
            ]:
                if field in data:
                    data[field] = _move_to_device(data[field], device)

            # get embeddings
            embeds = model(data, force_cross_modal=True)
            torch.save(embeds, os.path.join(output_dir, f"{data['meta']['ids'][0]}.pt"))


if __name__ == "__main__":
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
        "-c", "--config", type=str, default="./config.yaml", help="config file path"
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--n_gpu"], type=int, target=("n_gpu",)),
    ]
    config = ConfigParser(args, options, test=True)
    args = args.parse_args()
    ex.add_config(config.config)

    ex.run(options={"--loglevel": "DEBUG"})

