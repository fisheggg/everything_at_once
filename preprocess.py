import os
import csv
import math
import glob
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess.AVLnet import audio_to_spectrograms as audio
from preprocess.video_feature_extractor.video_loader import VideoLoader
from preprocess.video_feature_extractor.random_sequence_shuffler import (
    RandomSequenceSampler,
)
from preprocess.video_feature_extractor.preprocessing import Preprocessing
from preprocess.video_feature_extractor.model import get_model


def process_soundaction(args):
    """
    process the sound action dataset in EAO format.
    """
    print(f"=> Start processing SoundActions")

    ### generate csv file
    videos = glob.glob(os.path.join(args.data_path, "*.mp4"))
    if args.debug:
        print(f"=> DEBUG MODE")
        videos = videos[:2]
    print(f"=> Found {len(videos)} mp4 files")
    videos = [os.path.abspath(video) for video in videos]
    features = [os.path.abspath(
        os.path.join(
            args.output_path, 
            os.path.split(video)[-1].replace(".mp4", ".npy").replace(".", f"_{args.type}.")
        )
    ) for video in videos]
    csv_path = os.path.join(args.output_path, os.path.split(args.data_path)[-1]+".csv")
    os.makedirs(args.output_path, exist_ok=True)
    with open(csv_path, "w") as f:
        write = csv.writer(f)
        write.writerow(["video_path", "feature_path"])
        write.writerows(zip(videos, features))

    print(f"=> csv file written to{csv_path}")

    ### process video
    # from https://github.com/roudimit/video_feature_extractor
    dataset = VideoLoader(
        csv_path,
        framerate=1 if args.type == "2d" else 24,
        size=224 if args.type == "2d" else 112,
        centercrop=(args.type == "3d"),
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )
    preprocess = Preprocessing(args.type)
    model = get_model(args)

    with torch.no_grad():
        for k, data in enumerate(loader):
            input_file = data["input"][0]
            output_file = data["output"][0]
            if len(data["video"].shape) > 3:
                print(
                    "Computing features of video {}/{}: {}".format(
                        k + 1, n_dataset, input_file
                    )
                )
                video = data["video"].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    features = torch.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * args.batch_size
                        max_ind = (i + 1) * args.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = model(video_batch)
                        if args.l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if args.half_precision:
                        features = features.astype("float16")
                    np.save(output_file, features)
            else:
                print("Video {} already processed.".format(input_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Easy video feature extractor")

    parser.add_argument(
        "--data_path",
        type=str,
        help="path for original video files",
        default="/home/jinyueg/felles/Research/Project/AMBIENT/Datasets/SoundActions/video-HD"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path for output csv file and feature npy files",
        default="/home/jinyueg/felles/Research/Users/jinyueg/SoundAction/everything_at_once/data/SoundActions"
    )

    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--type", type=str, default="2d", help="CNN type")
    parser.add_argument(
        "--half_precision", type=int, default=1, help="output half precision float"
    )
    parser.add_argument(
        "--num_decoding_thread",
        type=int,
        default=4,
        help="Num parallel thread for video decoding",
    )
    parser.add_argument(
        "--l2_normalize", type=int, default=1, help="l2 normalize feature"
    )
    parser.add_argument(
        "--resnext101_model_path",
        type=str,
        default="model/resnext101.pth",
        help="Resnext model path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()


    process_soundaction(args)
