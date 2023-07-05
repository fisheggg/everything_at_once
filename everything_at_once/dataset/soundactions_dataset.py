"""Custom loader for SoundAction dataset. Modified from youcook_dataset.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import glob
import torch
import pickle
import logging
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from everything_at_once.dataset.utils import _tokenize_text, create_audio_features, create_text_features, \
    create_video_features, cut_into_clips

class SoundActionsDataset(Dataset):
    def __init__(
        self,
        we,
        video_feature_path=None,
        audio_feature_path=None,
        we_dim=300,
        max_words=20,
        n_video_tokens=12,
        num_audio_STFT_frames=768,
        video_sampling_strategy='clip',
        cut_clips=False,
        n_clips=1,
        use_2D=True,
        use_3D=True,
        debug=False,
    ):
        self.audio_data = glob.glob(os.path.join(audio_feature_path, '*.npy'))
        self.video_data_2d = None
        self.video_data_3d = None
        if use_2D:
            self.video_data_2d = glob.glob(os.path.join(video_feature_path, '*_2d.npy'))
            assert len(self.audio_data) == len(self.video_data_2d)
        if use_3D:
            self.video_data_3d = glob.glob(os.path.join(video_feature_path, '*_3d.npy'))
            assert len(self.audio_data) == len(self.video_data_3d)

        self.text = []
        self.id = []
        for data in self.audio_data:
            self.text.append(data.split(" - ")[-1].split("_hd")[0])
            self.id.append(data.split(" - ")[0].split("_")[-1])

        if debug:
            logging.info("Debug mode: using only 2 samples.")
            self.audio_data = self.audio_data[:2]
            if self.video_data_2d is not None:
                self.video_data_2d = self.video_data_2d[:2]
            if self.video_data_3d is not None:
                self.video_data_3d = self.video_data_3d[:2]
            self.text = self.text[:2]
            self.id = self.id[:2]

        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words

        self.num_audio_STFT_frames = num_audio_STFT_frames
        self.n_video_tokens = n_video_tokens
        self.video_sampling_strategy = video_sampling_strategy
        self.cut_clips = cut_clips
        self.n_clips = n_clips
        self.use_2d = use_2D
        self.use_3d = use_3D

        if not self.cut_clips:
            assert n_clips == 1

        if self.cut_clips:
            assert video_sampling_strategy == 'clip'

        logging.info(f"Found {len(self.audio_data)} samples in SoundActions dataset.")

    def __len__(self):
        return len(self.audio_data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def __getitem__(self, idx):
        # load 2d and 3d features
        feat_2d = None
        feat_3d = None
        if self.use_2d:
            feat_2d = torch.from_numpy(np.load(self.video_data_2d[idx]))
        if self.use_3d:
            feat_3d = torch.from_numpy(np.load(self.video_data_3d[idx]))
        target_nvideo_tokens = self.n_video_tokens * self.n_clips
        video, video_mask = create_video_features(
            feat_2d, feat_3d, target_nvideo_tokens,
            strategy=self.video_sampling_strategy,
            )

        # load audio features
        feat_audio = torch.from_numpy(np.load(self.audio_data[idx]))
        max_audio_STFT_nframes = self.num_audio_STFT_frames * self.n_clips
        audio, audio_mask, audio_STFT_nframes = create_audio_features(feat_audio, max_audio_STFT_nframes)
        
        # load text
        caption = self.text[idx]
        words = _tokenize_text(caption)
        text, text_mask, raw_text = create_text_features(words, self.max_words, self.we, self.we_dim)

        id_ = self.id[idx]
        dataset = 'SoundActions'

        if self.cut_clips:
            video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset = \
                cut_into_clips(video, video_mask, audio, audio_mask, text, text_mask, raw_text, audio_STFT_nframes, id_, dataset,
                               n_clips=self.n_clips)
            unroll_clips = 1
        else:
            unroll_clips = 0

        return {'video': video, 'audio': audio, 'text': text, 'audio_STFT_nframes': audio_STFT_nframes,
                'video_mask': video_mask, 'audio_mask': audio_mask, 'text_mask': text_mask,
                'raw_text': raw_text,
                'unroll_clips': unroll_clips,
                'meta': {'paths': id_, 'ids': id_, 'dataset': dataset}}
