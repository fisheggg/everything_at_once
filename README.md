# Study progress

- [x] SoundActions preprocessor
  - [x] Create csv file
  - [x] Get video features
  - [x] Get audio features
- [x] Sound Actions dataloader
  - data loader output:
  ```python
  dict_keys(['video', 'audio', 'text', 
             'audio_STFT_nframes', 'video_mask', 'audio_mask',
             'text_mask', 'raw_text', 'unroll_clips', 'meta'])
  video shape: torch.Size([1, 12, 2048])
  audio shape: torch.Size([1, 40, 768])
  text shape: torch.Size([1, 20, 300])
  ```