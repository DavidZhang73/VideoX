"""Dataset loader for the TACoS dataset."""
import json
import os

import h5py
import torch
import torch.nn.functional as F
import torchtext
from core.config import config
from core.eval import iou
from torch import nn
from torch.utils import data

from . import average_to_fixed_length


class TACoS(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(["<unk>"])
    vocab.stoi["<unk>"] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super().__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, f"{split}.json")) as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno["num_frames"] / video_anno["fps"]
            for timestamp, sentence in zip(video_anno["timestamps"], video_anno["sentences"]):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            "video": vid,  # video id, e.g. 's25-d52.avi'
                            "duration": duration,  # video duration in seconds
                            "times": [
                                max(timestamp[0] / video_anno["fps"], 0),  # start time in seconds
                                min(timestamp[1] / video_anno["fps"], duration),  # (end time in seconds)
                            ],
                            "description": sentence,  # str, description of the that segment
                        }
                    )
        self.annotations = anno_pairs

    def __getitem__(self, index):
        video_id = self.annotations[index]["video"]
        gt_s_time, gt_e_time = self.annotations[index]["times"]
        sentence = self.annotations[index]["description"]
        duration = self.annotations[index]["duration"]

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)

        # visual_input = sample_to_fixed_length(visual_input, random_sampling=config.DATASET.RANDOM_SAMPLING)
        visual_input = average_to_fixed_length(visual_input)
        num_clips = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
        overlaps = iou(
            torch.stack([s_times[:, None].expand(-1, num_clips), e_times[None, :].expand(num_clips, -1)], dim=2)
            .view(-1, 2)
            .tolist(),
            torch.tensor([gt_s_time, gt_e_time]).tolist(),
        ).reshape(num_clips, num_clips)

        item = {
            "visual_input": visual_input,  # 256, 4096
            "vis_mask": visual_mask,  # 448, 1
            "anno_idx": index,  # 7007
            "word_vectors": word_vectors,  # 9, 300
            "duration": duration,  # 184.625...
            "txt_mask": torch.ones(word_vectors.shape[0], 1),  # 9, 1
            "map_gt": torch.from_numpy(overlaps),  # 128, 128
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == "c3d"
        with h5py.File(os.path.join(self.data_dir, "tall_c3d_features.hdf5"), "r") as f:
            features = torch.from_numpy(f[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask


if __name__ == "__main__":
    tacos = TACoS()
