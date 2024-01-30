"""Dataset loader for the IAW dataset."""
import json
import os
from typing import Literal

import torch
import torch.nn.functional as F
from core.config import config
from core.eval import iou
from torch.utils import data

from . import average_to_fixed_length

IAW_DATASET_PATH = "/data/jiahao/IKEAAssemblyInstructionDataset/dataset"

IAW_DATASET_JSON_MAP = {
    "train": "split/train_v2.2.1.json",
    "val": "split/val_v2.2.1.json",
    "test": "split/test_v2.2.1.json",
}

IAW_VIDEO_FEATURES_MAP = {
    "train": "feature/aligned/video/train_video_videomaev2_vit_g_resize_768_wwa9kb75.pt",
    "val": "feature/aligned/video/val_video_videomaev2_vit_g_resize_768_wwa9kb75.pt",
    "test": "feature/aligned/video/test_video_videomaev2_vit_g_resize_768_wwa9kb75.pt",
}

IAW_DIAGRAM_FEATURES_PATHNAME = "feature/aligned/image/image_dino_v2_vit_g_14_768_wwa9kb75.pt"


class IAW(data.Dataset):
    def __init__(self, split: Literal["train", "val", "test"]):
        super().__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        dataset = json.load(open(os.path.join(IAW_DATASET_PATH, IAW_DATASET_JSON_MAP[split])))
        annotation = []
        for furniture in dataset:
            furniture_id = furniture["id"]
            for video in furniture["videoList"]:
                video_id = video["url"].split("/watch?v=")[-1]
                for i, a in enumerate(video["annotation"]):
                    video_key = f"{furniture_id}_{video_id}"
                    key = f"{furniture_id}_{video_id}_{i}"
                    annotation.append(
                        {
                            "key": key,
                            "video_key": video_key,
                            "duration": video["duration"],
                            "times": [a["start"], a["end"]],
                            "action": a["action"],  # int, action id
                        }
                    )
        self.annotations = annotation
        self.video_features = torch.load(
            os.path.join(IAW_DATASET_PATH, IAW_VIDEO_FEATURES_MAP[split]), map_location="cpu"
        )
        self.diagram_features = torch.load(
            os.path.join(IAW_DATASET_PATH, IAW_DIAGRAM_FEATURES_PATHNAME), map_location="cpu"
        )

    def __getitem__(self, index: int):
        video_key = self.annotations[index]["video_key"]
        gt_s_time, gt_e_time = self.annotations[index]["times"]
        action_id = self.annotations[index]["action"]
        duration = self.annotations[index]["duration"]

        word_vectors = self.diagram_features[video_key.split("_")[0]][action_id]

        visual_input, visual_mask = self.get_video_features(video_key)
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
            "visual_input": visual_input,
            "vis_mask": visual_mask,
            "anno_idx": index,
            "word_vectors": word_vectors,
            "duration": duration,
            "txt_mask": torch.ones(word_vectors.shape[0], 1),
            "map_gt": torch.from_numpy(overlaps),
        }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, key: str):
        features = self.video_features[key]
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask


if __name__ == "__main__":
    import tqdm

    iaw = IAW("train")
    for i in tqdm.tqdm(iaw):
        pass
