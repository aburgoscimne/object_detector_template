import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from src.lightning_classes.lightning_wheat import LitWheat
from src.utils.get_dataset import get_test_dataset
from src.utils.utils import set_seed, format_prediction_string, collate_fn


def predict(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)

    test_dataset = get_test_dataset(cfg)
    path = r'C:/Users/alberto/Documents/GitHub/object_detector_template/outputs/2020_07_09_19_52_14/saved_models/epoch=33_main_score=0.2985.ckpt'

    model = LitWheat.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    detection_threshold = 0.5
    results = []

    for images, _, image_ids in test_loader:

        # images = (image.to(cfg.general.device) for image in images)
        outputs = model(images)

        for i, _ in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            result = {'image_id': image_id, 'PredictionString': format_prediction_string(boxes, scores)}

            results.append(result)


@hydra.main(config_path='conf/config.yaml')
def run_model(cfg: DictConfig) -> None:
    print(cfg.pretty())
    predict(cfg)


if __name__ == '__main__':
    run_model()
