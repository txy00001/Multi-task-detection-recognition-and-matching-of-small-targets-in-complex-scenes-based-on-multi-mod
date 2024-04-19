# Ultralytics YOLO 🚀, AGPL-3.0 license

from wiring_inspect.utils import DictNoDefault
from wiring_inspect.tracker.bot_sort import BOTSORT
from wiring_inspect.tracker.byte_tracker import BYTETracker

from loguru import logger
import numpy as np
import torch
from typing import Tuple
from ultralytics.engine.results import Results
import yaml


# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


class Tracker:
    ''' 当前只支持单帧追踪 '''
    def __init__(self, cfg_file_path: str = 'cfg/botsort.yaml'):
        # 读取配置文件
        with open(cfg_file_path, 'r') as fr:
            cfg = DictNoDefault(yaml.safe_load(fr))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            logger.error(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

        self.cfg =cfg

        self.initialize_tracker()

    def initialize_tracker(self):
        # 初始化 tracker
        self.tracker = TRACKER_MAP[self.cfg.tracker_type](args=self.cfg, frame_rate=30)

    def __call__(self, frame: np.ndarray, results: Results, persist: bool = True, is_obb: bool = False) -> Tuple[np.ndarray, Results]:
        # 获取目标检测框
        det = (results.obb if is_obb else results.boxes).cpu().numpy()

        if len(det) == 0:
            return results

        tracks = self.tracker.update(det, frame)
        if len(tracks) == 0:
            return results

        idx = tracks[:, -1].astype(int)
        results = results[idx]

        update_args = dict()
        update_args["obb" if is_obb else "boxes"] = torch.as_tensor(tracks[:, :-1])
        results.update(**update_args)
        return idx, results