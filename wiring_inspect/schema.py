import numpy as np
import typing as T
from ultralytics.engine.results import Boxes


class InspectItem(object):
    """
    封装 ultralytics 和 paddleocr 的结果，每一帧里面的每一个检测结果都是一个 InspectItem

    Args:
        object (_type_): _description_
    """
    def __init__(self, frame_idx: int, target_type: str, **kwargs):
        self._box: Boxes = kwargs["box"] if "box" in kwargs else None
        self._mask_xy: np.ndarray = kwargs["mask"] if "mask" in kwargs else None

        self.text_boxes: T.List[np.ndarray] = []
        self.text_words: T.List[str] = []
        self.text_scores: T.List[float] = []
        self.frame_idx = frame_idx
        self.target_type = target_type
        self.updated_text = None

    @property
    def box_xyxy(self) -> np.ndarray:
        return self._box.xyxy[0]

    @property
    def box_xywh(self) -> np.ndarray:
        return self._box.xywh[0]

    @property
    def mask_xy(self) -> np.ndarray:
        return self._mask_xy

    @property
    def track_id(self) -> int:
        return int(self._box.id[0])

    @property
    def conf(self) -> float:
        return float(self._box.conf[0])

    @property
    def bottom_point(self):
        if self.mask_xy.any():
            bottom_point = self.mask_xy[0]
            for point in self.mask_xy:
                if point[1] > bottom_point[1]:
                    bottom_point = point
            return bottom_point

    @property
    def top_point(self):
        top_point = self.mask_xy[0]
        for point in self.mask_xy:
            if point[1] < top_point[1]:
                top_point = point
        return top_point

    def __repr__(self) -> str:
        return "InspectItem: frame_idx: {}, target_type: {}, box: {}, mask: {}, text_boxes: {}, text_words: {}, text_scores: {}".format(
            self.frame_idx, self.target_type, self._box, self._mask_xy, self.text_boxes, self.text_words, self.text_scores
        )