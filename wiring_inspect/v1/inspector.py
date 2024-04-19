
# 每个器件，有个位置编号，有个器件类型，有个丝印编号（丝印编号可能没有，比如端子排），有接线引脚，每个引脚有可能有接线，也可能没有接线

[
    {
        "location": {
            "meta": {
                "detect_type": "loc_marker/loc_marker",
                "track_id": 125,
            },
            "value": "XT15"
        },
        "device": {
            "device_type": "breaker",
            "nameplate": {
                "meta": {
                    "detect_type": "breaker/breaker", # breaker/iof, breaker/ELM
                    "track_id": 2,
                },
                "value": "iC65N C4A"
            }, 
        },
        "connections": [
            {
                "terminal": {
                    "meta": {
                        "detect_type": "breaker/pin", # breaker的terminal不是深度学习检测出来的，而是根据位置信息计算出来的，对应的track里面也应该有这个信息
                        "track_id": 323
                    },
                    "value": "L1",
                },
                "wire": {
                    "meta": {
                        "detect_type": "line_marker/line_marker",
                        "track_id": 323
                    },
                    "value": "-XT15:1"
                }
            },
            # "pin1": {
            #     "meta": {
            #         "detect_type": "junction_box/junction_box", # 端子排的pin是检测出来的
            #         "track_id": 323
            #     },
            #     "value": "L1",
            # },  "connection": None
            # }
        ]
    }
]


class FrameMetaStore:
    """保存每一帧的检测结果，并且提供方便的查询接口，支持按帧号查询单帧，或者帧号范围查询多帧，并且可以根据器件ID过滤结果。
    """
    def __init__(self):
        pass
    
    def add(self, frame_idx, inspect_item):
        pass
    
    def get(self, frame_idx):
        pass
    
    def get_range(self, start_idx, end_idx, device_id=None):
        pass
    
    # 是否有器件通过整个视野
    def device_pass(self):
        pass


class VideoInspector:
    def __init__(self, config):
        self._config = config
    
    def inspect(self, video_path, task_config, ):
        pass