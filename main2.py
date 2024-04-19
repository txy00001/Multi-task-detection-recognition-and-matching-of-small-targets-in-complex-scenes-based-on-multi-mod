import redis
import yaml

from loguru import logger
import numpy as np
import json
from pathlib import Path
import time
import sys

from wiring_inspect.config import DeviceConfigRepo
from wiring_inspect.inspector import VideoInspector
from wiring_inspect.schema import InspectItem
from wiring_inspect.task import TaskManager
from wiring_inspect.utils import setup_program


class InspectItemJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, InspectItem):
            # box = obj.box_xyxy.numpy().astype(np.int64).tolist()
            box = obj.box_xyxy.astype(np.int64).tolist()
            mask = obj.mask_xy.astype(np.int64).tolist()
            return {
                "frame_idx": obj.frame_idx,
                "box": box,
                "mask": mask,
            }
        return json.JSONEncoder.default(self, obj)


class Int64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yml")
        sys.exit(1)

    config_file = sys.argv[1]
    config = setup_program(config_file)

    stream_config = config["redis_stream"]
    r = redis.Redis(
        host=stream_config["host"],
        port=stream_config["port"],
        password=stream_config["password"],
        decode_responses=True,
    )

    video_config = config["video_process"]
    task_listener = TaskManager(
        r,
        stream_config["task_stream"],
        stream_config["consumer_group"],
        stream_config["consumer_name"],
        stream_config["progress_stream"],
    )

    inspector_config = config['inspector_config']
    dev_conf_repo = DeviceConfigRepo(inspector_config["device_repository"])
    logger.info(f'device config: {dev_conf_repo._device_config}')

    video_inspector = VideoInspector(
        inspector_config, video_config["output_dir"]
    )

    while True:
        task = task_listener.get_task()
        logger.info("Got task: {}".format(task))

        task_device_conf = dev_conf_repo.get_config(task.type)
        if task_device_conf is None:
            task_listener.update_task(task, "ERROR", 1.0, f"Unsupported inspect type: {task.type}")
            logger.error(f"No device config for task: {task.type}")
            task_listener.update_task(task, "ERROR", 1.0, f"No device config for task: {task.type}")
            continue

        input_video = f"{video_config['input_dir']}/{task.name}.mp4"
        if not Path(input_video).exists():
            task_listener.update_task(task, "ERROR", 1.0, f"File {input_video} not exists.")
            continue

        beg = time.time()
        def progress_callback(progress: float):
            task_listener.update_task(task, "RUNNING", progress, "")

        result = video_inspector.inspect(
            input_video, 
            task_device_conf['device_info'],
            task_device_conf['detect_config'],
            task_device_conf['loc_position'],
            task_device_conf['nameplate_config'],
            task_device_conf['terminal_config'],
            progress_callback=progress_callback
        )
        logger.warning(result['outputs_match'])

        if result['outputs_match']:
            out_file = Path(video_config["output_dir"]) / f"{task.name}.json"
            with open(out_file, "w") as f:
                json.dump(result['outputs_match'], f, cls=Int64Encoder)
                
            track_file = Path(video_config["output_dir"]) / f"{task.name}.track.json"
            with open(track_file, "w") as f:
                json.dump(result['outputs_data'], f, cls=InspectItemJsonEncoder)

            logger.warning(
                f"finish to inspect: {input_video}, with inspect type: {task.type}, output: {out_file} time: {time.time() - beg:.2f}s",
            )

            # 将结果写入redis stream
            task_listener.update_task(task, "SUCCESS", 1.0, "")
        else:
            task_listener.update_task(task, "ERROR", 1.0, "No result.")
            logger.warning(f"finish to inspect: {input_video} with empty result")

        r.xack(stream_config["task_stream"], stream_config["consumer_group"], task.mid)
