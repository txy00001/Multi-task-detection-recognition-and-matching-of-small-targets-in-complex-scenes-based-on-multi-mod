import yaml
import redis

import sys
from pathlib import Path
import json
import time
import logging

from wiring_inspect.task import TaskManager
from wiring_inspect.main import INSPECT_TYPE_MAPPING, inspect_video, init_ocr_model
from wiring_inspect.inspector import VideoInspector

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.INFO)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yml")
        sys.exit(1)

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    stream_config = config["redis_stream"]
    r = redis.Redis(
        host=stream_config["host"],
        port=stream_config["port"],
        password=stream_config["password"],
        decode_responses=True,
    )

    ocr_model = init_ocr_model()
    video_config = config["video_process"]
    task_listener = TaskManager(
        r,
        stream_config["task_stream"],
        stream_config["consumer_group"],
        stream_config["consumer_name"],
        stream_config["progress_stream"],
    )
    
    # video_inspector = VideoInspector(2, config["inspector_config"]["ocr"], None, None, video_config["output_dir"])
    while True:
        task = task_listener.get_task()
        Logger.info("Got task: {}".format(task))

        beg = time.time()
        input_video = f"{video_config['input_dir']}/{task.name}.mp4"
        if task.type not in INSPECT_TYPE_MAPPING:
            Logger.info("Unsupported inspect type: ", task.type)
            task_listener.update_task(task, "ERROR", 1.0, f"Unsupported inspect type: {task.type}")
            continue
            
        if not Path(input_video).exists():
            task_listener.update_task(task, "ERROR", 1.0, f"File {input_video} not exists.")
            continue
        
        def progress_callback(progress: float):
            task_listener.update_task(task, "RUNNING", progress, "")
        
        result = inspect_video(
            input_video,
            INSPECT_TYPE_MAPPING[task.type],
            video_config["output_dir"],
            ocr_model=ocr_model,
            progress_callback=progress_callback,
        )
        # result = video_inspector.inspect(input_video, INSPECT_TYPE_MAPPING[task.type], progress_callback=progress_callback)
        # result = [{"test": "test"}]
        Logger.info(result)
        if result:
            out_file = Path(video_config["output_dir"]) / f"{task.name}.json"
            with open(out_file, "w") as f:
                json.dump(result, f)
            Logger.info(
                "finish to inspect: ",
                input_video,
                " with inspect type:",
                task.type,
                " output: ",
                out_file,
                f" time: {time.time() - beg:.2f}s",
            )
            # 将结果写入redis stream
            task_listener.update_task(task, "SUCCESS", 1.0, "")
        else:
            task_listener.update_task(task, "ERROR", 1.0, "No result.")
            Logger.info("finish to inspect: ", input_video, " with empty result")
            
        r.xack(stream_config["task_stream"], stream_config["consumer_group"], task.mid)
