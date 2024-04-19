from functools import partial
from loguru import logger
from pathlib import Path
import pickle
from queue import Queue
import subprocess
import time
from threading import Thread
import typing as T

import cv2
import mmcv
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes, Masks

from wiring_inspect.preprocess import PREPROCESS_MAP
from wiring_inspect.matcher import wiring_match, WiringMatch
from wiring_inspect.schema import InspectItem
from wiring_inspect.tracker.track import Tracker


def convert_to_h264(input_video_path: str, output_video_path: str):
    # 运行ffmpeg需要重置环境变量，不然会和conda内装的ffmpeg冲突报错
    # command = f"LD_LIBRARY_PATH=/lib/x86_64-linux-gnu /usr/bin/ffmpeg -i {input_video_path} -vcodec h264_nvenc -y {output_video_path}"
    command = f"LD_LIBRARY_PATH=/usr/local/lib /usr/local/bin/ffmpeg -i {input_video_path} -vcodec h264_nvenc -y {output_video_path}"
    p = subprocess.Popen(command, shell=True)
    p.wait()  # 等待进程结束，避免json已经生成但视频还没转换完的情况


class VideoOutputWorker(Thread):
    def __init__(
        self, video_path: str, fps: float, width: int, height: int, queue: Queue
    ):
        self._video_path = video_path
        self._fps = fps
        self._width = width
        self._height = height
        self._queue = queue
        super().__init__(None)

    def run(self):
        # TODO 收集线程执行日志
        tmp_vid_path = f"/tmp/{Path(self._video_path).stem}-tmp.mp4"
        out_vid = cv2.VideoWriter(
            tmp_vid_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (self._width, self._height),
        )

        while True:
            frame = self._queue.get()
            if frame is None:
                out_vid.release()
                # 因为opencv默认不支持h264编码，所以这里需要转换一下
                convert_to_h264(tmp_vid_path, self._video_path)
                break
            out_vid.write(frame)


def update_coordinate(detect_region_lt_point: T.Tuple[int, int], terminal_config: T.Dict) -> T.Dict:
    '''
    从原图中裁剪出检测区域后，在原图上标注的铭牌和引脚坐标数据需要作相应的更新处理。
    
    TODO：考虑这一步在生成器件配置文件时做（只需做一次），还是在此处做（每次调用 inspect 时都要做一次）
    '''
    if 'detect' in terminal_config:
        return terminal_config

    x1, y1 = detect_region_lt_point

    for pattern in terminal_config['infer'].values():
        # 更新铭牌区域坐标
        pattern['body'][0] -= x1
        pattern['body'][1] -= y1
        pattern['body'][2] -= x1
        pattern['body'][3] -= y1

        # 更新引脚中心点坐标
        for terminal in pattern['terminals'].values():
            terminal['center'][0] -= x1
            terminal['center'][1] -= y1

    return terminal_config


def get_ocr_preprocess(detect_config: T.Dict, loc_position: str, terminal_config: T.Dict) -> T.Tuple:
    '''
    获取把图像喂给 ocr 模型之前的预处理。包含对 mask 和 crop 数据的预处理。
    对 mask 数据的预处理：mask 数据是从原图中获取分割的mask，预处理主要有 dilate；
    对 crop 数据的预处理：crop 数据是从原图中获取检测的目标，预处理主要有 旋转，
        因为位置标签和引线标签的字符可能不是水平方向，所以需要旋转。
        旋转角度根据位置标签的相对位置，引脚与引线的角度确定。
    '''
    # mask 预处理
    mask_processes: T.Dict[str, T.Callable] = {}
    for device_type, process_names in detect_config['mask_processes'].items():
        mask_processes[device_type] = [
            PREPROCESS_MAP[proc] for proc in process_names
        ]

    # 位置标签 旋转
    crop_processes: T.Dict[str, T.Callable] = {}
    if 'left' == loc_position:
        crop_processes['loc_marker'] = partial(PREPROCESS_MAP['rotate'], angle=270)

    # 引线标签 旋转
    if 'detect' in terminal_config:
        angle = terminal_config['detect']['angle']
    else:
        for pattern in terminal_config['infer'].values():
            # 对同一种器件，至少有一个引脚，其引脚与引线的角度是一致的
            terminal = list(pattern['terminals'].values())[0]
            angle = terminal['angle']
            break

    if angle > 0:
        angle = 360 - angle
    elif angle < 0:
        angle = 180 - angle
    else:
        pass

    if angle:
        crop_processes['line_marker'] = partial(PREPROCESS_MAP['rotate'], angle=angle)

    return (mask_processes, crop_processes)


def generate_item_crops(
    origin_img: np.ndarray,
    items: T.List[InspectItem],
    mask_processes: T.Dict[str, T.List[T.Callable]],
    crop_processes: T.Dict[str, T.List[T.Callable]],
) -> T.List[Image.Image]:
    """
    生成图片切片

    Args:
        origin_img (np.ndarray): 原始图片
        items (T.List[InspectItem]): 检测到的目标物体信息
        mask_processes (T.Dict[str, T.List[T.Callable]]): 生成mask之后执行的操作，比如膨胀
        crop_processes (T.Dict[str, T.List[T.Callable]]): 切片之后的操作，比如旋转

    Returns:
        List[Image]: 切片列表
    """
    # beg = time.time()
    # h, w, _ = origin_img.shape
    # mask_img = Image.new("L", (w, h), 0)
    # draw = ImageDraw.Draw(mask_img)
    # for item in items:
    #     if item.mask_xy.shape[1] < 2:
    #         print("*** item mask xy shape: ", item.mask_xy.shape, item.box_xyxy, item.target_type, "Skip.")
    #         continue
    #     try:
    #         draw.polygon(item.mask_xy, fill=255)
    #     except Exception as e:
    #         print(e)
    #         print("*** item mask xy shape: ", item.mask_xy.shape, item.box_xyxy, item.target_type, "Skip.")
    # draw_time = time.time() - beg

    # beg = time.time()
    # all_masks = [mask_img.crop(item.box_xyxy.int().tolist()) for item in items]

    # beg = time.time()
    # all_masks = []
    # for item in items:
    #     x1, y1, x2, y2 = item.box_xyxy.int().tolist()
    #     w, h = x2 - x1, y2 - y1
    #     mask = Image.new("L", (w, h), 0)

    #     if item.mask_xy.shape[1] < 2:
    #         logger.info(
    #             f"*** item mask xy shape: {item.mask_xy.shape}, box: {item.box_xyxy}, target_type: {item.target_type} Skip"
    #         )
    #         all_masks.append(mask)
    #         continue

    #     mask_draw = ImageDraw.Draw(mask)
    #     mask_rel_xy = item.mask_xy - item.box_xyxy[:2].numpy()
    #     mask_draw.polygon(mask_rel_xy, fill=255)
    #     all_masks.append(mask)
    # draw_time = time.time() - beg

    # beg = time.time()
    # processed_masks = []
    # for mask, item in zip(all_masks, items):
    #     processed_mask = mask
    #     if item.target_type in mask_processes:
    #         for process in mask_processes[item.target_type]:
    #             processed_mask = process(processed_mask)
    #     processed_masks.append(processed_mask)
    # mask_time = time.time() - beg

    # beg = time.time()
    # origin_pil = Image.fromarray(origin_img)

    # all_crops = [origin_pil.crop(item.box_xyxy.int().tolist()) for item in items]
    # all_masked_crops = [
    #     Image.composite(o, Image.new("RGB", o.size, (0, 0, 0)), m)
    #     for o, m in zip(all_crops, processed_masks)
    # ]

    # processed_crops = []
    # for crop, item in zip(all_masked_crops, items):
    #     processed_crop = crop
    #     if item.target_type in crop_processes:
    #         for process in crop_processes[item.target_type]:
    #             processed_crop = process(processed_crop)
    #     processed_crops.append(processed_crop)
    # crop_time = time.time() - beg
    # logger.info(
    #     f"draw_time: {draw_time:.4f} mask_time: {mask_time:.4f} crop_time: {crop_time:.4f}"
    # )

    gen_beg = time.time()
    origin_pil = Image.fromarray(origin_img)
    processed_crops = []
    for item in items:
        # 目标框的坐标
        x1, y1, x2, y2 = item.box_xyxy.astype(np.int64).tolist()
        w, h = x2 - x1, y2 - y1

        if item.mask_xy.shape[1] < 2:
            logger.info(
                f"*** item mask xy shape: {item.mask_xy.shape}, box: {item.box_xyxy}, target_type: {item.target_type} Skip"
            )
            processed_crops.append(Image.new("RGB", (w,h), (0, 0, 0)))
            continue

        # 根据目标的分割结果，生成 mask
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_rel_xy = item.mask_xy - item.box_xyxy[:2]
        mask_draw.polygon(mask_rel_xy, fill=255)

        # mask 预处理
        target_type = item.target_type
        if target_type in mask_processes:
            for process in mask_processes[target_type]:
                mask = process(mask)

        # 根据目标框坐标，从图像中取出目标数据，作为 crop
        crop = origin_pil.crop((x1, y1, x2, y2))

        # 根据 mask 获取目标数据
        masked_crop = Image.composite(crop, Image.new("RGB", crop.size, (0,0,0)), mask)

        # masked_crop 预处理
        if target_type in crop_processes:
            masked_crop = crop_processes[target_type](masked_crop)

        processed_crops.append(masked_crop)

    gen_time = time.time() - gen_beg
    logger.info(f"Generate crop time used: {gen_time:.4f} seconds.")

    return processed_crops


##初始化ocr
def init_ocr_model(
    det_model_dir="models/ocr/24_0118/det", rec_model_dir="models/ocr/24_0118/rec"
) -> PaddleOCR:
    ocr_model = PaddleOCR(
        use_angle_cls=False,
        # lang="en",  # enable: 24_0118, disable: ocr_impove
        det_model_dir=det_model_dir,  ##sim版
        rec_model_dir=rec_model_dir,
        det_limit_side_len=3840,
        # use_tensorrt=True,
        precision="int8",
        max_batch_size=50,
        use_mp=True,  ###根据情况可关闭，
        # total_process_num=2, ##进程数
        min_subgraph_size=10,
        use_dilation=True,  ###db膨胀，可关闭
        # process_id=0,
        # rec_char_dict_path="fonts/en_dict.txt",  # enable: 24_0118, disable: ocr_impove
        # rec_image_shape="3,48,256",  # enable: 24_0118, disable: ocr_impove 
        rec_batch_num=8,
        use_static=True,  ##在初次运行程序的时候会将 TensorRT 的优化信息进行序列化到磁盘上，
        ##下次运行时直接加载优化的序列化信息而不需要重新生成。
    )

    return ocr_model


class VideoInspector:
    def __init__(
        self,
        insepctor_cfg: T.Dict,
        output_dir: str,
    ):
        self._ocr_config = insepctor_cfg.ocr
        self._common_seg_config = insepctor_cfg.seg
        self._track_config = insepctor_cfg.track_config
        self._frame_sample_rate = insepctor_cfg.frame_sample_rate
        self._max_item_per_batch = insepctor_cfg.max_item_per_batch
        self._score_thresh = insepctor_cfg.score_thresh
        self._track_count_thresh = insepctor_cfg.track_count_thresh
        self._txt_count_thresh = insepctor_cfg.txt_count_thresh
        self.concat_cls_idx_name = insepctor_cfg.concat_cls_idx_name
        self.concat_name_cls_idx = {v: k for k, v in self.concat_cls_idx_name.items()}

        # 读取 ocr 模型
        self._ocr_model = init_ocr_model(**self._ocr_config)

        # 读取通用分割模型
        self.common_seg_model = {}
        for model_type, model_file_path in self._common_seg_config.items():
            self.common_seg_model[model_type] = YOLO(model_file_path)

        self._output_dir = output_dir

    def process_frame(
        self,
        frame_idx: int,
        img: np.ndarray,
        seg_model_dict: T.Dict[str, YOLO],
        tracker: Tracker,
        mask_processes: T.Dict[str, T.List[T.Callable]],
        crop_processes: T.Dict[str, T.List[T.Callable]],
        plot: bool = False,
    ):
        visual_res = None
        if plot:
            visual_res = img.copy()

        # items: T.List[InspectItem] = []
        # # 1. 跑分割模型
        # track_part_beg = time.time()
        # for target_type, model in seg_model_dict.items():
        #     track_beg = time.time()
        #     results: Results = model.track(
        #         img, persist=True, tracker="cfg/botsort.yaml"
        #     )[
        #         0
        #     ]  # 只传入一帧数据，所以默认只需要读取第0个结果
        #     track_time = time.time() - track_beg

        #     if plot:
        #         visual_res = results.plot(img=visual_res)

        #     if results.boxes is None or results.masks is None:
        #         logger.info(
        #             f"Frame {frame_idx} track result has None, boxes: {results.boxes}, masks: {results.masks}",
        #         )
        #         continue

        #     items_beg = time.time()
        #     for box, mask in zip(results.boxes, results.masks):
        #         # 根据这个问题（https://github.com/ultralytics/ultralytics/issues/3830#issuecomment-1642651960），
        #         # track_id 可能不会在第一次检测到物体就分配，所以这里需要判断一下
        #         mask_xy = mask.xy[0]
        #         if box.id is not None and mask_xy.shape[0] > 0:
        #             item = InspectItem(
        #                 frame_idx, target_type, box=box.cpu().numpy(), mask=mask_xy
        #             )
        #             items.append(item)
        #     items_time = time.time() - items_beg
        #     logger.info(f'{target_type}, track_time: {track_time:.4f}, items_time: {items_time:.4f}')
        # track_part_time = time.time() - track_part_beg

        items: T.List[InspectItem] = []
        # 1. 跑分割模型
        # 合并 results，再 track
        all_results: T.List[Results] = []
        target_types = []  # 所有的 target_types

        track_part_beg = time.time()
        for target_type, model in seg_model_dict.items():
            seg_beg = time.time()
            results: Results = model(img)[0]  # 只传入一帧数据，所以默认只需要读取第0个结果
            seg_time = time.time() - seg_beg

            # 给 results 添加 target_types 属性，便于后续生成 InspectItem
            target_types.extend([target_type] * len(results))
            all_results.append(results)

            # logger.info(f'frame idx: {frame_idx}, {target_type}, seg time used: {seg_time}')

        # 合并 results
        cat_beg =  time.time()
        orig_img = None
        img_file_path = None
        boxes = []  # 所有的 boxes
        masks = []  # 所有的 masks
        for results_idx, results in enumerate(all_results):
            if 0 == results_idx:
                orig_img = results.orig_img
                img_file_path = results.path

            # 没有检测到物体
            if results.boxes is None or results.masks is None:
                # logger.info(
                #     f"Frame {frame_idx} track result has None, boxes: {results.boxes}, masks: {results.masks}",
                # )
                continue

            # 改变类别为合并后类别
            box_data = results.boxes.cpu().data
            new_cls_idx = [self.concat_name_cls_idx[results.names[int(cls_idx.item())]] for cls_idx in box_data[:, -1]]
            # 更新合并后类别
            box_data[:, -1] = torch.tensor(new_cls_idx, dtype=torch.float)

            # 所有的 boxes, masks, target_types
            boxes.append(box_data)
            masks.append(results.masks.cpu().data)

        if len(boxes):
            boxes = torch.cat(boxes, dim=0)
            masks = torch.cat(masks, dim=0)

            concat_results = Results(orig_img, img_file_path, self.concat_cls_idx_name, boxes=boxes, masks=masks)

            cat_time = time.time() - cat_beg

            # track
            track_beg_time = time.time()
            track_idx, concat_results = tracker(img, concat_results, persist=True)
            track_target_types = [target_types[t_idx] for t_idx in track_idx]
            track_time = time.time() - track_beg_time

            if plot:
                visual_res = concat_results.plot(img=visual_res)

            # 生成 InspectItem
            items_beg = time.time()
            for target_type, box, mask in zip(track_target_types, concat_results.boxes, concat_results.masks):
                # 根据这个问题（https://github.com/ultralytics/ultralytics/issues/3830#issuecomment-1642651960），
                # track_id 可能不会在第一次检测到物体就分配，所以这里需要判断一下
                mask_xy = mask.xy[0]
                if box.id is not None and mask_xy.shape[0] > 0:
                    item = InspectItem(
                        frame_idx, target_type, box=box.numpy(), mask=mask_xy
                    )
                    items.append(item)
            items_time = time.time() - items_beg
            logger.info(f'frame idx: {frame_idx}, cat time: {cat_time: .4f}, track_time: {track_time:.4f}, items_time: {items_time:.4f}')
        track_part_time = time.time() - track_part_beg

        # 2. 处理分割结果，拼接mask
        if 0 == len(items):
            logger.info(f"!! Frame {frame_idx} has nothing detected, track part time used: {track_part_time:.4f} seconds")
        else:
            gen_crop_beg = time.time()
            all_crops = generate_item_crops(
                img, items, mask_processes=mask_processes, crop_processes=crop_processes
            )
            gen_crops_time = time.time() - gen_crop_beg

            # 如果一次性要跑的ocr的切片太多，会导致内存不够，所以这里需要分批跑
            # # ocr_results 保存字符识别结果 [ [box, (text, score)] ]
            # # box: T.List[T.List[int]] 4x2 的 2维列表，表示字符框的4个顶点坐标 (x, y)，从左上角起以顺时针顺序排列
            # # text: str 检测到的字符， score: float 字符的置信度
            ocr_results: T.List[T.List[T.List[T.List[int]], T.Tuple[str, float]]] = []
            # # 存放每个批次每个 mask 图像的宽高
            all_box_shapes: T.List[np.ndarray] = []
            cum_canvas_height = 0

            step_num = len(all_crops) // self._max_item_per_batch + 1
            logger.info(f'Run ocr in {step_num} steps, crops num: {len(all_crops)}!')
            for step_idx in range(step_num):
                ocr_step_beg = time.time()
                step_crops = all_crops[
                    step_idx * self._max_item_per_batch : (step_idx + 1) * self._max_item_per_batch
                ]

                if step_crops:
                    # 生成拼接画布，以所有分割框的最大宽度为宽，以所有分割框的高之和为高
                    merge_beg = time.time()
                    step_box_shapes = np.array([crop.size for crop in step_crops])
                    all_box_shapes.append(step_box_shapes)

                    canvas_width = np.max(step_box_shapes[:, 0])
                    canvas_height = np.sum(step_box_shapes[:, 1])
                    canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)

                    cum_height = 0
                    for crop in step_crops:
                        canvas[
                            cum_height : (cum_height + crop.height), : crop.width, :
                        ] = np.asarray(crop)
                        cum_height += crop.height
                    # cv2.imwrite(f"merge_items_{frame_idx}.jpg", canvas)
                    merge_time = time.time() - merge_beg

                    # 3. 跑ocr模型
                    ocr_beg = time.time()
                    step_ocr_results = self._ocr_model.ocr(canvas, cls=False)
                    if len(step_ocr_results):
                        step_ocr_results = step_ocr_results[0]  # 只传入一帧数据，所以默认只需要读取第0个结果

                    if not step_ocr_results:  # 第0个结果可能为 None
                        step_ocr_results = []

                    if len(step_ocr_results):
                        for res in step_ocr_results:  # res: [box, (text, score)]
                            for pi in range(len(res[0])):
                                res[0][pi][1] += cum_canvas_height

                    ocr_results.extend(step_ocr_results)
                    ocr_time = time.time() - ocr_beg
                    cum_canvas_height += canvas_height

                    logger.info(
                        f"OCR Step {step_idx + 1}: merge time - {merge_time:.4f}, ocr time - {ocr_time:.4f}, step time - {time.time() - ocr_step_beg:.4f}"
                    )
            all_box_shapes = np.vstack(all_box_shapes)

            # 4. 根据检测结果，将ocr结果填充到InspectItem中
            if 0 == len(ocr_results):
                logger.info(
                    f"frame_idx: {frame_idx}, track part time used: {track_part_time:.4f}, gen_crops_time: {gen_crops_time:.4f}, ocr part time: {time.time() - gen_crop_beg:.4f}"
                )
                return items, visual_res

            ocr_results = sorted(ocr_results, key=lambda x: x[0][0][1])  # 根据检测到的文本框第一个point的Y轴进行排序

            current_idx = 0
            current_height_top = all_box_shapes[current_idx][1]
            for points, (text, score) in ocr_results:
                if (
                    points[0][1] < current_height_top
                ):  # 如果文本的Y轴坐标小于当前切片所在坐标的Y轴坐标，则说明这个文本框属于当前切片
                    items[current_idx].text_boxes.append(np.array(points))
                    items[current_idx].text_words.append(text)
                    items[current_idx].text_scores.append(score)
                else:
                    while (
                        points[0][1] > current_height_top
                    ):  # 如果文本Y轴坐标大于当前坐标，需要一直往下找，直到找到一个切片，其Y轴大于文本Y轴坐标
                        current_idx += 1
                        current_height_top += all_box_shapes[current_idx][1]
                    items[current_idx].text_boxes.append(np.array(points))
                    items[current_idx].text_words.append(text)
                    items[current_idx].text_scores.append(score)

            logger.info(
                f"frame_idx: {frame_idx}, track part time used: {track_part_time:.4f}, gen_crops_time: {gen_crops_time:.4f}, ocr part time: {time.time() - gen_crop_beg:.4f}"
            )
        return items, visual_res

    def inspect(
        self,
        video_path: str,
        device_info: T.Dict,
        detect_config: T.Dict,
        loc_position: str,
        nameplate_config: T.Dict,
        terminal_config: T.Dict,
        progress_callback: T.Callable = None,
        output_video: bool = False,
    ):
        # 读取分割模型
        seg_model_dict = {
            **self.common_seg_model,  # 通用分割模型
        }

        try:
            if isinstance(nameplate_config['detect'], dict):
                seg_model_dict[nameplate_config['detect']['nameplate_type']] = YOLO(nameplate_config['detect']['model_path'])

            if 'detect' in terminal_config:
                seg_model_dict[terminal_config['detect']['terminal_type']] = YOLO(terminal_config['detect']['model_path'])
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            raise e

        # 读取追踪模型
        # TODO: 注意 reset 或者 在函数结束的时候 del
        tracker = Tracker(cfg_file_path=self._track_config)

        vid = mmcv.VideoReader(video_path)

        frame_results: T.List[T.List[InspectItem]] = []

        if output_video:
            out_vid_path = f"{self._output_dir}/{Path(video_path).stem}-out.mp4"
            output_frame_queue = Queue()
            video_worker = VideoOutputWorker(
                out_vid_path,
                vid.fps // self._frame_sample_rate,
                vid.width,
                vid.height,
                output_frame_queue,
            )
            video_worker.start()

        # 监测区域
        x1, y1, x2, y2 = detect_config['detect_region']
        detect_width = x2 - x1

        # 获取 mask 和 crop 的预处理操作
        mask_processes, crop_processes = get_ocr_preprocess(detect_config, loc_position, terminal_config)

        progress_interval = vid.frame_cnt // 100
        for idx, frame in enumerate(vid):
            if idx % self._frame_sample_rate != 0:
                continue

            if (
                progress_callback is not None
                and progress_interval > 0
                and idx % progress_interval == 0
            ):
                progress_callback(idx / vid.frame_cnt)

            frame_result, visual_frame = self.process_frame(
                idx,
                frame[y1:y2, x1:x2, :],
                seg_model_dict,
                tracker,
                mask_processes,
                crop_processes,
                plot=output_video,
            )
            frame_results.append(frame_result)

            if output_video:
                win_width = detect_config["match_window_width"]
                win_left = (vid.width - win_width) // 2
                win_right = (vid.width + win_width) // 2
                line1 = [win_left, 0], [win_left, vid.height]
                line2 = [win_right, 0], [win_right, vid.height]
                cv2.line(visual_frame, line1[0], line1[1], (0, 255, 0), 3)
                cv2.line(visual_frame, line2[0], line2[1], (0, 255, 0), 3)
                output_frame_queue.put(visual_frame)

        vid.vcap.release()

        if output_video:
            output_frame_queue.put(None)

        with open(f"{self._output_dir}/{Path(video_path).stem}.pkl", "wb") as f:
            pickle.dump(frame_results, f)

        # with open(f"{self._output_dir}/{Path(video_path).stem}.pkl", "rb") as f:
        #     frame_results = pickle.load(f)

        # 基于检测区域，更新 terminal_config 中的 body 和 center
        terminal_config = update_coordinate((x1, y1), terminal_config)

        if detect_config["match_window_width"] >= detect_width:
            match_win_left, match_win_right = 0, detect_width
        else:
            match_win_left = (detect_width - detect_config["match_window_width"]) // 2
            match_win_right = (detect_width + detect_config["match_window_width"]) // 2

        wiring_match = WiringMatch(
            (match_win_left, match_win_right),
            nameplate_config,
            terminal_config,
            self._score_thresh,
            self._track_count_thresh,
            self._txt_count_thresh
        )
        match_results = wiring_match(device_info['device_type'], frame_results)

        if output_video:
            video_worker.join()

        return match_results