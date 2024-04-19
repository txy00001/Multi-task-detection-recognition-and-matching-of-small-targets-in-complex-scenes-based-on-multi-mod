import datetime
from paddleocr import PaddleOCR, draw_ocr
from ultralytics import YOLO
import numpy as np
import typing as T
import cv2
import mmcv
import sys
import json
from ultralytics.engine.results import Results, Boxes, Masks
from PIL import Image, ImageDraw, ImageFilter
import time
import torch
from collections import OrderedDict, defaultdict
import os
from pathlib import Path
from itertools import chain
from collections import Counter
import subprocess
import tracemalloc
import linecache
from torch.profiler import profile, record_function, ProfilerActivity
import pickle
from scipy.optimize import linear_sum_assignment
import pandas as pd
from itertools import groupby
import re
from threading import Thread
from queue import Queue

from .schema import InspectItem


FRAME_SAMPLE_RATE = 2

DEVICE_MODEL_PATH_MAPPING = {
    'breaker': 'models/breaker/breaker0104-640s.pt',
    'junction_box': 'models/juncion_box/junction_box0107-640s.pt'
}
LOC_MARKER_MODEL_PATH = 'models/loc_marker/loc_marker0108-640s.pt'
LINE_MARKER_MODEL_PATH = 'models/line_marker/line_marker0115-640s.pt'

OCR_DET_MODEL_DIR = 'models/ocr/ocr_lightweight/det_lightweight'
OCR_REC_MODEL_DIR = 'models/ocr/ocr_lightweight/rec_lightweight'

VALID_WINDOW_WIDTH_MAPPING = {
    "breaker": 1160,
    "junction_box": 360
}


# OCR易识别错误的字母数字
EASY_ERROR_OCR_PATTERS = {
    "alpha_to_digital":{
        "O":"0",
        "o":"0",
        "r":"1",
        "l":"1",
        "T":"1",
        "Z":"2",
        "z":"2",
        "S":"5",
        "s":"5"
    },
    "digital_to_alpha":{ # 只考虑大写
        "0":"O",
        "1":"T",
        "2":"Z",
        "5":"S"
    }
}


OCR_MATCH_PATTERNS = {
    # "junction_box": re.compile(r'^\d{1,2}$'),
    "line_marker": re.compile(r'^[+-]?[A-Za-z]{2}\d{2,3}:\d{1,2}$'),
    "loc_marker": re.compile(r'^[A-Za-z]{2}\d{2,3}$')
}


MATCH_PATTERNS = {
    "breaker": {
        "IC65NC10A": {
            "body": [860, 1414, 1548, 1832],
            "terminals": {
                1: {"center": [1060, 1254], "radius": 60, "min_dist": 250},
                2: {"center": [1060, 2430], "radius": 60, "min_dist": 250},
                3: {"center": [1364, 1254], "radius": 60, "min_dist": 250},
                4: {"center": [1364, 2430], "radius": 60, "min_dist": 250},
            }
        },
        "IC65NC4A": {
            "body": [860, 1414, 1548, 1832],
            "terminals": {
                1: {"center": [1060, 1254], "radius": 60, "min_dist": 250},
                2: {"center": [1060, 2430], "radius": 60, "min_dist": 250},
                3: {"center": [1364, 1254], "radius": 60, "min_dist": 250},
                4: {"center": [1364, 2430], "radius": 60, "min_dist": 250},
            }
        },
        "IC65ND16A": {
            "body": [676, 1912, 1538, 2266],
            "terminals": {
                1: {"center": [850, 1753], "radius": 60, "min_dist": 250},
                2: {"center": [850, 2779], "radius": 60, "min_dist": 250},
                3: {"center": [1109, 1753], "radius": 60, "min_dist": 250},
                4: {"center": [1109, 2779], "radius": 60, "min_dist": 250},
                5: {"center": [1369, 1753], "radius": 60, "min_dist": 250},
                6: {"center": [1369, 2779], "radius": 60, "min_dist": 250},
            }
        },
        "VIGIIC65ELM": {
            "body": [800, 2007, 1326, 2333],
            "terminals": {
                1: {"center": [940, 2804], "radius": 60, "min_dist": 300},
                2: {"center": [1158, 2804], "radius": 60, "min_dist": 300},
            }
        },
        "IOF": {
            "body": [1094, 1223, 1272, 1692],
            "terminals": {
                1: {"center": [1131, 2252], "radius": 50, "min_dist": 250},
            }
        }
    },
    "junction_box": {
        "/": {
            "body": [910, 1561, 1014, 1703],
            "terminals": {
                1: {"center": [962, 1561], "radius": 60, "min_dist": 50},
                2: {"center": [954, 2231], "radius": 60, "min_dist": 80},
            }
        }
    }
}


def replace_digital_2_alpha(char):
    # 将本该是字母但是识别成数字的结果替换成字母
    result = ""
    for c in char:
        if c.isdigit():
            if c in EASY_ERROR_OCR_PATTERS['digital_to_alpha'].keys():
                result += EASY_ERROR_OCR_PATTERS['digital_to_alpha'][c]
            else:
                result += c 
        else:
            result += c
    return result

        
def replace_alpha_2_digital(char):
    # 将本该是数字但是识别成字母的结果替换成数字
    result = ""
    for c in char:
        if c.isalpha():
            if c in EASY_ERROR_OCR_PATTERS['alpha_to_digital'].keys():
                result += EASY_ERROR_OCR_PATTERS['alpha_to_digital'][c]
            else:
                result += c 
        else:
            result += c
    return result


def get_ocr_result_for_one_item(ocr_results, target_type):
    # 对每一个检测的物体根据该物体的类型获取其最终的ocr结果

    # 首先根据target_type来获取对应类型的文本范式
    if target_type not in OCR_MATCH_PATTERNS.keys():
        return ocr_results
    
    pattern = OCR_MATCH_PATTERNS[target_type]
    
    # 根据范式来过滤出满足范式和不满足范式的结果
    correct_ocr_results = [txt for txt in ocr_results if pattern.match(txt)]
    error_ocr_results = [txt for txt in ocr_results if not pattern.match(txt)]
    
    # 对于不满足范式的结果再用先验知识看能否召回回来
    for error_ocr_result in error_ocr_results:
        corrected_ocr_result = ""
        if target_type == "line_marker":
            # 首先对易混淆的字母和数字进行纠正，比如1识别成T这种情况，正负号后前两位一定是字母，紧跟着的两位是数字，比如-XT14:8
            if error_ocr_result[0] in ["-","+"] and len(error_ocr_result) >= 5:
                corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[1:3])
                corrected_digital_part = replace_alpha_2_digital(error_ocr_result[3:5])
                error_ocr_result = error_ocr_result[0] + corrected_alpha_part + corrected_digital_part + error_ocr_result[5:]
                corrected_ocr_result = error_ocr_result
            elif error_ocr_result[0] not in ["-","+"] and len(error_ocr_result) >= 4:
                corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[0:2])
                corrected_digital_part = replace_alpha_2_digital(error_ocr_result[2:4])
                error_ocr_result = corrected_alpha_part + corrected_digital_part + error_ocr_result[4:]
                corrected_ocr_result = error_ocr_result
            # 第一种情况：ocr识别成了-XT158，忽略了中间的冒号
            if re.compile(r'^[+-]?[A-Za-z]{2}\d{3,4}$').match(error_ocr_result):
                if error_ocr_result[0] in ["-","+"]:
                    corrected_ocr_result = error_ocr_result[:5] + ":" + error_ocr_result[5:]
                else:
                    corrected_ocr_result = error_ocr_result[:4] + ":" + error_ocr_result[4:]
            # 第二种情况：冒号识别成其他符号,比如-XT15-8
            elif re.compile(r'^[+-]?[A-Za-z]{2}\d{2}-\d{1,3}$').match(error_ocr_result):   
                if error_ocr_result[0] in ["-","+"]:
                    corrected_ocr_result = error_ocr_result[:5] + ":" + error_ocr_result[6:]
                else:
                    corrected_ocr_result = error_ocr_result.replace("-","=")
        elif target_type == "loc_marker":  
            # loc_marker的范式是两位字母+数字 
            corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[0:2])
            corrected_digital_part = replace_alpha_2_digital(error_ocr_result[2:])
            corrected_ocr_result = corrected_alpha_part + corrected_digital_part   
            
        if pattern.match(corrected_ocr_result):
            correct_ocr_results.append(corrected_ocr_result)
    return correct_ocr_results


def convert_frame_results_to_tracks(data: T.List[T.List[InspectItem]]) -> T.Dict[str, T.Dict[str, T.List[InspectItem]]]:
    """将每一帧的检测结果转换成轨迹

    Args:
        data (T.List[T.List[InspectItem]]): 每一帧检测结果以InspectItem数组的形式存储

    Returns:
        T.Dict[str, T.Dict[str, T.List[InspectItem]]]: 返回结果为两层字典, 第一层Key是目标类型, 第二层Key是目标ID, Value是InspectItem数组
    """
    tracks = {}
    for frame in data:
        for item in frame:
            if item.target_type not in tracks:
                tracks[item.target_type] = {}
            if item.track_id not in tracks[item.target_type]:
                tracks[item.target_type][item.track_id] = []
            tracks[item.target_type][item.track_id].append(item)
    for target_type in tracks:
        new_track_dict = {}
        for track_id in tracks[target_type]:
            # 出现的帧数大于30
            if len(tracks[target_type][track_id])> 30:
                # 检出字的帧数大于20
                if len([1 for item in tracks[target_type][track_id] if item.text_words]) > 20:
                    new_track_dict[track_id] = sorted(tracks[target_type][track_id], key=lambda x: x.frame_idx)
            # tracks[target_type][track_id] = sorted(tracks[target_type][track_id], key=lambda x: x.frame_idx)
        tracks[target_type] = new_track_dict
            
    # TODO 考虑对track进行过滤：1. 出现帧数过少的track，2. track中的item mask面积变化过大，3. ...
    return tracks


def init_segment_models(target_list: T.List[str]):
    seg_model_dict = {}
    for target in target_list:
        if target == "loc_marker":
            seg_model_dict[target] = YOLO(LOC_MARKER_MODEL_PATH)
        elif target == "line_marker":
            seg_model_dict[target] = YOLO(LINE_MARKER_MODEL_PATH)
        else:
            if target not in DEVICE_MODEL_PATH_MAPPING:
                raise ValueError(f"target {target} not supported")
            seg_model_dict[target] = YOLO(DEVICE_MODEL_PATH_MAPPING[target])
    return seg_model_dict


##初始化ocr	
def init_ocr_model():
    ocr_model = PaddleOCR(
        use_angle_cls=False, 
        det_model_dir = OCR_DET_MODEL_DIR,
        rec_model_dir = OCR_REC_MODEL_DIR,
        det_limit_side_len=3840, 
        # use_tensorrt=True, 
        precision="int8", 
        max_batch_size=50,
        use_mp = True,###根据情况可关闭，
        # total_process_num=2, ##进程数
        min_subgraph_size=10,
        use_dilation=True,###db膨胀，可关闭
        # process_id=0,
        # re_char_dict_path="fonts/en_dict.txt",
        rec_image_shape ="3,48,256",
        rec_batch_num = 8,

        # use_static=True,##在初次运行程序的时候会将 TensorRT 的优化信息进行序列化到磁盘上，
        # ##下次运行时直接加载优化的序列化信息而不需要重新生成。
        )
    
 
    
    return ocr_model


def generate_item_crops(
    origin_img: np.ndarray, 
    items: T.List[InspectItem], 
    mask_processes: T.Dict[str, T.List[T.Callable]], 
    crop_processes: T.Dict[str, T.List[T.Callable]]):
    """生成图片切片

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
    
    beg = time.time()
    all_masks = []
    for item in items:
        x1, y1, x2, y2 = item.box_xyxy.int().tolist()
        w, h = x2 - x1, y2 - y1
        mask = Image.new("L", (w, h), 0)
        if item.mask_xy.shape[1] < 2:
            print("*** item mask xy shape: ", item.mask_xy.shape, item.box_xyxy, item.target_type, "Skip.")
            all_masks.append(mask)
            continue
        mask_draw = ImageDraw.Draw(mask)
        mask_rel_xy = item.mask_xy - item.box_xyxy[:2].numpy()
        mask_draw.polygon(mask_rel_xy, fill=255)
        all_masks.append(mask)
    draw_time = time.time() - beg
    
    beg = time.time()
    processed_masks = []
    for mask, item in zip(all_masks, items):
        processed_mask = mask
        if item.target_type in mask_processes:
            for process in mask_processes[item.target_type]:
                processed_mask = process(processed_mask)
        processed_masks.append(processed_mask)
    mask_time = time.time() - beg
    
    beg = time.time()    
    origin_pil = Image.fromarray(origin_img)
    
    all_crops = [origin_pil.crop(item.box_xyxy.int().tolist()) for item in items]
    all_masked_crops = [Image.composite(o, Image.new("RGB", o.size, (0, 0, 0)), m) for o, m in zip(all_crops, processed_masks)]
    processed_crops = []
    for crop, item in zip(all_masked_crops, items):
        processed_crop = crop
        if item.target_type in crop_processes:
            for process in crop_processes[item.target_type]:
                processed_crop = process(processed_crop)
        processed_crops.append(processed_crop)
    crop_time = time.time() - beg
    print(f"draw_time: {draw_time:.4f} mask_time: {mask_time:.4f} crop_time: {crop_time:.4f}")
    return processed_crops

def dilate(mask: Image):
    return mask.filter(ImageFilter.MaxFilter(19))
    
def dilate_torch(mask: Image):
    mask = torch.from_numpy(np.asarray(mask)).to("cuda").unsqueeze(0).unsqueeze(0).float()
    mask = torch.nn.functional.max_pool2d(mask, 19, 1, 9)
    return Image.fromarray(mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8))

def rotate(crop: Image):
    return crop.transpose(Image.ROTATE_270)

MAX_ITEM_PER_OCR_TIME = 50

def process_frame(frame_idx: int, img: np.ndarray, seg_model_dict: T.Dict[str, YOLO], ocr_model: PaddleOCR):
    visual_res = img.copy()
    
    items: T.List[InspectItem] = []
    # 1. 跑分割模型
    beg = time.time()
    for model_target, model in seg_model_dict.items():
        results: Results = model.track(img, persist=True,tracker='cfg/botsort.yaml')[0] # 只传入一帧数据，所以默认只需要读取第0个结果
        visual_res = results.plot(img=visual_res)
        if results.boxes is None or results.masks is None:
            print(f"Frame {frame_idx} track result has None, ", results.boxes, results.masks)
            continue
        for box, mask in zip(results.boxes, results.masks):
            # 根据这个问题（https://github.com/ultralytics/ultralytics/issues/3830#issuecomment-1642651960），
            # track_id可能不会在第一次检测到物体就分配，所以这里需要判断一下
            mask_xy = mask.xy[0]
            if box.id is not None and mask_xy.shape[0] > 0: 
                item = InspectItem(frame_idx,model_target, box=box.cpu(), mask=mask_xy)
                items.append(item)
    seg_time = time.time() - beg
    
    # 2. 处理分割结果，拼接mask
    beg = time.time()
    
    if not items:
        print(f"!! Frame {frame_idx} has nothing detected")
    else:
        all_crops = generate_item_crops(img, items, mask_processes={"line_marker": [dilate_torch]}, crop_processes={"line_marker": [rotate]})
        gen_crops_time = time.time() - beg
        
        # 如果一次性要跑的ocr的切片太多，会导致内存不够，所以这里需要分批跑
        step_num = len(all_crops) // MAX_ITEM_PER_OCR_TIME + 1
        ocr_results = []
        all_box_shapes = []
        cum_canvas_height = 0
        for step_idx in range(step_num):
            step_start = time.time()
            step_crops = all_crops[step_idx*MAX_ITEM_PER_OCR_TIME: (step_idx+1)*MAX_ITEM_PER_OCR_TIME]
            if step_crops:
                # 生成拼接画布，以所有分割框的最大宽度为宽，以所有分割框的高之和为高
                step_box_shapes = np.array([crop.size for crop in step_crops])
                all_box_shapes.append(step_box_shapes)
                canvas_width = np.max(step_box_shapes[:, 0])
                canvas_height = np.sum(step_box_shapes[:, 1])
                canvas = np.zeros((canvas_height, canvas_width, 3), np.uint8)
                cum_height = 0
                for crop in step_crops:
                    canvas[cum_height: cum_height+crop.height, :crop.width, :] = np.asarray(crop)
                    cum_height += crop.height
                # cv2.imwrite(f"merge_items_{frame_idx}.jpg", canvas)
                merge_time = time.time() - beg
            
                # 3. 跑ocr模型
                beg = time.time()
                step_ocr_results = ocr_model.ocr(canvas, cls=False)
                if step_ocr_results:
                    step_ocr_results = step_ocr_results[0]
                
                if not step_ocr_results:
                    step_ocr_results = []
                if step_ocr_results:
                    for res in step_ocr_results:
                        for pi in range(len(res[0])):
                            res[0][pi][0] += cum_canvas_height
                ocr_results.extend(step_ocr_results)
                ocr_time = time.time() - beg
                cum_canvas_height += canvas_height
                
                print(f"OCR Step {step_idx + 1}: merge time - {merge_time:.4f}, ocr time - {ocr_time:.4f}, step time - {time.time() - step_start:.4f}")
        all_box_shapes = np.vstack(all_box_shapes)
        # 4. 根据检测结果，将ocr结果填充到InspectItem中
        if not ocr_results:
            return items, visual_res
        ocr_results = sorted(ocr_results, key=lambda x: x[0][0][1]) # 根据检测到的文本框第一个point的Y轴进行排序
        current_idx = 0
        current_height_top = all_box_shapes[current_idx][1]
        for points, (text, score) in ocr_results:
            if points[0][1] < current_height_top: # 如果文本的Y轴坐标小于当前切片所在坐标的Y轴坐标，则说明这个文本框属于当前切片
                items[current_idx].text_boxes.append(np.array(points))
                items[current_idx].text_words.append(text)
                items[current_idx].text_scores.append(score)
            else:
                while points[0][1] > current_height_top: # 如果文本Y轴坐标大于当前坐标，需要一直往下找，直到找到一个切片，其Y轴大于文本Y轴坐标
                    current_idx += 1
                    current_height_top += all_box_shapes[current_idx][1]
                items[current_idx].text_boxes.append(np.array(points))
                items[current_idx].text_words.append(text)
                items[current_idx].text_scores.append(score)
    
        for item in items:
            draw_ocr(visual_res, np.array(item.text_boxes), item.text_words, np.array(item.text_scores), font_path='fonts/simfang.ttf')
        print(f"frame_idx: {frame_idx}, seg_time: {seg_time:.4f} merge_time: {merge_time:.4f} (include gen crops: {gen_crops_time:.4f}) ocr_time: {ocr_time: .4f}")
    return items, visual_res


def merge_ocr_thru_frames(frame_results: T.List[T.List[InspectItem]], score_thresh: float = 0.8, track_count_thresh: int = 5, txt_count_thresh: int = 3):
    merge_results = {}
    for frame in frame_results:
        for item in frame:
            if item.target_type not in merge_results:
                merge_results[item.target_type] = {}
                
            if item.track_id not in merge_results[item.target_type]:
                merge_results[item.target_type][item.track_id] = []
                
            merge_results[item.target_type][item.track_id].append(item)

    for target_type, target_type_results in merge_results.items():
        ignore_ids = []
        for track_id, track_id_results in target_type_results.items():
            # 如果同一个track_id出现的帧数少于track_count_thresh，可能是误检，忽略
            # 这个逻辑暂时先注释掉，因为如果在这里屏蔽，后面查询的时候会出现keyerror
            # if len(track_id_results) < track_count_thresh:
            #     ignore_ids.append(track_id)
            #     continue
            
            # line_marker可能会因为线本身有字，检出多个，这时要取第一个，其他情况（主要是器件）上面有多行文字，取最后一个
            # TODO 这个地方不严谨
            if target_type == "line_marker":
                text_index = 0
            else:
                text_index = -1
            if track_id_results:
                track_txts = None
                if track_id_results:
                    track_txts = [item.text_words[text_index] for item in track_id_results if item.text_words and item.text_scores[text_index] > score_thresh]
                    track_txts = get_ocr_result_for_one_item(track_txts, target_type)
                if track_txts:
                    # print(target_type, track_id, Counter(track_txts).most_common(3))
                    txt, cnt = Counter(track_txts).most_common(1)[0]
                    if cnt > txt_count_thresh:
                        merge_results[target_type][track_id] = txt
                    else:
                        merge_results[target_type][track_id] = "*" # 如果出现次数最多的结果出现次数小于txt_count_thresh，则认为是噪声，用*代替
                else:
                    merge_results[target_type][track_id] = "*"
            else:
                merge_results[target_type][track_id] = "*"
                
        for track_id in ignore_ids:
            del target_type_results[track_id]
    return merge_results


def transform_pattern(pattern: T.Dict[str, T.Any], item: InspectItem):
    '''
    将pattern与item对齐和缩放。
    对齐: 将item的左上角与pattern的左上角对齐
    缩放: 将item的宽高缩放到与pattern的宽高一致
    '''
    shift_x = item.box_xyxy[0] - pattern['body'][0]
    shift_y = item.box_xyxy[1] - pattern['body'][1]
    zoom_x = (item.box_xyxy[2] - item.box_xyxy[0]) / (pattern['body'][2] - pattern['body'][0])
    zoom_y = (item.box_xyxy[3] - item.box_xyxy[1]) / (pattern['body'][3] - pattern['body'][1])
    px1, py1, px2, py2 = pattern['body']
    pw = px2 - px1
    ph = py2 - py1
    transformed_body = [px1 + shift_x, py1 + shift_y, px1 + shift_x + pw * zoom_x, py1 + shift_y + ph * zoom_y]
    transformed_terminals = {}
    for tid, tmnl in pattern['terminals'].items():
        cx, cy = tmnl['center']
        tx = (cx - px1) * zoom_x + shift_x + px1
        ty = (cy - py1) * zoom_y + shift_y + py1
        # TODO 这里的缩放逻辑需要优化
        transformed_terminals[tid] = {"center": [tx, ty], 'radius': tmnl['radius'] * zoom_x, 'min_dist': tmnl['min_dist'] * zoom_x}
    return {
        "body": transformed_body,
        "terminals": transformed_terminals
    }


def search_pattern(pattern: T.Dict[str, T.Any], target_type: str, model_type: str):
    pattern = None
    if target_type in MATCH_PATTERNS:
        device_pattern_mapping = MATCH_PATTERNS[target_type]
        # 去掉型号中的空格，并转成大写
        model_type = model_type.replace(" ", "").upper()
        if model_type in device_pattern_mapping or ("/" in device_pattern_mapping):
            if model_type in device_pattern_mapping:
                pattern = device_pattern_mapping[model_type]
            else:
                pattern = device_pattern_mapping["/"]
    return pattern


def wiring_match4(frame_results: T.List[T.List[InspectItem]], frame_shape: T.Tuple[int, int], device_types: T.Tuple[str] = ("breaker", "junction_box")):
    w, h = frame_shape
    item_orc_merge_results = merge_ocr_thru_frames(frame_results)
    item_tracks = convert_frame_results_to_tracks(frame_results)
    
    # 为了避免同一个线被识别成多根，用OCR检测结果做去重
    # text2lm_id = {}
    # for lm_id in item_tracks['line_marker']:
    #     text = item_orc_merge_results['line_marker'][lm_id]
    #     if text not in text2lm_id:
    #         text2lm_id[text] = []
    #     text2lm_id[text].append(lm_id)
    # for text, lm_ids in text2lm_id.items():
    #     if len(lm_ids) > 1:
    #         lm_ids = list(sorted(lm_ids, key=lambda x: len(item_tracks['line_marker'][x]), reverse=True))
    #         for lm_id in lm_ids[1:]:
    #             del item_tracks['line_marker'][lm_id]

    device_pattern_map = {}
    row_indices = []
    for device_type in device_types:
        if device_type not in device_pattern_map:
            device_pattern_map[device_type] = {}
        if device_type not in item_tracks:
            continue
        for device_id in item_tracks[device_type]:
            if device_id in item_orc_merge_results[device_type]:
                pattern = search_pattern(MATCH_PATTERNS, device_type, item_orc_merge_results[device_type][device_id])
            else:
                pattern = None
            device_pattern_map[device_type][device_id] = pattern
            if pattern is not None:
                for terminal_key in pattern['terminals']:
                    row_indices.append((device_type, device_id, terminal_key))
    if 'line_marker' not in item_tracks:
        return {
            "match_result": [],
        }
    # 使用pandas dataframe来存储距离矩阵，row key为(device_type, device_id, terminal_key)，col key为(line_marker_id)
    dist_matrix = pd.DataFrame(
        np.ones((len(row_indices), len(item_tracks['line_marker'])), np.float32) * 10000, 
        index=pd.MultiIndex.from_tuples(row_indices, names=['device_type', 'device_id', 'terminal_key']), 
        columns=item_tracks['line_marker'].keys())
    
    pattern_debug_info = {}
    for frame in frame_results:
        frame_item_map = {key: list(items) for key, items in groupby(frame, lambda x: x.target_type)}
        for item in frame:
            if item.target_type in device_types:
                if item.track_id not in item_tracks[item.target_type]:
                    continue
                if item.track_id not in device_pattern_map[item.target_type]:
                    continue
                pattern = device_pattern_map[item.target_type][item.track_id]
                if pattern is None:
                    continue
                win_width = VALID_WINDOW_WIDTH_MAPPING[item.target_type]
                win_left = (w - win_width) // 2
                win_right = (w + win_width) // 2
                if win_left < item.box_xyxy[0] and item.box_xyxy[2] < win_right:
                    pattern = transform_pattern(pattern, item)
                    pattern_debug_info[(item.target_type, item.frame_idx, item.track_id)] = pattern
                    for terminal_key, tmnl in pattern['terminals'].items():
                        if 'line_marker' not in frame_item_map:
                            continue
                        for lm in frame_item_map['line_marker']:
                            if lm.track_id not in item_tracks['line_marker']:
                                continue
                            if lm.box_xyxy[1] < (frame_shape[1] // 2):
                                pinpoint = lm.bottom_point
                            else:
                                pinpoint = lm.top_point
                            dist = np.sqrt(np.sum(np.square(np.array(tmnl['center']) - np.array(pinpoint))))
                            if dist < tmnl['min_dist'] and dist < dist_matrix.loc[(item.target_type, item.track_id, terminal_key), lm.track_id]:
                                dist_matrix.loc[(item.target_type, item.track_id, terminal_key), lm.track_id] = dist
    row_ind, col_ind = linear_sum_assignment(dist_matrix.values)
    
    # TODO 根据device的类型，匹配loc_marker
    
    match_results = {}
    for ri, ci in zip(row_ind, col_ind):
        device_type, device_id, terminal_key = dist_matrix.index[ri]
        lm_id = dist_matrix.columns[ci]
        if (device_type, device_id) not in match_results:
            match_results[(device_type, device_id)] = {
                "model_type": item_orc_merge_results[device_type][device_id],
                "loc_marker": "*", # TODO 这里需要补充
                "line_marker": {}
            }
        if dist_matrix.loc[(device_type, device_id, terminal_key), lm_id] < 1000:
            match_results[(device_type, device_id)]['line_marker'][int(terminal_key)] = item_orc_merge_results['line_marker'][lm_id]

    # loc_marker 临时方案
    for device_type, device_id in match_results:
        match_res = match_results[(device_type, device_id)]
        loc_list = []
        for lm_text in match_res['line_marker'].values():
            loc_part = lm_text.split(":")[0]
            if loc_part.startswith("-"):
                loc_part = loc_part[1:]
            loc_list.append(loc_part)
        if loc_list:
            match_res['loc_marker'] = Counter(loc_list).most_common(1)[0][0]
            
    # 端子排，按照位置编号合并结果
    if match_results and list(match_results.keys())[0][0] == "junction_box":
        loc2devices = {}
        for device_type, device_id in match_results:
            loc = match_results[(device_type, device_id)]['loc_marker']
            if loc not in loc2devices:
                loc2devices[loc] = []
            loc2devices[loc].append(match_results[(device_type, device_id)])
        match_results = {}
        for loc, devices in loc2devices.items():
            match_results[loc] = {
                "model_type": "junction_box",
                "loc_marker": loc,
                "line_marker": {}
            }
            for device in devices:
                if 1 in device['line_marker']:
                    match_results[loc]['line_marker'][device['model_type']] = device['line_marker'][1]

    return {
        "match_result": list(match_results.values()),
        "ocr_result": item_orc_merge_results,
        "track_result": item_tracks,
        "pattern_debug_info": pattern_debug_info,
        "dist_matrix": dist_matrix,
    }

def wiring_match3(frame_results: T.List[T.List[InspectItem]], frame_shape: T.Tuple[int, int]):
    # 将所有相同ID的OCR结果合并，按照出现次数排序，取出现次数最多的结果
    item_orc_merge_results = merge_ocr_thru_frames(frame_results)
    
    # 重排组织frame_results结果，以器件id为key，将器件所有出现的帧，以及帧中器件的位置和line_marker的对应关系都记录下来
    device_tracking_map = {}
    w, h = frame_shape
    for frame in frame_results:
        current_frame_matches = []
        for item in frame:
            if item.target_type in ["breaker", "junction_box"]:
                if item.track_id not in device_tracking_map:
                    device_tracking_map[item.track_id] = {
                        "loc_marker": "*",
                        "line_marker": {},
                        "track": [],
                        "model_type": item_orc_merge_results[item.target_type][item.track_id]
                    }
                frame_match = {
                    "frame_idx": item.frame_idx,
                    "is_in_window": False,
                    "device": item,
                    "loc_marker": [],
                    "line_marker": [],
                    "pattern": None,
                    "matches": {}
                }
                win_width = VALID_WINDOW_WIDTH_MAPPING[item.target_type]
                win_left = (w - win_width) // 2
                win_right = (w + win_width) // 2
                # print(item.frame_idx, item.track_id, item.box_xyxy, w, win_width, win_left, win_right)
                if win_left < item.box_xyxy[0] and item.box_xyxy[2] < win_right:
                    frame_match['is_in_window'] = True
                    # 查找pattern
                    model_type = item_orc_merge_results[item.target_type][item.track_id]
                    pattern = search_pattern(MATCH_PATTERNS, item.target_type, model_type)
                    if pattern is not None:
                        transformed_pattern = transform_pattern(pattern, item)
                        frame_match['pattern'] = transformed_pattern
                device_tracking_map[item.track_id]['track'].append(frame_match)
                current_frame_matches.append(frame_match)
                        
        for item in frame:
            if item.target_type == "loc_marker":
                for fmatch in current_frame_matches:
                    if fmatch['is_in_window'] is False:
                        continue
                    device = fmatch['device']
                    if item.box_xyxy is not None:
                        if device.box_xyxy[0] < item.box_xyxy[0] and item.box_xyxy[2] < device.box_xyxy[2]:
                            fmatch['loc_marker'].append(item)

            if item.target_type == "line_marker":
                for fmatch in current_frame_matches:
                    if fmatch['is_in_window'] is False:
                        continue
                    device = fmatch['device']
                    if item.box_xyxy[1] < (h // 2):
                        pinpoint = item.bottom_point
                    else:
                        pinpoint = item.top_point
                    
                    if device.box_xyxy[0] < pinpoint[0] < device.box_xyxy[2]:
                        # print("frame_id ", item.frame_idx, " line_marker: ", item.track_id, " match with device: ", dev_id, " pinpoint: ", pinpoint, " device box: ", device.box_xyxy, ",".join(item.text_words))
                        fmatch['line_marker'].append(item)
    
    dist_row_inds = []
    dist_col_inds = []
    row_num = 0
    col_num = 0
    for dev_id, fmatches in device_tracking_map.items():
        pattern = None
        for fmatch in fmatches['track']:
            if fmatch['pattern'] is not None:
                pattern = fmatch['pattern']
                break
        if pattern is None:
            print(dev_id, "没有找到pattern，跳过")
            continue
        row_num += len(pattern['terminals'])
        for terminal_key in pattern['terminals']:
            dist_row_inds.append((dev_id, terminal_key))
    
    for lm_id in item_orc_merge_results['line_marker']:
        dist_col_inds.append(lm_id)
        col_num += 1
    
    dist_matrix = np.ones((row_num, col_num), np.float32) * 10000
    for dev_id, fmatches in device_tracking_map.items():
        for fmatch in fmatches['track']:
            if fmatch['is_in_window'] is False:
                continue
            # 计算pattern的四个引脚和line_marker的距离
            if fmatch['pattern'] is not None:
                pattern = fmatch['pattern']
                for terminal_key, tmnl in pattern['terminals'].items():
                    for lm in fmatch['line_marker']:
                        if lm.box_xyxy[1] < (h // 2):
                            pinpoint = lm.bottom_point
                        else:
                            pinpoint = lm.top_point
                        dist = np.sqrt(np.sum(np.square(np.array(tmnl['center']) - np.array(pinpoint))))
                        if dev_id == 123:
                            print("dev_id: ", dev_id, " terminal_key: ", terminal_key, " lm_id ", lm.track_id, " dist: ", dist, lm.box_xyxy[1], tmnl['center'], pinpoint, lm.top_point)
                        if dist < tmnl['min_dist']:
                            ri, ci = dist_row_inds.index((dev_id, terminal_key)), dist_col_inds.index(lm.track_id)
                            if dist < dist_matrix[ri, ci]:
                                dist_matrix[ri, ci] = dist
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    for ri, ci in zip(row_ind, col_ind):
        dev_id, terminal_key = dist_row_inds[ri]
        lm_id = dist_col_inds[ci]
        if dist_matrix[ri, ci] < 1000:
            device_tracking_map[dev_id]['line_marker'][terminal_key] = item_orc_merge_results['line_marker'][lm_id]
    
    final_result = [
        {
            "model_type": v['model_type'],
            "loc_marker": v['loc_marker'],
            "line_marker": v['line_marker']
        }
        for k, v in device_tracking_map.items()
    ]
    return final_result, device_tracking_map, dist_matrix, row_ind, col_ind, dist_row_inds, dist_col_inds


def wiring_match2(frame_results: T.List[T.List[InspectItem]], frame_shape: T.Tuple[int, int]):
    # 将所有相同ID的OCR结果合并，按照出现次数排序，取出现次数最多的结果
    item_orc_merge_results = merge_ocr_thru_frames(frame_results)
    
    all_frame_matches = {}
    w, h = frame_shape
    for frame_idx, frame in enumerate(frame_results):
        # 记录每一帧的匹配结果
        frame_match: T.Dict[int, T.Dict[str, T.List[int]]] = {}
        # 1. 找到当前帧的有效匹配的器件
        for item in frame:
            if item.track_id not in item_orc_merge_results[item.target_type]:
                continue
            if item.target_type in ["breaker", "junction_box"]:
                win_width = VALID_WINDOW_WIDTH_MAPPING[item.target_type]
                win_left = (w - win_width) // 2
                win_right = (w + win_width) // 2
                # print(item.frame_idx, item.track_id, item.box_xyxy, w, win_width, win_left, win_right)
                if win_left < item.box_xyxy[0] and item.box_xyxy[2] < win_right:
                    assert item.track_id not in frame_match, "同一帧中出现了两个相同track_id的器件"
                    frame_match[item.track_id] = {
                        "device": [item],
                        "loc_marker": [],
                        "line_marker": [],
                        "pattern": None,
                        "matches": {}
                    }
                    # 查找pattern
                    if item.target_type in MATCH_PATTERNS:
                        device_pattern_mapping = MATCH_PATTERNS[item.target_type]
                        model_type = item_orc_merge_results[item.target_type][item.track_id]
                        # 去掉型号中的空格，并转成大写
                        model_type = model_type.replace(" ", "").upper()
                        if model_type in device_pattern_mapping or ("/" in device_pattern_mapping):
                            if model_type in device_pattern_mapping:
                                pattern = device_pattern_mapping[model_type]
                            else:
                                pattern = device_pattern_mapping["/"]
                            # 找到可以用来匹配的模板后，将其与检测结果对齐和缩放，记录到对应的item中
                            shift_x = item.box_xyxy[0] - pattern['body'][0]
                            shift_y = item.box_xyxy[1] - pattern['body'][1]
                            zoom_x = (item.box_xyxy[2] - item.box_xyxy[0]) / (pattern['body'][2] - pattern['body'][0])
                            zoom_y = (item.box_xyxy[3] - item.box_xyxy[1]) / (pattern['body'][3] - pattern['body'][1])
                            px1, py1, px2, py2 = pattern['body']
                            pw = px2 - px1
                            ph = py2 - py1
                            transformed_body = [px1 + shift_x, py1 + shift_y, px1 + shift_x + pw * zoom_x, py1 + shift_y + ph * zoom_y]
                            transformed_terminals = {}
                            for tid, tmnl in pattern['terminals'].items():
                                cx, cy = tmnl['center']
                                tx = (cx - px1) * zoom_x + shift_x + px1
                                ty = (cy - py1) * zoom_y + shift_y + py1
                                # TODO 这里的缩放逻辑需要优化
                                transformed_terminals[tid] = {"center": [tx, ty], 'radius': tmnl['radius'] * zoom_x, 'min_dist': tmnl['min_dist'] * zoom_x}
                            frame_match[item.track_id]['pattern'] = {
                                "body": transformed_body,
                                "terminals": transformed_terminals
                            }
                    # print(item.frame_idx, item.track_id, item.target_type, item.box_xyxy, ",".join(item.text_words), w, win_width, win_left, win_right)
    
        # 2. 找到每个器件关联的位置标签和号码管
        for item in frame:
            if item.target_type == "loc_marker":
                for dev_id, match_result in frame_match.items():
                    device = match_result['device'][0]
                    if item.box_xyxy is not None:
                        if device.box_xyxy[0] < item.box_xyxy[0] and item.box_xyxy[2] < device.box_xyxy[2]:
                            frame_match[dev_id]['loc_marker'].append(item)

            if item.target_type == "line_marker":
                for dev_id, match_result in frame_match.items():
                    device = match_result['device'][0]
                    # win_width = VALID_WINDOW_WIDTH_MAPPING[device.target_type]
                    # win_left = (w - win_width) // 2
                    # win_right = (w + win_width) // 2
                    
                    # if not (win_left < item.box_xyxy[0] and item.box_xyxy[2] < win_right):
                    #     continue
                    
                    if item.box_xyxy[1] < (h // 2):
                        pinpoint = item.bottom_point
                    else:
                        pinpoint = item.top_point
                    
                    if device.box_xyxy[0] < pinpoint[0] < device.box_xyxy[2]:
                        # print("frame_id ", item.frame_idx, " line_marker: ", item.track_id, " match with device: ", dev_id, " pinpoint: ", pinpoint, " device box: ", device.box_xyxy, ",".join(item.text_words))
                        frame_match[dev_id]['line_marker'].append(item)
        
        for dev_id, match_result in frame_match.items():
            if match_result['pattern'] is not None:
                pattern = match_result['pattern']
                terminal_centers = np.array([tmnl['center'] for tmnl in pattern['terminals'].values()])
                if match_result['line_marker']:
                    line_marker_pinpoints = np.array([item.bottom_point if item.box_xyxy[1] < (h // 2) else item.top_point for item in match_result['line_marker']])
                    # 计算pinpoint与center之间的两两距离
                    dists = np.sqrt(np.sum(np.square(terminal_centers[:, None, :] - line_marker_pinpoints[None, :, :]), axis=-1))
                    # 使用scikit-learn的linear_sum_assignment方法，计算最小距离的匹配
                    row_ind, col_ind = linear_sum_assignment(dists)
                    for row, col in zip(row_ind, col_ind):
                        terminal_key = list(pattern['terminals'].keys())[row]
                        if dists[row, col] < pattern['terminals'][terminal_key]['min_dist']:
                            match_result['matches'][terminal_key] = match_result['line_marker'][col] # TODO 这里的terminal key是有问题的
                        # else:
                        #     print(frame_idx, "距离太远，不匹配", dists[row, col], line_marker_pinpoints[col], terminal_centers[row], h)
                        #     print(match_result['pattern'])
            
            if dev_id not in all_frame_matches:
                all_frame_matches[dev_id] = []
            all_frame_matches[dev_id].append(match_result)
            
    # 合并每个器件的匹配结果
    merge_match_results = {}
    for dev_id, frame_match_results in all_frame_matches.items():
        matches = {}
        all_loc_markers = set(chain(*[[item.track_id for item in res['loc_marker']] for res in frame_match_results]))
        for res in frame_match_results:
            for key, value in res['matches'].items():
                if key not in matches:
                    matches[key] = []
                matches[key].append(value.track_id)
        for key in matches:
            matches[key] = list(set(matches[key]))[0] # TODO 如果多个帧都有匹配，且tracker_id不一样，暂时先取第一个，后面需要改进
        # all_line_markers = set(chain(*[[item.track_id for item in res['line_marker']] for res in frame_match_results]))
        merge_match_results[dev_id] = {
            "loc_marker": list(all_loc_markers),
            "line_marker": matches
        }
            
    # 将OCR结果填充到merge_match_results中
    final_results = []
    for dev_id, match_result in merge_match_results.items():
        loc_markers = [item_orc_merge_results['loc_marker'][loc_marker_id] for loc_marker_id in match_result['loc_marker']]
        if loc_markers:
            merge_loc_marker = max(loc_markers, key=len) # 如果有位置标签，取最长的
        else:
            merge_loc_marker = "*" # 如果没有位置标签，用*代替
        
        line_marker_matches = {
            mk: item_orc_merge_results['line_marker'][mv]
            for mk, mv in match_result['line_marker'].items()
        }
        if 'breaker' in item_orc_merge_results and dev_id in item_orc_merge_results['breaker']:
            final_results.append({
                "model_type": item_orc_merge_results['breaker'][dev_id],
                "loc_marker": merge_loc_marker,
                "line_marker": line_marker_matches
            })
        elif 'junction_box' in item_orc_merge_results and dev_id in item_orc_merge_results['junction_box']:
            final_results.append({
                "model_type": item_orc_merge_results['junction_box'][dev_id],
                "loc_marker": merge_loc_marker,
                "line_marker": line_marker_matches
            })
            
    return final_results
    

def wiring_match(frame_results):
    content = []
    final_results = {}

    for frame in frame_results:
        for item in frame:
            target_type = item.target_type
            item_id = int(item.track_id.item())
            item_xywh = item.box_xywh.tolist()
            text_words = item.text_words

            if target_type == 'breaker' or target_type == 'junction_box':
                model_id = item_id
                model_text = text_words[-1] if text_words else None
                model_xywh = item_xywh
                if model_id not in final_results:
                    final_results[model_id] = {
                        'model_xywh':[],
                        'model_max_result': None,
                        'model_max_count': 0,
                        'loc_marker_results': {'max_result': None, 'max_count': 0},
                        'line_marker_results': defaultdict(lambda: {'max_result': None, 'max_count': 0})
                    }
                
                final_results[model_id]['model_xywh'] = model_xywh
                final_results[model_id]['model_max_result'] = model_text
                final_results[model_id]['model_max_count'] += 1

            elif target_type == 'loc_marker':
                loc_marker_center_x = item_xywh[0] + item_xywh[2] / 2
                loc_marker_center_y = item_xywh[1] + item_xywh[3] / 2
                for model_id, model_data in final_results.items():
                    model_xywh = model_data['model_xywh']
                    if (model_xywh[0] < loc_marker_center_x < model_xywh[0] + model_xywh[2]) \
                            and (loc_marker_center_y < model_xywh[1]):
                        final_results[model_id]['loc_marker_results']['max_result'] = item.text_words[-1] if text_words else None
                        final_results[model_id]['loc_marker_results']['max_count'] += 1
                    
                    # 端子排的情况，位置标签在端子排的左侧，并且和端子排平行
                    elif (loc_marker_center_x < model_xywh[0]) and (loc_marker_center_y < model_xywh[3]) :
                        final_results[model_id]['loc_marker_results']['max_result'] = text_words[-1] if text_words else None
                        final_results[model_id]['loc_marker_results']['max_count'] += 1

            elif target_type == 'line_marker':
                line_marker_center_x = item.box_xywh[0] + item.box_xywh[2] / 2
                line_marker_center_y = item.box_xywh[1] + item.box_xywh[3] / 2

                for model_id, model_data in final_results.items():
                    model_xywh = model_data['model_xywh']
                    if (model_xywh[0] < line_marker_center_x < model_xywh[0] + model_xywh[2]):
                        if line_marker_center_y < model_xywh[1]:
                            line_marker_id = len(final_results[model_id]['line_marker_results']) + 1
                            final_results[model_id]['line_marker_results'][line_marker_id]['max_result'] = item.text_words[-1] if text_words else None
                            final_results[model_id]['line_marker_results'][line_marker_id]['max_count'] += 1
                        elif line_marker_center_y > model_xywh[1] + model_xywh[3]:
                            line_marker_id = len(final_results[model_id]['line_marker_results'])
                            final_results[model_id]['line_marker_results'][line_marker_id]['max_result'] = item.text_words[-1] if text_words else None
                            final_results[model_id]['line_marker_results'][line_marker_id]['max_count'] += 1
    for model_id, model_data in final_results.items():
        final_results[model_id]['line_marker_results'] = dict(final_results[model_id]['line_marker_results'])

    for key, value in final_results.items():
        content_entry = {
            "model_type": value.get('model_max_result', ''),
            "loc_marker": value['loc_marker_results'].get('max_result', ''),
            "line_marker": {}
        }
        for line_key, line_value in value['line_marker_results'].items():
            content_entry["line_marker"][str(line_key)] = line_value['max_result']
        content.append(content_entry)
    return content

def convert_to_h264(input_video_path: str, output_video_path: str):
    # 运行ffmpeg需要重置环境变量，不然会和conda内装的ffmpeg冲突报错
    command = f"LD_LIBRARY_PATH=/lib/x86_64-linux-gnu /usr/bin/ffmpeg -i {input_video_path} -vcodec h264_nvenc -y {output_video_path}"
    p = subprocess.Popen(command, shell=True)
    p.wait() # 等待进程结束，避免json已经生成但视频还没转换完的情况

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def inspect_video(video_path: str, inspect_targets: T.List[str], out_dir: str, ocr_model: PaddleOCR = None, progress_callback: T.Callable = None):
    seg_model_dict = init_segment_models(inspect_targets)
    if ocr_model is None:
        ocr_model = init_ocr_model()
    
    vid = mmcv.VideoReader(video_path)
    
    frame_results = []
    
    out_vid_path = f"{out_dir}/{Path(video_path).stem}-out.mp4"
    tmp_vid_path = f"/tmp/{Path(video_path).stem}-tmp.mp4"
    out_vid = cv2.VideoWriter(tmp_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), vid.fps / FRAME_SAMPLE_RATE, (vid.width, vid.height))
    
    # tracemalloc.start()
    # with profile(activities=[ProfilerActivity.CPU],
    #     profile_memory=True, record_shapes=True, with_stack=True, with_modules=True) as prof:
    
    progress_interval = vid.frame_cnt // 100
    for idx, frame in enumerate(vid):
        if idx % FRAME_SAMPLE_RATE != 0:
            continue
        
        if progress_callback is not None and progress_interval != 0 and idx % progress_interval == 0:
            progress_callback(idx / vid.frame_cnt)
        
        frame_result, visual_frame = process_frame(idx, frame, seg_model_dict, ocr_model)
        frame_results.append(frame_result)
        
        if "breaker" in inspect_targets:
            win_width = VALID_WINDOW_WIDTH_MAPPING['breaker']
        elif "junction_box" in inspect_targets:
            win_width = VALID_WINDOW_WIDTH_MAPPING['junction_box']
        win_left = (vid.width - win_width) // 2
        win_right = (vid.width + win_width) // 2
        line1 = [win_left, 0], [win_left, vid.height]
        line2 = [win_right, 0], [win_right, vid.height]
        cv2.line(visual_frame, line1[0], line1[1], (0, 255, 0), 3)
        cv2.line(visual_frame, line2[0], line2[1], (0, 255, 0), 3)
        out_vid.write(visual_frame)
    
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by='cpu_memory_usage', row_limit=10))
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    
    with open(f"{out_dir}/{Path(video_path).stem}.pkl", "wb") as f:
        # print(type(frame_results[0][0]._box.data), sys.getsizeof(frame_results[0][0]._box.data), sys.getsizeof(frame_results[0][0]._mask))
        pickle.dump(frame_results, f)
    
    # match_results = wiring_match2(frame_results, (vid.width, vid.height))
    match_results = wiring_match4(frame_results, (vid.width, vid.height))['match_result']
    print(match_results)
    out_vid.release()
    vid.vcap.release()
    # 因为opencv默认不支持h264编码，所以这里需要转换一下
    time.sleep(1)
    beg = time.time()
    convert_to_h264(tmp_vid_path, out_vid_path)
    print(time.time() - beg, "*******************************")
    
    return match_results


INSPECT_TYPE_MAPPING = {
    "breaker": ["breaker", "loc_marker", "line_marker"],
    "junction_box_top": ["junction_box", "loc_marker", "line_marker"]
}


class DirWatcher(object):
    def __init__(self, dir_path: str, out_path: str, interval: int = 1, ocr_model: PaddleOCR = None, since: datetime.datetime = None):
        self.dir_path = Path(dir_path)
        self.out_path = Path(out_path)
        self._last_mtime = since
        self.interval = interval
        if ocr_model is None:
            ocr_model = init_ocr_model()
        self.ocr_model = ocr_model
    
    def check(self):
        to_inspect_list = []
        for mp4_file in self.dir_path.glob("*.mp4"):
            file_modified_time = datetime.datetime.fromtimestamp(mp4_file.stat().st_mtime)
            if file_modified_time > self._last_mtime:
                to_inspect_list.append(mp4_file)
        self._last_mtime = datetime.datetime.now()
        return to_inspect_list
    
    def start_loop(self):
        while True:
            to_inspect_list = self.check()
            print(f"{datetime.datetime.now().isoformat()} - found: {len(to_inspect_list)} files to analyze: ", to_inspect_list)
            for mp4_file in to_inspect_list:
                file_parts = mp4_file.stem.split("-")
                inspect_type = file_parts[1]
                print("start to inspect: ", mp4_file, " with inspect type:", inspect_type)
                beg = time.time()
                if inspect_type in INSPECT_TYPE_MAPPING:
                    result = inspect_video(str(mp4_file), INSPECT_TYPE_MAPPING[inspect_type], str(self.out_path), self.ocr_model)
                    print(result)
                    if result:
                        out_file = self.out_path / f"{mp4_file.stem}.json"
                        with open(out_file, "w") as f:
                            json.dump(result, f)
                        print("finish to inspect: ", mp4_file, " with inspect type:", inspect_type, " output: ", out_file, f" time: {time.time() - beg:.2f}s")
            time.sleep(self.interval)

if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     print("Usage: python main.py <video_path> <inspect_type> <output_path>")

    # video_path = sys.argv[1]
    # inspect_type = sys.argv[2]
    # output_path = sys.argv[3]
    # if inspect_type not in INSPECT_TYPE_MAPPING:
    #     print(f"inspect_type not supported, current support type: {INSPECT_TYPE_MAPPING.keys()}")
        
    # inspect_targets = INSPECT_TYPE_MAPPING[inspect_type]
    
    # match_results = inspect_video(video_path, inspect_targets)
    
    # print(match_results)
    # with open(output_path, "w") as f:
    #     json.dump(match_results, f)
    
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python main.py <video_dir_path> <output_dir_path> (<since>)")
    if len(sys.argv) == 3:
        watcher = DirWatcher(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        watcher = DirWatcher(sys.argv[1], sys.argv[2], since=datetime.datetime.fromisoformat(sys.argv[3]))
    watcher.start_loop()
    