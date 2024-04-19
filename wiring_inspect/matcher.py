from collections import Counter, defaultdict
from copy import deepcopy
from loguru import logger
from itertools import groupby
import re
import typing as T

import cv2 as cv
import math
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from wiring_inspect.schema import InspectItem


# OCR易识别错误的字母数字
EASY_ERROR_OCR_PATTERS = {
    "alpha_to_digital": {
        "O": "0",
        "o": "0",
        "r": "1",
        "l": "1",
        "T": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
    },
    "digital_to_alpha": {"0": "O", "1": "T", "2": "Z", "5": "S"},  # 只考虑大写
}


OCR_MATCH_PATTERNS = {
    # "junction_box": re.compile(r'^\d{1,2}$'),
    "line_marker": re.compile(r"^[+-]?[A-Za-z]{2}\d{2,3}:\d{1,2}$"),
    "loc_marker": re.compile(r"^[A-Za-z]{2}\d{2,3}$"),
}


def search_pattern(device_pattern_map: T.Dict, model_type: str):
    pattern = None
    # 去掉型号中的空格，并转成大写
    model_type = model_type.replace(" ", "").upper()
    if model_type in device_pattern_map or ("/" in device_pattern_map):
        if model_type in device_pattern_map:
            pattern = device_pattern_map[model_type]
        else:
            pattern = device_pattern_map["/"]
    return pattern


def convert_frame_results_to_tracks(
    frame_results: T.List[T.List[InspectItem]],
) -> T.Dict[str, T.Dict[str, T.List[InspectItem]]]:
    """将每一帧的检测结果转换成轨迹

    Args:
        frame_results (T.List[T.List[InspectItem]]): 每一帧检测结果以InspectItem数组的形式存储

    Returns:
        T.Dict[str, T.Dict[str, T.List[InspectItem]]]: 返回结果为两层字典, 第一层Key是目标类型, 第二层Key是目标ID, Value是InspectItem数组
    """
    tracks = {}
    for frame in frame_results:
        for item in frame:
            if item.target_type not in tracks:
                tracks[item.target_type] = {}
            if item.track_id not in tracks[item.target_type]:
                tracks[item.target_type][item.track_id] = []
            tracks[item.target_type][item.track_id].append(item)

    for target_type in tracks:
        for track_id in tracks[target_type]:
            tracks[target_type][track_id] = sorted(
                tracks[target_type][track_id], key=lambda x: x.frame_idx
            )

    # TODO 考虑对track进行过滤：1. 出现帧数过少的track，2. track中的item mask面积变化过大，3. ...
    return tracks


def replace_digital_2_alpha(char):
    # 将本该是字母但是识别成数字的结果替换成字母
    result = ""
    for c in char:
        if c.isdigit():
            if c in EASY_ERROR_OCR_PATTERS["digital_to_alpha"].keys():
                result += EASY_ERROR_OCR_PATTERS["digital_to_alpha"][c]
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
            if c in EASY_ERROR_OCR_PATTERS["alpha_to_digital"].keys():
                result += EASY_ERROR_OCR_PATTERS["alpha_to_digital"][c]
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
            if error_ocr_result[0] in ["-", "+"] and len(error_ocr_result) >= 5:
                corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[1:3])
                corrected_digital_part = replace_alpha_2_digital(error_ocr_result[3:5])
                error_ocr_result = (
                    error_ocr_result[0]
                    + corrected_alpha_part
                    + corrected_digital_part
                    + error_ocr_result[5:]
                )
                corrected_ocr_result = error_ocr_result
            elif error_ocr_result[0] not in ["-", "+"] and len(error_ocr_result) >= 4:
                corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[0:2])
                corrected_digital_part = replace_alpha_2_digital(error_ocr_result[2:4])
                error_ocr_result = (
                    corrected_alpha_part + corrected_digital_part + error_ocr_result[4:]
                )
                corrected_ocr_result = error_ocr_result
            # 第一种情况：ocr识别成了-XT158，忽略了中间的冒号
            if re.compile(r"^[+-]?[A-Za-z]{2}\d{3,4}$").match(error_ocr_result):
                if error_ocr_result[0] in ["-", "+"]:
                    corrected_ocr_result = (
                        error_ocr_result[:5] + ":" + error_ocr_result[5:]
                    )
                else:
                    corrected_ocr_result = (
                        error_ocr_result[:4] + ":" + error_ocr_result[4:]
                    )
            # 第二种情况：冒号识别成其他符号,比如-XT15-8
            elif re.compile(r"^[+-]?[A-Za-z]{2}\d{2}-\d{1,3}$").match(error_ocr_result):
                if error_ocr_result[0] in ["-", "+"]:
                    corrected_ocr_result = (
                        error_ocr_result[:5] + ":" + error_ocr_result[6:]
                    )
                else:
                    corrected_ocr_result = error_ocr_result.replace("-", "=")
        elif target_type == "loc_marker":
            # loc_marker的范式是两位字母+数字
            corrected_alpha_part = replace_digital_2_alpha(error_ocr_result[0:2])
            corrected_digital_part = replace_alpha_2_digital(error_ocr_result[2:])
            corrected_ocr_result = corrected_alpha_part + corrected_digital_part

        if pattern.match(corrected_ocr_result):
            correct_ocr_results.append(corrected_ocr_result)
    return correct_ocr_results


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


def affine_transform_pattern(pattern: T.Dict[str, T.Any], item: InspectItem) -> T.Dict:
    '''
    非共线的3对对应点确定唯一一个仿射变换。
    找到 pattern 的 body （铭牌区域）与 item.box_xy （检测到的铭牌区域）的仿射矩阵。
    再根据仿射矩阵，计算 terminal center 对应的坐标。
    '''
    bx1, by1, bx2, by2 = pattern['body']
    ix1, iy1, ix2, iy2 = item.box_xyxy

    body_points = np.array([[bx1, by1], [bx1, by2], [bx2, by2]], dtype=np.float32)
    item_points = np.array([[ix1, iy1], [ix1, iy2], [ix2, iy2]], dtype=np.float32)
    matrix = cv.getAffineTransform(body_points, item_points)  # 注意参数顺序

    transformed_terminals = {}
    for terminal_name, terminal in pattern['terminals'].items():
        x, y = terminal['center']
        new_x, new_y = matrix @ np.array([x, y, 1], dtype=np.float32)

        transformed_terminals[terminal_name] = {}
        transformed_terminals[terminal_name]['center'] = [new_x, new_y]
        transformed_terminals[terminal_name]['angle'] = terminal['angle']

    return transformed_terminals


def merge_ocr_thru_frames(
    frame_results: T.List[T.List[InspectItem]],
    score_thresh: float = 0.8,
    track_count_thresh: int = 5,
    txt_count_thresh: int = 3,
):
    merge_results = {}
    for frame in frame_results:
        for item in frame:
            if item.target_type not in merge_results:
                merge_results[item.target_type] = {}

            if item.track_id not in merge_results[item.target_type]:
                merge_results[item.target_type][item.track_id] = []

            merge_results[item.target_type][item.track_id].append(item)

    # TODO: 由于 track 精度问题，可能导致不同的 item 具有相同的 track id
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

            if len(track_id_results):
                track_txts = None

                if len(track_id_results):
                    track_txts = [
                        item.text_words[text_index]
                        for item in track_id_results
                        if len(item.text_words)
                        and item.text_scores[text_index] > score_thresh
                    ]
                    track_txts = get_ocr_result_for_one_item(track_txts, target_type)
                if track_txts:
                    # logger.info(f'target_type: {target_type}, track_id: {track_id}, most_common(3): {Counter(track_txts).most_common(3)}')
                    txt, cnt = Counter(track_txts).most_common(1)[0]
                    if cnt > txt_count_thresh:
                        merge_results[target_type][track_id] = txt
                    else:
                        merge_results[target_type][
                            track_id
                        ] = "*"  # 如果出现次数最多的结果出现次数小于txt_count_thresh，则认为是噪声，用*代替
                else:
                    merge_results[target_type][track_id] = "*"
            else:
                merge_results[target_type][track_id] = "*"

        for track_id in ignore_ids:
            del target_type_results[track_id]
    return merge_results


def generate_output(match_results: T.Dict, item_tracks: T.Dict) -> T.Tuple[T.List, T.Dict]:
    '''
    根据 match_results 和 item_tracks 整理输出结果。
    
    Args:
        match_results:
        item_track:
    Returns:
        outputs_match: 将 match_results 按照AI检测结果格式说明第二版进行整理
        outputs_data: 记录 match_results 中出现的有效结果，用于可视化处理
    '''
    outputs_match = []
    outputs_data = {}

    oputputs_data_detect_type = {
        'breaker/breaker': 'breaker',
        'junction_box/terminal': 'junction_box',
        'line_marker/line_marker': 'line_marker'
    }

    for result in match_results.values():
        # outputs_match
        # # device
        device_info = {
            'device_type': result['device_type'],
            'nameplate': {
                'meta': {
                    'detect_type': f'{result["device_type"]}/{result["device_type"]}',
                    'track_id': result['track_id'],
                },
                'value': result['model_type']
            }
        }

        # # location
        location_info = {
            'meta': {
                'detect_type': 'loc_marker/loc_marker',
                'track_id': -1
            },
            'value': result['loc_marker']
        }

        device_type = result["device_type"]

        # # connection
        connection_info = []
        for terminal_key, line_data in result['line_marker'].items():
            connection_info.append(
                {
                    'terminal': {
                        'meta': {
                            'detect_type': f'{device_type}/terminal',
                            'track_id': -1 if 'breaker' == device_type else line_data['device_track_id']
                        },
                        'value': terminal_key
                    },
                    'wire': {
                        'meta': {
                            'detect_type': 'line_marker/line_marker',
                            'track_id': line_data['line_track_id']
                        },
                        'value': line_data['text']
                    }
                }
            )

        outputs_match.append({
            'location': location_info,
            'device': device_info,
            'connections': connection_info
        })

        # outputs_data
        # # get object data of device
        detect_type = device_info['nameplate']['meta']['detect_type']
        track_id = int(device_info['nameplate']['meta']['track_id'])
        if detect_type in oputputs_data_detect_type and -1 != track_id:
            if detect_type not in outputs_data:
                outputs_data[detect_type] = {}

            target_type = oputputs_data_detect_type[detect_type]
            outputs_data[detect_type][track_id] = item_tracks[target_type][track_id]

        # # get object data of connections
        for conn_info in connection_info:
            # terminal
            detect_type = conn_info['terminal']['meta']['detect_type']
            track_id = int(conn_info['terminal']['meta']['track_id'])
            if detect_type in oputputs_data_detect_type and -1 != track_id:
                if detect_type not in outputs_data:
                    outputs_data[detect_type] = {}

                target_type = oputputs_data_detect_type[detect_type]
                outputs_data[detect_type][track_id] = item_tracks[target_type][track_id]
            # wire
            detect_type = conn_info['wire']['meta']['detect_type']
            track_id = int(conn_info['wire']['meta']['track_id'])
            if detect_type in oputputs_data_detect_type and -1 != track_id:
                if detect_type not in outputs_data:
                    outputs_data[detect_type] = {}

                target_type = oputputs_data_detect_type[detect_type]
                outputs_data[detect_type][track_id] = item_tracks[target_type][track_id]

    return (outputs_match, outputs_data)


class WiringMatch:
    def __init__(
        self,
        match_win: T.Tuple[int, int],
        nameplate_config: T.Dict,
        terminal_config: T.Dict,
        score_thresh: float = 0.8,
        track_count_thresh: int = 5,
        txt_count_thresh: int = 3,
        mask_for_match: bool = True,
    ):
        self.win_left, self.win_right = match_win
        self.nameplate_config = nameplate_config
        self.terminal_config = terminal_config
        self.score_thresh = score_thresh
        self.track_count_thresh = track_count_thresh
        self.txt_count_thresh = txt_count_thresh
        if mask_for_match:
            self.find_terminal_line = self.find_terminal_line_mask
        else:
            self.find_terminal_line = self.find_terminal_line_box

    def generate_outputs_data(self, outputs_match: T.List[T.Dict], item_tracks: T.Dict) -> T.Dict:
        outputs_data = {}

        for match in outputs_match:
            # 获取丝印数据
            nameplate_meta = match['device']['nameplate']['meta']
            if -1 != nameplate_meta['track_id']:
                detect_type = nameplate_meta['detect_type']
                track_id = nameplate_meta['track_id']
                if detect_type not in outputs_data:
                    outputs_data[detect_type] = {}
                nameplate_type = detect_type.split('/')[0]
                outputs_data[detect_type][track_id] = item_tracks[nameplate_type][track_id]

            for conn in match['connections']:
                # 获取引脚数据
                terminal_meta = conn['terminal']['meta']
                if -1 != terminal_meta['track_id']:
                    detect_type = terminal_meta['detect_type']
                    track_id = terminal_meta['track_id']
                    if detect_type not in outputs_data:
                        outputs_data[detect_type] = {}
                    terminal_type = detect_type.split('/')[0]
                    outputs_data[detect_type][track_id] = item_tracks[terminal_type][track_id]

                # 获取引线数据
                wire_meta = conn['wire']['meta']
                detect_type = wire_meta['detect_type']
                track_id = wire_meta['track_id']
                if detect_type not in outputs_data:
                    outputs_data[detect_type] = {}
                line_type = detect_type.split('/')[0]
                outputs_data[detect_type][track_id] = item_tracks[line_type][track_id]

        return outputs_data

    def find_terminal_line_mask(self, angle: float, point: T.Tuple[int, int], line_items: T.List[T.Tuple[int, InspectItem]]) -> int:
        '''
        根据引脚中心点坐标，引脚接线角度，引线的 mask_xy，找到与引脚匹配的引线

        Args:
            angle: 引脚与引线的角度
            point: 引脚中心点坐标
            line_items: 引线 track_id, item
        Returns:
            idx: 引线的索引，如果为-1，则表示没有找到匹配的引线
        '''
        px, py = point  # 记引脚中心点为 p

        # 计算一般式直线方程 ax + by + c = 0，找到与角度同向，且在直线上的一个点，记为 q
        if angle in [0, 180, -180]:
            # 直线方程 y = py, y - py = 0
            a, b, c = 0, 1, -py

            # 确定 q 坐标
            qy = py
            if 0 == angle:
                qx = px + 1
            else:
                qx = px - 1
        elif angle in [90, -90]:
            # 直线方程 x = px, x - px = 0
            a, b, c = 1, 0, -px

            # 确定 q 坐标
            qx = px
            if 90 == angle:
                qy = py - 1
            else:
                qy = py + 1
        else:
            # 点斜式直线方程：y - py = k(x - px)，kx - y + py - kpx = 0
            k = math.tan(math.radians(angle))
            a, b, c = k, -1, py - k * px

            # 确定 q 坐标
            if -90 < angle < 90:
                qx = px + 1
                qy = py + k
            else:
                qx = px - 1
                qy = py - k
        # logger.info(f'一般式直线方程：(a,b,c) = ({a}, {b}, {c}), point: {point}, angle: {angle}, q point: ({qx}, {qy})')

        # 在 mask_xy 中取第一个点 m，计算 v = pq * pm
        # 若 v >= 0，则 mask_xy 与 a 同向，是候选目标引线；否则，反向
        same_side_line_idx = []
        for idx, (_, item) in enumerate(line_items):
            mx, my = item.mask_xy[0]
            if (mx - px) * (qx - px) + (my - py) * (qy - py) >= 0:
                same_side_line_idx.append(idx)

        if 0 == len(same_side_line_idx):
            # logger.info(f'No lines are in the same direction with terminal')
            return -1

        # 把 mask_xy 中的每一个点带入直线方程，记结果为 n
        # 若 n 中同时存在大于和小于0的值，表示以引脚中心引出的角度为angle的射线穿过 mask_xy ，则 mask_xy 是候选目标引线
        candidate_line_idx = []
        for idx in same_side_line_idx:
            # mask_xy = line_items[idx][1].mask_xy
            all_mask_xy = line_items[idx][1].mask_xy
            # 用一半 mask_xy 匹配，距离引脚中心点太远的 mask_xy 可能因为引线弯曲而不准确
            y_refer = all_mask_xy[:, 1].mean()
            if angle > 0:
                # y_refer = np.percentile(all_mask_xy[:, 1], 50)
                mask_xy = all_mask_xy[all_mask_xy[:, 1] >= y_refer]
            else:
                # y_refer = np.percentile(all_mask_xy[:, 1], 50)
                mask_xy = all_mask_xy[all_mask_xy[:, 1] <= y_refer]
            v = mask_xy[:, 0] * a + mask_xy[:, 1] * b + c
            greater_than_0_num = sum(v > 0)
            less_than_0_num = sum(v < 0)
            if greater_than_0_num > 0 and less_than_0_num > 0:
                candidate_line_idx.append(idx)
                # logger.info(f'Line mask contain the terminal ray! total num: {len(v)}, >0 num: {greater_than_0_num}, <0 num: {less_than_0_num} ')
            # else:
            #     logger.info(f'---------- Line mask does not contain the terminal ray! total num: {len(v)}, >0 num: {greater_than_0_num}, <0 num: {less_than_0_num} ')

        if 0 == len(candidate_line_idx):
            # logger.info(f'No lines mask contain the terminal ray')
            return -1
        elif 1 == len(candidate_line_idx):
            return candidate_line_idx[0]
        else:
            pass

        # 此时，若候选目标引线存在多个，则计算 mask_xy 中 每一个点与 p 的距离，记录最小距离
        # 最小距离中的最小者，即为目标引线
        matched_line_idx = -1
        min_terminal_line_distance = math.inf
        for idx in candidate_line_idx:
            mask_xy = line_items[idx][1].mask_xy
            # all_mask_xy = line_items[idx][1].mask_xy
            # cy = all_mask_xy[:, 1].mean()
            # if angle > 0:
            #     mask_xy = all_mask_xy[all_mask_xy[:, 1] >= cy]
            # else:
            #     mask_xy = all_mask_xy[all_mask_xy[:, 1] <= cy]
            distance = (mask_xy[:, 0] - px) ** 2 + (mask_xy[:, 1] - py) ** 2
            min_dist = distance.min()

            if min_dist < min_terminal_line_distance:
                min_terminal_line_distance = min_dist
                matched_line_idx = idx

        return matched_line_idx


    def find_terminal_line_box(self, angle: float, point: T.Tuple[int, int], line_items: T.List[T.Tuple[int, InspectItem]]) -> int:
        '''
        根据引脚中心点坐标，引脚接线角度，引线的 box，找到与引脚匹配的引线

        Args:
            angle: 引脚与引线的角度
            point: 引脚中心点坐标
            line_items: 引线 track_id, item
        Returns:
            idx: 引线的索引，如果为-1，则表示没有找到匹配的引线
        '''
        px, py = point  # 记引脚中心点为 p

        # 计算一般式直线方程 ax + by + c = 0，找到与角度同向，且在直线上的一个点，记为 q
        if angle in [0, 180, -180]:
            # 直线方程 y = py, y - py = 0
            a, b, c = 0, 1, -py

            # 确定 q 坐标
            qy = py
            if 0 == angle:
                qx = px + 1
            else:
                qx = px - 1
        elif angle in [90, -90]:
            # 直线方程 x = px, x - px = 0
            a, b, c = 1, 0, -px

            # 确定 q 坐标
            qx = px
            if 90 == angle:
                qy = py - 1
            else:
                qy = py + 1
        else:
            # 点斜式直线方程：y - py = k(x - px)，kx - y + py - kpx = 0
            k = math.tan(math.radians(angle))
            a, b, c = k, -1, py - k * px

            # 确定 q 坐标
            if -90 < angle < 90:
                qx = px + 1
                qy = py + k
            else:
                qx = px - 1
                qy = py - k
        # logger.info(f'一般式直线方程：(a,b,c) = ({a}, {b}, {c}), point: {point}, angle: {angle}, q point: ({qx}, {qy})')

        # 在 box_xyxy 中取左上角顶点 m，计算 v = pq * pm
        # 若 v >= 0，则 mask_xy 与 a 同向，是候选目标引线；否则，反向
        same_side_line_idx = []
        for idx, (_, item) in enumerate(line_items):
            mx, my = item.box_xyxy[:2]
            if (mx - px) * (qx - px) + (my - py) * (qy - py) >= 0:
                same_side_line_idx.append(idx)

        if 0 == len(same_side_line_idx):
            # logger.info(f'No lines are in the same direction with terminal')
            return -1

        # 把 box_xyxy 中的左上和右下顶点带入直线方程，
        # 若二者相乘小于0，则表示以引脚中心引出的角度为 angle 的射线穿过 box，则对应引线是候选目标引线
        candidate_line_idx = []
        for idx in same_side_line_idx:
            x1, y1, x2, y2 = line_items[idx][1].box_xyxy
            xy1_v = x1 * a + y1 * b + c
            xy2_v = x2 * a + y2 * b + c
            box_text = line_items[idx][1].text_words

            if xy1_v * xy2_v < 0:
                candidate_line_idx.append(idx)
                # logger.info(f'Line mask contain the terminal ray! xy1_v: {xy1_v}, xy2_v: {xy2_v}, box_text: {box_text}, idx: {idx}')
            # else:
            #     logger.info(f'---------- Line mask does not contain the terminal ray! xy1_v: {xy1_v}, xy2_v: {xy2_v}, box_text: {box_text}, idx: {idx}')

        # logger.info(f'candidate_line_idx: {candidate_line_idx}')
        if 0 == len(candidate_line_idx):
            # logger.info(f'No lines mask contain the terminal ray')
            return -1
        elif 1 == len(candidate_line_idx):
            return candidate_line_idx[0]
        else:
            pass

        # 此时，若候选目标引线存在多个，则计算 mask_xy 中 每一个点与 p 的距离，记录最小距离
        # 最小距离中的最小者，即为目标引线
        matched_line_idx = -1
        min_terminal_line_distance = math.inf
        for idx in candidate_line_idx:
            mask_xy = line_items[idx][1].mask_xy
            distance = (mask_xy[:, 0] - px) ** 2 + (mask_xy[:, 1] - py) ** 2
            min_dist = distance.min()

            if min_dist < min_terminal_line_distance:
                min_terminal_line_distance = min_dist
                matched_line_idx = idx

        return matched_line_idx

    def terminal_line_match(
        self, terminal_type: str, item_tracks: T.Dict, item_orc_merge_results: T.Dict
    ) -> T.Dict:
        ''' 对应器件引脚检测模式，引脚被模型检测出来，引脚与引线匹配 '''
        line_item_tracks = deepcopy(item_tracks['line_marker'])

        matched_results = {}

        for terminal_track_id, terminal_items in item_tracks[terminal_type].items():
            # 必须非空
            if 0 == len(terminal_items):
                continue

            terminal_target_type = terminal_type
            terminal_track_id = terminal_track_id
            terminal_text = item_orc_merge_results[terminal_type][terminal_track_id]

            matched_lines = defaultdict(list)
            for term_item in terminal_items:  # 出现在多帧中的同一个引脚
                x1, y1, x2, y2 = term_item.box_xyxy
                if self.win_left < x1 and x2 < self.win_right:  # 确保引脚区域必须在匹配窗口内
                    # 找到 terminal 对应的 frame 中所有的 line
                    frame_line_items: T.Tuple[int, InspectItem] = []
                    for line_track_id, line_items in line_item_tracks.items():
                        for li_item in line_items:
                            if term_item.frame_idx == li_item.frame_idx:
                                frame_line_items.append((line_track_id, li_item))

                    # logger.info(f'terminal track_id: {terminal_track_id} in match window, got frame_line_items num: {len(frame_line_items)}')
                    if 0 == len(frame_line_items):
                        continue

                    angle = self.terminal_config['detect']['angle']
                    # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 以 box 的中心点作为引脚中心点

                    mask_xy = term_item.mask_xy
                    # 以 mask 的中心点作为引脚中心点
                    cx, cy = mask_xy.mean(axis=0)
                    # # 以 mask 25 和 75 分位数之间的值计算引脚中心点
                    # per_25, per_75 = np.percentile(mask_xy[:, 1], (25, 75))
                    # mask_for_cxy = np.logical_and(mask_xy[:, 1] >= per_25, mask_xy[:, 1] <= per_75)
                    # cx_2575, cy_2575 = mask_xy[mask_for_cxy].mean(axis=0)

                    # for li_track_id, li_item in frame_line_items:
                    #     line_target_type = li_item.target_type
                    #     logger.info(
                    #         f'terminal(track_id, text): ({terminal_track_id}, {terminal_text}), '
                    #         f'frame_line_items line(track_id, text, text_words): ({li_track_id}, {item_orc_merge_results[line_target_type][li_track_id]}, {li_item.text_words})\n'
                    #         f'term (cx, cy): ({(x1+x2)//2}, {(y1+y2)//2}) mask_cxy: {mask_xy.mean(axis=0)} mask_cxy_2575: ({cx_2575}, {cy_2575}), line box: {li_item.box_xyxy}'
                    #         # f'line box: {li_item.box_xyxy}, mask: {li_item.mask_xy}'
                    #     )

                    matched_line_idx = self.find_terminal_line(angle, (cx, cy), frame_line_items)
                    if -1 != matched_line_idx:
                        _, line_item = frame_line_items[matched_line_idx]

                        # # 从 line_item_tracks 中去除找到匹配到的 line item
                        # line_item_tracks.pop(line_track_id, None)

                        line_target_type = line_item.target_type
                        line_track_id = line_item.track_id
                        line_text = item_orc_merge_results[line_target_type][line_track_id]

                        # TODO: 出现 line_text 与 line_item.text_words 不一致的情况，估计是因为 track 不精确导致
                        matched_lines[line_text].append((line_target_type, line_track_id))
                        # logger.info(f'\nmatched_line_idx: {matched_line_idx}, terminal(track_id, text): ({terminal_track_id}, {terminal_text}), line(track_id, text, text_words): ({line_track_id}, {line_text}, {line_item.text_words})\n')

                        # TODO: 是否找到一个匹配后 就跳出循环
                        # break

            # logger.info(f'terminal track_id: {terminal_track_id}, text: {terminal_text}, matched_lines: {matched_lines}')
            if 0 == len(matched_lines):
                continue

            # 获取最佳匹配
            # 一个引脚匹配一根引线，
            # 如果 terminal_text 出现在 li_text 中，则其是最佳匹配；
            # 否则，匹配到最多 line_track_id 的是最佳匹配
            # TODO: 是否强制 terminal_text 出现在 li_text 中？强制的话，能提升精准率！
            # TODO: 如果 terminal_text 是 *，li_text 是正常的，是否将 li_text 中分号后的值赋值给 terminal_text ?
            max_num = 0
            line_text = ''
            line_target_type = ''
            line_track_id = -1
            for li_text, type_ids in matched_lines.items():
                if terminal_text in li_text:
                    line_text = li_text
                    line_target_type, line_track_id = type_ids[0]
                    break

                if len(type_ids) > max_num and ('*' != li_text):  # 出现次数最多的 li_text 作为最终结果，确保引线标签字符不是*
                    line_text = li_text
                    line_target_type, line_track_id = type_ids[0]  # 取第一个结果
                    max_num = len(type_ids)

            # 未找到最佳匹配
            if -1 == line_track_id:
                continue

            # 找到最佳匹配
            # 从 line_item_tracks 中去除找到匹配到的 line item
            for _, line_track_id in matched_lines[line_text]:
                line_item_tracks.pop(line_track_id, None)

            matched_results[terminal_track_id] = {
                'terminal': {
                    'meta': {
                        'detect_type': f'{terminal_target_type}/terminal',
                        'track_id': terminal_track_id
                    },
                    'value': line_text.split(':')[-1] if '*' == terminal_text else terminal_text  # terminal_text
                },
                'wire': {
                    'meta': {
                        'detect_type': f'{line_target_type}/{line_target_type}',
                        'track_id': line_track_id
                    },
                    'value': line_text
                },
            }

        return matched_results

    def nameplate_line_match(
        self, device_type: str, nameplate_type: str, item_tracks: T.Dict, item_orc_merge_results: T.Dict
    ) -> T.List[T.Dict]:
        ''' 对应器件引脚推断模式，铭牌被模型检测出来，根据铭牌与引脚的相对位置，引脚与引线匹配 '''
        line_item_tracks = deepcopy(item_tracks['line_marker'])

        matched_results = []
        for nameplate_track_id, nameplate_items in item_tracks[nameplate_type].items():
            if 0 == len(nameplate_items):
                continue

            nameplate_name = item_orc_merge_results[nameplate_type][nameplate_track_id]
            nameplate_name = nameplate_name.replace(' ', '').upper()

            if nameplate_name not in self.terminal_config['infer']:
                # logger.warning(f'Recognized {nameplate_name} is not in terminal_config')
                continue

            match = {
                'location': {},
                'device': {
                    'device_type': device_type,
                    'nameplate': {
                        'meta': {
                            'detect_type': f'{device_type}/{nameplate_type}',
                            'track_id': nameplate_track_id
                        },
                        'value': item_orc_merge_results[nameplate_type][nameplate_track_id]
                    }
                },
                'connections': [],
            }

            # logger.info(f'nameplate_track_id: {nameplate_track_id}, items num: {len(nameplate_items)}')
            np_connections = defaultdict(list)
            for np_idx, np_item in enumerate(nameplate_items):
                x1, _, x2, _ = np_item.box_xyxy
                if self.win_left < x1 and x2 < self.win_right:
                    # 找到 nameplate 对应的 frame 中所有的 line
                    frame_line_items: T.Tuple[int, InspectItem] = []
                    for line_track_id, line_items in line_item_tracks.items():
                        for li_item in line_items:
                            if np_item.frame_idx == li_item.frame_idx:
                                frame_line_items.append((line_track_id, li_item))
                    # logger.info(f'nameplate_track_id: {nameplate_track_id}, frame_idx: {np_item.frame_idx}, line items num: {len(frame_line_items)}')

                    if 0 == len(frame_line_items):
                        continue

                    # 配置文件中铭牌对应的 pattern
                    pattern = self.terminal_config['infer'][nameplate_name]
                    # 当前帧检测到的铭牌数据到配置文件中的铭牌数据的仿射变换
                    transformed_terminals = affine_transform_pattern(pattern, np_item)

                    # 在同一个 frame 中找到每一个引脚对应的引线
                    for terminal_name, terminal in transformed_terminals.items():
                        cx, cy = terminal['center']
                        angle = terminal['angle']

                        matched_line_idx = self.find_terminal_line(angle, (cx, cy), frame_line_items)
                        if -1 != matched_line_idx:
                            line_track_id, line_item = frame_line_items[matched_line_idx]

                            # # 从 line_item_tracks 中去除找到匹配到的 line item
                            # line_item_tracks.pop(line_track_id, None)

                            line_target_type = line_item.target_type
                            line_track_id = line_item.track_id
                            line_text = item_orc_merge_results[line_target_type][line_track_id]

                            terminal_target_type = nameplate_type  # 跟 nameplate_type 相同
                            terminal_track_id = -1  # 推断的引脚没有 track_id
                            terminal_text = terminal_name

                            np_connections[np_idx].append({
                                'terminal': {
                                    'meta': {
                                        'detect_type': f'{terminal_target_type}/terminal',
                                        'track_id': terminal_track_id
                                    },
                                    'value': terminal_text
                                },
                                'wire': {
                                    'meta': {
                                        'detect_type': f'{line_target_type}/{line_target_type}',
                                        'track_id': line_track_id
                                    },
                                    'value': line_text
                                }
                            })

            # 找到最好的匹配
            # 一般丝印区域宽度较大，一个丝印对应的多个引脚在不同帧里找到匹配的引线，因此这里需要合并操作
            # 将所有的匹配合并，在同类项中选择最优者，最终得到最好的匹配
            # TODO: 是否强制 terminal_text 出现在 wire_text 中？强制的话，能提升精准率！
            # logger.info(f'nameplate track_id: {nameplate_track_id}, matched: {np_connections}')
            connections = {}  # 最好的匹配，以引脚名为 key，以引脚引线对为 value
            for conns in np_connections.values():
                for term_wire in conns:
                    terminal_text = term_wire['terminal']['value']
                    wire_text = term_wire['wire']['value']
                    if terminal_text not in connections:
                        connections[terminal_text] = term_wire
                    else:
                        wire_text_old = connections[terminal_text]['wire']['value']
                        if '*' == wire_text_old:  # 引脚名是确定的，但引线标签可能是 *
                            connections[terminal_text] = term_wire
                        else:
                            if terminal_text in wire_text:  # 正常情况下，引脚名包含在引线标签中
                                connections[terminal_text] = term_wire
            # connections 按 terminal_text 排序
            connections = sorted(connections.items(), key=lambda x: x[0])
            match['connections'] = [conn[1] for conn in connections]
            matched_results.append(match)

            # 从 line_item_tracks 中去除找到匹配到的 line item
            for _, conn in connections:
                line_track_id = conn['wire']['meta']['track_id']
                line_item_tracks.pop(line_track_id, None)
        return matched_results

    def __call__(self, device_type: str, frame_results: T.List[T.List[InspectItem]]) -> T.Dict:
        item_orc_merge_results = merge_ocr_thru_frames(
            frame_results, self.score_thresh, self.track_count_thresh, self.txt_count_thresh
        )

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

        # 默认返回结果
        results = {
            'match_result': [],
            'outputs_match': [],
            'outputs_data': {},
            'ocr_result': item_orc_merge_results,
            'track_result': item_tracks,
        }

        if 'line_marker' not in item_tracks:
            return results

        # 获取铭牌配置
        nameplate_detect = False
        nameplate_type = ''
        if isinstance(self.nameplate_config['detect'], dict):
            nameplate_detect = True
            nameplate_type = self.nameplate_config['detect']['nameplate_type']

        # 获取引脚配置
        terminal_detect = False
        terminal_type = ''
        if 'detect' in self.terminal_config:
            terminal_detect = True
            terminal_type = self.terminal_config['detect']['terminal_type']

        if (not nameplate_detect) and terminal_detect and (terminal_type in item_tracks):  # 引脚检测
            # 获取 引脚与引线匹配结果
            matched_results = self.terminal_line_match(terminal_type, item_tracks, item_orc_merge_results)

            # 确定位置标签
            # 当前通过引线的字符获取位置标签，这种方式可能因为引线的字符识别错误导致位置标签识别错误
            # TODO: 编写更好的位置标签匹配规则，跟佟哥讨论
            locations = defaultdict(list)
            for term_track_id, term_wire_match in matched_results.items():
                wire_value = term_wire_match['wire']['value']
                loc_text = wire_value.split(':')[0]
                if loc_text.startswith('-'):
                    loc_text = loc_text[1:]
                locations[loc_text].append(term_track_id)

            # 确定输出的匹配结果
            outputs_match: T.List[T.Dict] = []
            for loc_text, terminal_track_ids in locations.items():
                connections = []
                for term_track_id in terminal_track_ids:
                    connections.append(matched_results[term_track_id])

                outputs_match.append({
                    'location': {
                        'meta': {
                            'detect_type': 'loc_marker/loc_marker',
                            'track_id': -1,
                        },
                        'value': loc_text
                    },
                    'device': {
                        'device_type': device_type,
                        'nameplate': {
                            'meta': {
                                'detect_type': f'{device_type}/{terminal_type}',
                                'track_id': -1
                            },
                            'value': ""
                        },
                    },
                    'connections': connections
                })
            # 确定输出的数据
            outputs_data = self.generate_outputs_data(outputs_match, item_tracks)
        elif nameplate_detect and (nameplate_type in item_tracks) and (not terminal_detect):  # 引脚推断
            # 获取 器件，引脚与引线匹配结果
            matched_results = self.nameplate_line_match(device_type, nameplate_type, item_tracks, item_orc_merge_results)

            # TODO: 如果 nameplate_type 是 breaker，且检测到有 ELM，则 ELM 前面的主件不考虑与 ELM 相同的引脚

            # 确定位置标签
            # 当前通过引线的字符获取位置标签，这种方式可能因为引线的字符识别错误导致位置标签识别错误
            # TODO: 编写更好的位置标签匹配规则，跟佟哥讨论
            for match in matched_results:
                # location 默认值
                match['location'] = {
                    'meta': {
                        'detect_type': 'loc_marker/loc_marker',
                        'track_id': -1
                    },
                    'value': '*'
                }

                connections = match['connections']
                if len(connections) > 0:
                    locations = []
                    for conn in connections:
                        wire_value = conn['wire']['value']
                        loc_text = wire_value.split(':')[0]
                        if loc_text.startswith('-'):
                            loc_text = loc_text[1:]
                        locations.append(loc_text)
                    match['location']['value'] = Counter(locations).most_common(1)[0][0]

            # 确定输出的匹配结果
            outputs_match: T.List[T.Dict] = matched_results
            # 确定输出的数据
            outputs_data = self.generate_outputs_data(outputs_match, item_tracks)
        else:
            pass

        return {
            'outputs_match': outputs_match,
            'outputs_data': outputs_data,
            'ocr_result': item_orc_merge_results,
            'track_result': item_tracks,
            'matched_results': matched_results
        }


def wiring_match(
    frame_results: T.List[T.List[InspectItem]],
    pattern_map: T.Dict,
    frame_shape: T.Tuple[int, int],
    valid_win_width: int,
    device_types: T.Tuple[str] = ("breaker", "junction_box"),
    score_thresh: float = 0.8,
    track_count_thresh: int = 5,
    txt_count_thresh: int = 3,
):
    item_orc_merge_results = merge_ocr_thru_frames(
        frame_results, score_thresh, track_count_thresh, txt_count_thresh
    )

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
                pattern = search_pattern(
                    pattern_map, item_orc_merge_results[device_type][device_id]
                )
            else:
                pattern = None
            device_pattern_map[device_type][device_id] = pattern
            if pattern is not None:
                for terminal_key in pattern["terminals"]:
                    row_indices.append((device_type, device_id, terminal_key))

    if "line_marker" not in item_tracks:
        return {
            "match_result": [],
        }

    # 使用pandas dataframe来存储距离矩阵，row key为(device_type, device_id, terminal_key)，col key为(line_marker_id)
    dist_matrix = pd.DataFrame(
        np.ones((len(row_indices), len(item_tracks["line_marker"])), np.float32)
        * 10000,
        index=pd.MultiIndex.from_tuples(
            row_indices, names=["device_type", "device_id", "terminal_key"]
        ),
        columns=item_tracks["line_marker"].keys(),
    )

    # 计算 任一种 target_type 的 item box center 与 line_marker 的距离
    # # 这里并没有直接使用 item box center 进行计算，
    # # 而是使用 item target_type 对应的 pattern terminals center,
    # # 将其经过 transform_pattern 后与 line_marker 分割结果最下/上面的点计算距离
    w, h = frame_shape
    pattern_debug_info = {}
    for frame in frame_results:
        frame_item_map = {
            key: list(items) for key, items in groupby(frame, lambda x: x.target_type)
        }
        for item in frame:
            if item.target_type in device_types:
                if item.track_id not in item_tracks[item.target_type]:
                    continue
                if item.track_id not in device_pattern_map[item.target_type]:
                    continue
                pattern = device_pattern_map[item.target_type][item.track_id]
                if pattern is None:
                    continue
                win_width = valid_win_width
                win_left = (w - win_width) // 2
                win_right = (w + win_width) // 2
                if win_left < item.box_xyxy[0] and item.box_xyxy[2] < win_right:
                    pattern = transform_pattern(pattern, item)
                    pattern_debug_info[
                        (item.target_type, item.frame_idx, item.track_id)
                    ] = pattern
                    for terminal_key, tmnl in pattern["terminals"].items():
                        if "line_marker" not in frame_item_map:
                            continue
                        for lm in frame_item_map["line_marker"]:
                            if lm.track_id not in item_tracks["line_marker"]:
                                continue
                            if lm.box_xyxy[1] < (frame_shape[1] // 2):
                                pinpoint = lm.bottom_point
                            else:
                                pinpoint = lm.top_point
                            dist = np.sqrt(
                                np.sum(
                                    np.square(
                                        np.array(tmnl["center"]) - np.array(pinpoint)
                                    )
                                )
                            )
                            if (
                                dist < tmnl["min_dist"]
                                and dist
                                < dist_matrix.loc[
                                    (item.target_type, item.track_id, terminal_key),
                                    lm.track_id,
                                ]
                            ):
                                dist_matrix.loc[
                                    (item.target_type, item.track_id, terminal_key),
                                    lm.track_id,
                                ] = dist

    # 距离之和最小的指派， 将 line_marker 指派给对应 target_type 的 item
    row_ind, col_ind = linear_sum_assignment(dist_matrix.values)

    # TODO 根据device的类型，匹配loc_marker

    match_results = {}
    for ri, ci in zip(row_ind, col_ind):
        device_type, device_id, terminal_key = dist_matrix.index[ri]
        lm_id = dist_matrix.columns[ci]
        if (device_type, device_id) not in match_results:
            match_results[(device_type, device_id)] = {
                "device_type": device_type,
                "model_type": item_orc_merge_results[device_type][device_id],
                "loc_marker": "*",  # TODO 这里需要补充
                "line_marker": {},
                "track_id": device_id,
            }
        if dist_matrix.loc[(device_type, device_id, terminal_key), lm_id] < 1000:
            match_results[(device_type, device_id)]["line_marker"][
                terminal_key
            ] = {"line_track_id": lm_id, "device_track_id": device_id, "text": item_orc_merge_results["line_marker"][lm_id]}

    # loc_marker 临时方案
    for device_type, device_id in match_results:
        match_res = match_results[(device_type, device_id)]
        loc_list = []
        for lm_match in match_res["line_marker"].values():
            lm_text = lm_match['text']
            loc_part = lm_text.split(":")[0]
            if loc_part.startswith("-"):
                loc_part = loc_part[1:]
            loc_list.append(loc_part)
        if loc_list:
            match_res["loc_marker"] = Counter(loc_list).most_common(1)[0][0]

    # 端子排，按照位置编号合并结果
    if match_results and list(match_results.keys())[0][0] == "junction_box":
        loc2devices = {}
        for device_type, device_id in match_results:
            loc = match_results[(device_type, device_id)]["loc_marker"]
            if loc not in loc2devices:
                loc2devices[loc] = []
            loc2devices[loc].append(match_results[(device_type, device_id)])

        match_results = {}
        for loc, devices in loc2devices.items():
            match_results[loc] = {
                "device_type": devices[0]["device_type"],
                "model_type": "",
                "loc_marker": loc,
                "line_marker": {},
                "track_id": -1,
            }
            for device in devices:
                if "1" in device["line_marker"]:
                    match_results[loc]["line_marker"][device["model_type"]] = device[
                        "line_marker"
                    ]["1"]

    # 生成输出结果
    outputs_match, outputs_data = generate_output(match_results, item_tracks)

    return {
        "match_result": list(match_results.values()),
        "ocr_result": item_orc_merge_results,
        "track_result": item_tracks,
        "pattern_debug_info": pattern_debug_info,
        "dist_matrix": dist_matrix,
        "outputs_match": outputs_match,
        "outputs_data": outputs_data
    }