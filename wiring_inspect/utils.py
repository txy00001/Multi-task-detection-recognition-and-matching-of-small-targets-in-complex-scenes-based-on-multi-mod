from addict import Dict as ADDict
from loguru import logger
import os
import sys
import typing as T
import yaml


def setup_logger(output: str, name: T.Optional[str] = 'wiring_inspect'):
    '''
    Setup logger using loguru.
    
    Args:
        output: write the log file to the output directory.
        name: the logger name.
    '''
    if name is None:
        name = 'run'

    log_file_path = os.path.join(output, f'{name}.log')

    # 去除之前 add 的 handler  # 默认 handler
    logger.remove()

    # 配置文件日志
    logger.add(
        log_file_path,
        rotation='20 MB',  # 当日志文件大小超过该设置时，则在当前路径下创建一个新的日志文件
        retention='3 months',  # 配置一个日志文件最长保留3个月
        enqueue=True,  # 异步和多进程安全（默认为False，线程安全）
        level='DEBUG',
    )
    
    # 配置命令行日志
    logger.add(
        sys.stdout,
        enqueue=True,
        level='DEBUG'
    )


def cfg_format(cfg: T.Dict, level: int, out_str: str) -> str:
    '''
    将配置内容写成字符串
    '''
    for k, v in cfg.items():
        if not isinstance(v, dict):
            out_str += ('  ' * level + str(k) + ': ' + str(v) + '\n')
        else:
            out_str += ('  ' * level + str(k) + ':\n')
            out_str = cfg_format(v, level+1, out_str)
    return out_str


class DictNoDefault(ADDict):
    '''
    复写 missing 函数，如果不存在对应的 key，则返回 None
    '''
    def __missing__(self, key):
        return None


def setup_program(cfg_file_path: str) -> T.Dict:
    '''
    读取配置文件，配置logger，打印配置
    '''
    assert os.path.exists(cfg_file_path), f'File not found! {cfg_file_path}'

    # 读取配置文件
    with open(cfg_file_path, 'r') as fr:
        cfg = DictNoDefault(yaml.safe_load(fr))

    # 配置 logger
    # # 设置输出路径
    if cfg.output is None:
        output = './output'
    else:
        output = cfg.output
    output = os.path.abspath(output)

    if os.path.exists(output):
        assert os.path.isdir(output)
    else:
        os.makedirs(output)

    setup_logger(output)

    # 打印配置文件
    logger.info(f'Running program with the config:\n{cfg_format(cfg, 0, "")}')

    return cfg