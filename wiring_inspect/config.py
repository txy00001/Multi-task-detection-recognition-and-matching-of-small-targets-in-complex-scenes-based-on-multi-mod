from loguru import logger
from pathlib import Path
import yaml


class DeviceConfigRepo:
    def __init__(self, device_config_dir: str):
        self._device_config_dir = Path(device_config_dir)
        self._device_config = {}
        self._load_device_config()

    def _load_device_config(self):
        for conf_file in self._device_config_dir.glob("*.yml"):
            device_name = conf_file.stem

            with open(conf_file, "r") as f:
                try:
                    device_conf = yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"Error loading device config file {conf_file}: {e}")
                    continue

            self._device_config[device_name] = device_conf

    def get_config(self, device_name: str):
        return self._device_config.get(device_name, None)