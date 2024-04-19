import redis

import dataclasses
import time

@dataclasses.dataclass
class Task:
    mid: str
    uuid: str
    type: str
    name: str

class TaskManager:
    """监听redis stream中的任务，将任务转换成Task对象
    """
    def __init__(self, redis_client: redis.Redis, task_stream_name: str, consumer_group: str, consumer_name: str, progress_stream_name: str):
        self._r = redis_client
        self._task_stream_name = task_stream_name
        self._consumer_group = consumer_group
        self._consumer_name = consumer_name
        self._progress_stream_name = progress_stream_name
        if self._consumer_group not in [g['name'] for g in self._r.xinfo_groups(self._task_stream_name)]:
            self._r.xgroup_create(self._task_stream_name, self._consumer_group, mkstream=True)
            self._r.xgroup_setid(self._task_stream_name, self._consumer_group, "$")
        self._last_id = "0-0"
    
    def get_task(self):
        """从redis stream中获取任务
        """
        task = None
        while True:
            # TODO
            # 1. 先检查pending任务，是否又没完成的任务
            # 2. 如果有没完成的任务，说明之前运行中断了，需要重新跑
            # 3. 如果没有没完成的任务，再从stream中读取任务
            
            task = self._r.xreadgroup(
                streams={self._task_stream_name: ">"},
                groupname=self._consumer_group,
                consumername=self._consumer_name,
                count=1,
                block=0,
            )[0][1]
            time.sleep(1)
            if task:
                break
        self._last_id = task[0][0]
        return Task(mid=task[0][0], **task[0][1])
    
    def update_task(self, task: Task, status: str, progress: float, message: str):
        """更新任务状态
        """
        self._r.xadd(
            self._progress_stream_name,
            {
                "uuid": task.uuid,
                "request_id": task.mid,
                "status": status,
                "progress": progress,
                "message": message,
            },
        )