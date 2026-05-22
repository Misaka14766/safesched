import time
import threading
import psutil
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, gpu_ids: List[int] = None, thresholds: dict = None):
        self.gpu_ids = gpu_ids or []
        self.thresholds = thresholds or {
            'cpu': 90,
            'mem': 70,
            'disk': 90,
            'io': 90,
            'gpu_mem': 75
        }
        self._stop = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        
        self._status = {
            "cpu": 0.0, "mem": 0.0, "disk": 0.0, "io": 0.0,
            "gpu_mem": {g: 0.0 for g in self.gpu_ids},
            "overloaded": False
        }
        self._lock = threading.Lock()
        self._thread.start()

    def _get_disk_io(self) -> float:
        try:
            r = subprocess.run(
                "iostat -x 1 2 | grep -E '^nvme|^sd' | tail -n1",
                shell=True, capture_output=True, text=True, timeout=8
            )
            if r.returncode == 0:
                return float(r.stdout.strip().split()[-1])
        except:
            pass
        return 0.0

    def _get_gpu_mem(self, gpu_id: int) -> float:
        try:
            r = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), 
                 "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                used, total = map(int, r.stdout.strip().split(','))
                return (used / total) * 100
        except:
            pass
        return 0.0

    def _update_status(self):
        with self._lock:
            self._status["cpu"] = psutil.cpu_percent(interval=0.1)
            self._status["mem"] = psutil.virtual_memory().percent
            self._status["disk"] = psutil.disk_usage('/').percent
            self._status["io"] = self._get_disk_io()
            
            for g in self.gpu_ids:
                self._status["gpu_mem"][g] = self._get_gpu_mem(g)
            
            # 资源阈值检查
            self._status["overloaded"] = (
                self._status["cpu"] > self.thresholds['cpu'] or
                self._status["mem"] > self.thresholds['mem'] or
                self._status["disk"] > self.thresholds['disk'] or
                self._status["io"] > self.thresholds['io']
            )

    def _monitor_loop(self):
        while not self._stop:
            try:
                self._update_status()
                time.sleep(5)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(2)

    def is_overloaded(self) -> bool:
        with self._lock:
            return self._status["overloaded"]

    def is_gpu_overloaded(self, gpu_id: int) -> bool:
        with self._lock:
            return self._status["gpu_mem"].get(gpu_id, 0) > self.thresholds['gpu_mem']

    def get_summary(self) -> str:
        with self._lock:
            return (f"CPU={self._status['cpu']:.0f}% MEM={self._status['mem']:.0f}% "
                    f"IO={self._status['io']:.0f}%")

    def shutdown(self):
        self._stop = True
        self._thread.join(timeout=5)