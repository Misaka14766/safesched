#!/usr/bin/env python3
"""
资源监控模块 - 支持软上限和硬上限
"""

import time
import threading
import psutil
import subprocess
import logging
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """资源监控器 - 支持软上限和硬上限"""
    
    def __init__(
        self,
        gpu_ids: List[int] = None,
        thresholds: Dict = None,
        update_interval: int = 5,
        enforce_interval: int = 2
    ):
        self.gpu_ids = gpu_ids or []
        self.update_interval = update_interval
        self.enforce_interval = enforce_interval
        
        # 初始化阈值
        if thresholds is None:
            self.thresholds = {
                'cpu': {'soft': 80, 'hard': 95},
                'mem': {'soft': 70, 'hard': 90},
                'disk': {'soft': 85, 'hard': 95},
                'io': {'soft': 80, 'hard': 95},
                'gpu_mem': {'soft': 70, 'hard': 85}
            }
        else:
            # 确保阈值格式正确
            self.thresholds = {}
            for key in ['cpu', 'mem', 'disk', 'io', 'gpu_mem']:
                if key in thresholds:
                    if isinstance(thresholds[key], dict):
                        self.thresholds[key] = thresholds[key]
                    else:
                        # 兼容旧格式
                        self.thresholds[key] = {'soft': thresholds[key], 'hard': thresholds[key] + 10}
                else:
                    self.thresholds[key] = {'soft': 80, 'hard': 95}
        
        self._stop = False
        self._enforce_stop = False
        
        self._status = {
            "cpu": 0.0, "mem": 0.0, "disk": 0.0, "io": 0.0,
            "gpu_mem": {g: 0.0 for g in self.gpu_ids},
            "overloaded_soft": False,
            "overloaded_hard": False
        }
        self._lock = threading.Lock()
        
        # 硬上限强制回调
        self._hard_limit_callback: Optional[Callable] = None
        
        # 启动线程
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._enforce_thread = threading.Thread(target=self._enforce_loop, daemon=True)
        self._monitor_thread.start()
        self._enforce_thread.start()
    
    def set_hard_limit_callback(self, callback: Callable):
        """设置硬上限触发时的回调函数"""
        self._hard_limit_callback = callback
    
    def _get_disk_io(self) -> float:
        try:
            r = subprocess.run(
                "iostat -x 1 2 | grep -E '^nvme|^sd' | tail -n1",
                shell=True, capture_output=True, text=True, timeout=8
            )
            if r.returncode == 0:
                return float(r.stdout.strip().split()[-1])
        except Exception:
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
        except Exception:
            pass
        return 0.0
    
    def _update_status(self):
        """更新资源使用状态"""
        with self._lock:
            self._status["cpu"] = psutil.cpu_percent(interval=0.1)
            self._status["mem"] = psutil.virtual_memory().percent
            self._status["disk"] = psutil.disk_usage('/').percent
            self._status["io"] = self._get_disk_io()
            
            for g in self.gpu_ids:
                self._status["gpu_mem"][g] = self._get_gpu_mem(g)
            
            # 检查软上限
            self._status["overloaded_soft"] = (
                self._status["cpu"] > self.thresholds['cpu']['soft'] or
                self._status["mem"] > self.thresholds['mem']['soft'] or
                self._status["disk"] > self.thresholds['disk']['soft'] or
                self._status["io"] > self.thresholds['io']['soft']
            )
            
            # 检查硬上限
            self._status["overloaded_hard"] = (
                self._status["cpu"] > self.thresholds['cpu']['hard'] or
                self._status["mem"] > self.thresholds['mem']['hard'] or
                self._status["disk"] > self.thresholds['disk']['hard'] or
                self._status["io"] > self.thresholds['io']['hard']
            )
    
    def _monitor_loop(self):
        """资源监控主循环"""
        while not self._stop:
            try:
                self._update_status()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(2)
    
    def _enforce_loop(self):
        """硬上限强制检查循环"""
        while not self._enforce_stop:
            try:
                if self.is_overloaded_hard():
                    if self._hard_limit_callback:
                        self._hard_limit_callback()
                time.sleep(self.enforce_interval)
            except Exception as e:
                logger.error(f"强制检查循环异常: {e}")
                time.sleep(1)
    
    def is_overloaded_soft(self) -> bool:
        """检查是否达到软上限"""
        with self._lock:
            return self._status["overloaded_soft"]
    
    def is_overloaded_hard(self) -> bool:
        """检查是否达到硬上限"""
        with self._lock:
            return self._status["overloaded_hard"]
    
    def is_gpu_overloaded_soft(self, gpu_id: int) -> bool:
        """检查GPU是否达到软上限"""
        with self._lock:
            return self._status["gpu_mem"].get(gpu_id, 0) > self.thresholds['gpu_mem']['soft']
    
    def is_gpu_overloaded_hard(self, gpu_id: int) -> bool:
        """检查GPU是否达到硬上限"""
        with self._lock:
            return self._status["gpu_mem"].get(gpu_id, 0) > self.thresholds['gpu_mem']['hard']
    
    def get_available_resources(self) -> Dict[str, float]:
        """
        获取当前剩余可用资源
        
        Returns:
            各资源剩余量字典: {'cpu': 剩余%, 'mem': 剩余%, 'disk': 剩余%, ...}
        """
        with self._lock:
            result = {
                'cpu': max(0, self.thresholds['cpu']['hard'] - self._status['cpu']),
                'mem': max(0, self.thresholds['mem']['hard'] - self._status['mem']),
                'disk': max(0, self.thresholds['disk']['hard'] - self._status['disk']),
                'io': max(0, self.thresholds['io']['hard'] - self._status['io'])
            }
            
            for gpu in self.gpu_ids:
                result[f'gpu_{gpu}_mem'] = max(
                    0, self.thresholds['gpu_mem']['hard'] - self._status['gpu_mem'][gpu]
                )
            
            return result
    
    def get_current_usage(self) -> Dict[str, float]:
        """获取当前资源使用状态"""
        with self._lock:
            return {
                'cpu': self._status['cpu'],
                'mem': self._status['mem'],
                'disk': self._status['disk'],
                'io': self._status['io'],
                'gpu_mem': dict(self._status['gpu_mem'])
            }
    
    def get_summary(self) -> str:
        """获取资源状态摘要"""
        with self._lock:
            cpu = self._status['cpu']
            mem = self._status['mem']
            io_status = self._status['io']
            cpu_hard = self.thresholds['cpu']['hard']
            mem_hard = self.thresholds['mem']['hard']
            
            return (f"CPU={cpu:.0f}/{cpu_hard}% MEM={mem:.0f}/{mem_hard}% "
                    f"IO={io_status:.0f}%")
    
    def shutdown(self):
        """关闭监控"""
        self._stop = True
        self._enforce_stop = True
        self._monitor_thread.join(timeout=5)
        self._enforce_thread.join(timeout=5)
