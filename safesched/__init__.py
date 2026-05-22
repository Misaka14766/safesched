"""
safesched - 安全的资源感知任务调度器
永远不会把你的机器跑崩。零侵入，自动监控CPU/内存/磁盘IO/GPU显存，过载时自动暂停调度。

使用方式:
  cat tasks.txt | safesched python process.py {}
  safesched run python train.py
"""

__version__ = "0.1.1"
__author__ = "Misaka14766"
__email__ = "misaka14766@gmail.com"