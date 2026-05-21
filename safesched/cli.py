#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import threading
import time
from queue import Queue
from typing import List
import psutil
import signal

# 极简日志配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 全局状态
stop_requested = False
active_processes = set()

def signal_handler(signum, frame):
    global stop_requested
    logger.info("\n收到中断信号，正在终止所有任务...")
    stop_requested = True
    
    # 杀死所有活动子进程
    for pid in list(active_processes):
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except:
                    pass
            parent.kill()
        except:
            pass
    
    sys.exit(130)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def detect_gpus() -> List[int]:
    """自动检测NVIDIA GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
    except:
        pass
    return []

def get_idlest_gpu(gpus: List[int]) -> int:
    """获取最空闲的GPU"""
    if not gpus:
        return -1
    
    min_mem = 100.0
    idlest_gpu = gpus[0]
    
    for gpu in gpus:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                mem_usage = (used / total) * 100
                if mem_usage < min_mem:
                    min_mem = mem_usage
                    idlest_gpu = gpu
        except:
            pass
    
    return idlest_gpu

def worker_thread(
    gpu_id: int,
    task_queue: Queue,
    command_template: List[str],
    gpus: List[int],
    timeout: int,
    verbose: bool
):
    """工作线程"""
    from .monitor import ResourceMonitor
    monitor = ResourceMonitor(gpus)
    
    while not stop_requested:
        try:
            # 过载保护
            while monitor.is_overloaded() or (gpu_id != -1 and monitor.is_gpu_overloaded(gpu_id)):
                if stop_requested:
                    break
                logger.warning(f"系统过载 {monitor.get_summary()}，等待60秒...")
                for _ in range(60):
                    if stop_requested:
                        break
                    time.sleep(1)
                if stop_requested:
                    break
            
            if stop_requested:
                break
            
            # 获取任务
            try:
                task_param, remaining_retries = task_queue.get_nowait()
            except:
                time.sleep(0.5)
                continue
            
            # 构建命令
            cmd = []
            for part in command_template:
                cmd.append(part.replace("{}", task_param))
            
            # 设置环境变量
            env = os.environ.copy()
            if gpu_id != -1:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            
            if verbose:
                logger.info(f"{'GPU'+str(gpu_id) if gpu_id != -1 else 'CPU'} 开始处理: {task_param}")
            
            success = False
            try:
                proc = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE if not verbose else None,
                    stderr=subprocess.PIPE if not verbose else None,
                    text=True
                )
                
                active_processes.add(proc.pid)
                
                stdout, stderr = proc.communicate(timeout=timeout)
                
                if proc.returncode == 0:
                    success = True
                    if verbose:
                        logger.info(f"✅ {'GPU'+str(gpu_id) if gpu_id != -1 else 'CPU'} 完成: {task_param}")
                else:
                    if verbose:
                        logger.error(f"❌ {'GPU'+str(gpu_id) if gpu_id != -1 else 'CPU'} 失败: {task_param}")
                        if stdout:
                            logger.error(f"输出:\n{stdout}")
                        if stderr:
                            logger.error(f"错误:\n{stderr}")
            
            except subprocess.TimeoutExpired:
                logger.error(f"⏰ {'GPU'+str(gpu_id) if gpu_id != -1 else 'CPU'} 超时: {task_param}")
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except:
                            pass
                    parent.kill()
                except:
                    pass
            
            except Exception as e:
                logger.error(f"💥 {'GPU'+str(gpu_id) if gpu_id != -1 else 'CPU'} 异常: {task_param} {e}")
            
            finally:
                if proc.pid in active_processes:
                    active_processes.remove(proc.pid)
            
            # 重试逻辑
            if not success and remaining_retries > 0:
                logger.warning(f"🔄 任务失败，剩余重试次数: {remaining_retries} - {task_param}")
                time.sleep(30)
                task_queue.put((task_param, remaining_retries - 1))
            
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"工作线程异常: {e}")
            time.sleep(2)
    
    monitor.shutdown()

def main():
    parser = argparse.ArgumentParser(
        description="safesched - 安全的资源感知任务调度器，永远不会把你的机器跑崩",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量处理视频列表（自动检测GPU/CPU）
  cat video_list.txt | safesched python process_video.py {}
  
  # 指定GPU和并发数
  cat tasks.txt | safesched -g 0,1 -j 4 python process.py {}
  
  # 运行单个脚本（自动选最空闲GPU）
  safesched run python train_model.py
  
  # 任务超时1小时，重试3次
  cat big_tasks.txt | safesched -t 3600 -r 3 python process_big.py {}
        """
    )
    
    # 全局选项
    parser.add_argument("-g", "--gpus", type=lambda s: [int(x) for x in s.split(',')], 
                       help="可用GPU列表，逗号分隔（默认自动检测）")
    parser.add_argument("-j", "--jobs", type=int, 
                       help="最大并发任务数（默认GPU:8个/GPU，CPU:一半核心）")
    parser.add_argument("-r", "--retries", type=int, default=2, 
                       help="最大重试次数（默认2次）")
    parser.add_argument("-t", "--timeout", type=int, default=3600, 
                       help="单个任务超时时间(秒)（默认1小时）")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="详细日志输出")
    
    # 子命令
    subparsers = parser.add_subparsers(title="子命令", dest="subcommand")
    
    # run子命令：运行单个命令
    run_parser = subparsers.add_parser("run", help="运行单个命令，自动选择最空闲GPU")
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="要运行的命令")
    
    # 主命令：批量处理（默认）
    parser.add_argument("command", nargs=argparse.REMAINDER, help="命令模板，用{}表示任务参数")
    
    args = parser.parse_args()
    
    # 处理run子命令
    if args.subcommand == "run":
        if not args.command:
            logger.error("没有指定要运行的命令")
            sys.exit(1)
        
        gpus = detect_gpus()
        if gpus:
            gpu_id = get_idlest_gpu(gpus)
            logger.info(f"自动选择最空闲GPU: {gpu_id}")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        else:
            logger.info("未检测到GPU，使用CPU运行")
            env = os.environ.copy()
        
        env["OMP_NUM_THREADS"] = "1"
        
        try:
            result = subprocess.run(args.command, env=env)
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            sys.exit(130)
    
    # 批量处理模式
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 移除--分隔符（兼容旧语法）
    if args.command[0] == '--':
        args.command = args.command[1:]
    
    # 检查命令模板是否包含{}
    if '{}' not in ' '.join(args.command):
        logger.error("命令模板必须包含{}作为任务参数占位符")
        logger.error("示例: safesched python process.py {}")
        sys.exit(1)
    
    # 自动检测GPU和CPU
    if args.gpus is None:
        args.gpus = detect_gpus()
        if args.gpus:
            logger.info(f"✅ 自动检测到GPU: {args.gpus}")
            default_jobs = len(args.gpus) * 8
        else:
            logger.info("ℹ️ 未检测到NVIDIA GPU，使用CPU模式")
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 4
            default_jobs = max(1, cpu_count // 2)
            logger.info(f"✅ 自动检测到CPU核心数: {cpu_count}，默认并发数: {default_jobs}")
    else:
        default_jobs = len(args.gpus) * 8
    
    # 设置并发数
    if args.jobs is None:
        args.jobs = default_jobs
    
    logger.info(f"🚀 启动safesched，最大并发数: {args.jobs}")
    
    # 读取所有任务
    tasks = []
    logger.info("📥 读取任务列表...")
    for line in sys.stdin:
        line = line.strip()
        if line:
            tasks.append(line)
    
    if not tasks:
        logger.error("❌ 没有从标准输入读取到任务")
        sys.exit(1)
    
    logger.info(f"✅ 读取到 {len(tasks)} 个任务")
    
    # 创建任务队列
    task_queue = Queue()
    for task in tasks:
        task_queue.put((task, args.retries))
    
    # 启动工作线程
    threads = []
    
    if args.gpus:
        # GPU模式：每个GPU一个工作线程，内部管理并发
        for gpu_id in args.gpus:
            for _ in range(args.jobs // len(args.gpus)):
                t = threading.Thread(
                    target=worker_thread,
                    args=(gpu_id, task_queue, args.command, args.gpus, args.timeout, args.verbose),
                    daemon=True
                )
                t.start()
                threads.append(t)
    else:
        # CPU模式：多个工作线程
        for _ in range(args.jobs):
            t = threading.Thread(
                target=worker_thread,
                args=(-1, task_queue, args.command, [], args.timeout, args.verbose),
                daemon=True
            )
            t.start()
            threads.append(t)
    
    # 等待所有任务完成
    try:
        task_queue.join()
        logger.info("🎉 所有任务处理完成！")
    except KeyboardInterrupt:
        pass
    
    sys.exit(0)