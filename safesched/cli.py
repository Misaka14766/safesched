#!/usr/bin/env python3
"""
safesched 主命令行入口
支持软上限和硬上限资源保护
"""

import argparse
import sys
import os
import subprocess
import shlex
import threading
import time
from queue import Queue
from typing import List, Tuple, Dict
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

from .config import Config, load_config, generate_config_template
from .monitor import ResourceMonitor

# 全局状态（线程安全版本）
stop_event = threading.Event()

# 跟踪活动进程（按启动时间排序，栈式管理）
active_processes: List[Tuple[int, str, float]] = []  # (pid, task_param, start_time)
active_processes_lock = threading.Lock()


def signal_handler(signum, frame):
    logger.info("\n收到中断信号，正在终止所有任务...")
    stop_event.set()
    
    with active_processes_lock:
        pids = [p[0] for p in active_processes]
    
    for pid in pids:
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception as e:
                    logger.warning(f"终止子进程失败: {e}")
            parent.kill()
        except Exception as e:
            logger.warning(f"终止进程失败: {e}")
    
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
    except Exception:
        pass
    return []


def kill_latest_process(monitor: ResourceMonitor):
    """
    按栈顺序kill最新启动的进程，直到资源低于硬上限
    
    Args:
        monitor: 资源监控器
    """
    global active_processes
    
    with active_processes_lock:
        if not active_processes:
            return
        
        # 复制列表避免修改原表时问题
        processes_to_check = list(active_processes)
    
    # 从最新开始遍历（栈顺序）
    for pid, task_param, _ in reversed(processes_to_check):
        try:
            logger.warning(f"⚠️ 硬上限触发，正在kill: {task_param} (PID={pid})")
            
            # 杀死进程及其子进程
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
            
            # 等待检查资源是否已回落
            time.sleep(1)
            
            if not monitor.is_overloaded_hard():
                logger.info("✅ 资源已回落至安全范围")
                return
        
        except Exception as e:
            logger.error(f"终止进程失败: {e}")


def worker_thread(
    gpu_id: int,
    task_queue: Queue,
    command_template: List[str],
    gpus: List[int],
    timeout: int,
    verbose: bool,
    thresholds: Dict,
    config: Config
):
    """工作线程"""
    monitor = ResourceMonitor(gpus, thresholds, config.monitor.update_interval, config.monitor.enforce_interval)
    is_batch_mode = '{}' in ' '.join(command_template)
    
    def enforce_hard_limit():
        """硬上限强制回调"""
        if not stop_event.is_set():
            kill_latest_process(monitor)
    
    monitor.set_hard_limit_callback(enforce_hard_limit)
    
    while not stop_event.is_set():
        try:
            # 软上限保护：达到软上限暂停新任务
            while (
                monitor.is_overloaded_soft() or 
                (gpu_id != -1 and monitor.is_gpu_overloaded_soft(gpu_id))
            ):
                if stop_event.is_set():
                    break
                logger.warning(f"系统负载 {monitor.get_summary()}，等待资源恢复...")
                for _ in range(60):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                if stop_event.is_set():
                    break
            
            if stop_event.is_set():
                break
            
            # 获取任务
            try:
                task_param, remaining_retries = task_queue.get_nowait()
            except Exception:
                time.sleep(0.5)
                continue
            
            # 构建命令（使用 shlex.quote 防止命令注入）
            cmd = []
            if is_batch_mode:
                escaped_param = shlex.quote(task_param)
                for part in command_template:
                    cmd.append(part.replace("{}", escaped_param))
            else:
                cmd = command_template.copy()
            
            # 设置环境变量
            env = os.environ.copy()
            if gpu_id != -1:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            
            if verbose:
                if is_batch_mode:
                    logger.info(f"{f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 开始处理: {task_param}")
                else:
                    logger.info(f"{f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 开始执行: {' '.join(cmd)}")
            
            success = False
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.DEVNULL if not verbose else None,
                    stderr=subprocess.DEVNULL if not verbose else None,
                    text=True
                )
                
                # 记录进程（按启动时间加入栈）
                start_time = time.time()
                with active_processes_lock:
                    active_processes.append((proc.pid, task_param, start_time))
                
                stdout, stderr = proc.communicate(timeout=timeout)
                
                if proc.returncode == 0:
                    success = True
                    if verbose:
                        logger.info(f"✅ {f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 完成: {task_param}")
                else:
                    if verbose:
                        logger.error(f"❌ {f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 失败: {task_param}")
                        if stdout:
                            logger.error(f"输出:\n{stdout}")
                        if stderr:
                            logger.error(f"错误:\n{stderr}")
            
            except subprocess.TimeoutExpired:
                logger.error(f"⏰ {f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 超时: {task_param}")
                try:
                    if proc:
                        parent = psutil.Process(proc.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except Exception as e:
                                logger.warning(f"终止子进程失败: {e}")
                        parent.kill()
                except Exception as e:
                    logger.error(f"终止进程失败: {e}")
            
            except Exception as e:
                logger.error(f"💥 {f'GPU{gpu_id}' if gpu_id != -1 else 'CPU'} 异常: {task_param} {e}")
            
            finally:
                # 从活动列表移除
                with active_processes_lock:
                    if proc and any(p[0] == proc.pid for p in active_processes):
                        active_processes = [p for p in active_processes if p[0] != proc.pid]
            
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
  # 批量处理视频列表
  cat video_list.txt | safesched python process_video.py {}
  
  # 指定GPU和并发数
  cat tasks.txt | safesched -g 0,1 -j 4 python process.py {}
  
  # 单次任务运行
  safesched python train_model.py
  
  # 使用配置文件
  safesched -c config.yaml python process.py {}
  
  # 生成配置文件模板
  safesched --generate-config my_config.yaml
"""
    )

    # 配置文件选项
    parser.add_argument("-c", "--config", type=str, help="配置文件路径 (YAML)")
    parser.add_argument("--generate-config", type=str, help="生成配置文件模板到指定路径")

    # 全局选项
    parser.add_argument("-g", "--gpus", type=lambda s: [int(x) for x in s.split(',')], 
                        help="可用GPU列表，逗号分隔 (默认自动检测)")
    parser.add_argument("-j", "--jobs", type=int, 
                        help="最大并发任务数 (默认自动检测)")
    parser.add_argument("-r", "--retries", type=int, 
                        help="最大重试次数 (默认2)")
    parser.add_argument("-t", "--timeout", type=int, 
                        help="单个任务超时 (秒，默认3600)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="详细日志输出")
    
    # 阈值覆盖选项
    parser.add_argument("--cpu-soft", type=int, help="CPU软上限 (%)")
    parser.add_argument("--cpu-hard", type=int, help="CPU硬上限 (%)")
    parser.add_argument("--mem-soft", type=int, help="内存软上限 (%)")
    parser.add_argument("--mem-hard", type=int, help="内存硬上限 (%)")
    parser.add_argument("--disk-soft", type=int, help="磁盘使用率软上限 (%)")
    parser.add_argument("--disk-hard", type=int, help="磁盘使用率硬上限 (%)")
    parser.add_argument("--io-soft", type=int, help="磁盘IO软上限 (%)")
    parser.add_argument("--io-hard", type=int, help="磁盘IO硬上限 (%)")
    parser.add_argument("--gpu-soft", type=int, help="GPU显存软上限 (%)")
    parser.add_argument("--gpu-hard", type=int, help="GPU显存硬上限 (%)")

    # 主命令：批量处理（默认）
    parser.add_argument("command", nargs=argparse.REMAINDER, help="命令模板，用{}表示任务参数")
    args = parser.parse_args()

    # 处理配置文件生成
    if args.generate_config:
        try:
            path = generate_config_template(args.generate_config)
            print(f"✅ 配置文件模板已生成: {path}")
            sys.exit(0)
        except Exception as e:
            print(f"❌ 生成配置文件失败: {e}")
            sys.exit(1)

    # 检查命令
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        sys.exit(1)
    
    # 命令行覆盖配置
    thresholds = {
        'cpu': config.thresholds.cpu.copy(),
        'mem': config.thresholds.mem.copy(),
        'disk': config.thresholds.disk.copy(),
        'io': config.thresholds.io.copy(),
        'gpu_mem': config.thresholds.gpu_mem.copy()
    }
    
    if args.cpu_soft is not None: thresholds['cpu']['soft'] = args.cpu_soft
    if args.cpu_hard is not None: thresholds['cpu']['hard'] = args.cpu_hard
    if args.mem_soft is not None: thresholds['mem']['soft'] = args.mem_soft
    if args.mem_hard is not None: thresholds['mem']['hard'] = args.mem_hard
    if args.disk_soft is not None: thresholds['disk']['soft'] = args.disk_soft
    if args.disk_hard is not None: thresholds['disk']['hard'] = args.disk_hard
    if args.io_soft is not None: thresholds['io']['soft'] = args.io_soft
    if args.io_hard is not None: thresholds['io']['hard'] = args.io_hard
    if args.gpu_soft is not None: thresholds['gpu_mem']['soft'] = args.gpu_soft
    if args.gpu_hard is not None: thresholds['gpu_mem']['hard'] = args.gpu_hard
    
    # 判断运行模式
    command_str = ' '.join(args.command)
    is_batch_mode = '{}' in command_str
    
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
    
    # 设置并发数和其他参数
    if args.jobs is None:
        args.jobs = config.scheduler.default_jobs or default_jobs
    if args.retries is None:
        args.retries = config.scheduler.default_retries or 2
    if args.timeout is None:
        args.timeout = config.scheduler.default_timeout or 3600
    
    logger.info(f"🚀 启动safesched，最大并发数: {args.jobs}")
    logger.info(f"⚙️  阈值配置: CPU={thresholds['cpu']['soft']}/{thresholds['cpu']['hard']}% "
                f"MEM={thresholds['mem']['soft']}/{thresholds['mem']['hard']}%")
    
    # 加载任务
    if is_batch_mode:
        logger.info("📥 读取任务列表...")
        tasks = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                tasks.append(line)
        
        if not tasks:
            logger.error("❌ 没有从标准输入读取到任务")
            sys.exit(1)
        
        logger.info(f"✅ 读取到 {len(tasks)} 个任务")
        
        task_queue = Queue()
        for task in tasks:
            task_queue.put((task, args.retries))
    else:
        logger.info("▶️ 单任务模式，直接执行命令")
        task_queue = Queue()
        task_queue.put(("", args.retries))
    
    # 启动工作线程
    threads = []
    
    if args.gpus:
        for gpu_id in args.gpus:
            for _ in range(args.jobs // len(args.gpus)):
                t = threading.Thread(
                    target=worker_thread,
                    args=(gpu_id, task_queue, args.command, args.gpus, 
                          args.timeout, args.verbose, thresholds, config),
                    daemon=True
                )
                t.start()
                threads.append(t)
    else:
        for _ in range(args.jobs):
            t = threading.Thread(
                target=worker_thread,
                args=(-1, task_queue, args.command, [], args.timeout,
                      args.verbose, thresholds, config),
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


if __name__ == "__main__":
    main()
