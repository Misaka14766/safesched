safesched
=======

安全的资源感知任务调度器（safesched）——自动监控 CPU/内存/磁盘 IO/GPU 显存，过载时暂停调度，避免把机器跑崩。

作者与联系方式
- 作者: Misaka14766
- 邮箱: misaka14766@gmail.com
- 项目主页: https://github.com/Misaka14766/safesched

快速开始

安装（推荐虚拟环境）:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install .
```

示例用法

从 stdin 批量调度:

```bash
cat tasks.txt | safesched python process.py {}
```

运行单个命令（自动选择最空闲 GPU）:

```bash
safesched run python train.py
```

测试

推荐使用 `pytest` 进行单元测试。安装并运行：

```bash
pip install pytest
pytest -q
```

贡献

欢迎 Issue 与 PR。若需联系请使用上方邮箱或在 GitHub 上发起讨论。
