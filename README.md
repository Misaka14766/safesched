# safesched

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![PyPI](https://img.shields.io/badge/PyPI-v0.1.0-blueviolet)

**安全的资源感知任务调度器** — 自动监控 CPU/内存/磁盘 IO/GPU 显存，过载时暂停调度，永远不会把你的机器跑崩。

## ✨ 主要特性

- 🔍 **自动资源监控**：实时监控 CPU、内存、磁盘 IO 和 GPU 显存
- 🛡️ **智能过载保护**：系统过载时自动暂停，避免宕机
- 🎯 **GPU 自动选择**：自动检测并选择最空闲的 GPU
- ♻️ **任务重试**：失败任务自动重试，可配置重试次数
- 📊 **并发管理**：灵活配置并发任务数和超时时间

## 📦 安装

### 推荐方式（使用 pipx，全局隔离安装）

```bash
pipx install safesched
```

### 或使用 pip（在虚拟环境中）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install safesched
```

### 从源码安装（开发者）

```bash
git clone https://github.com/Misaka14766/safesched.git
cd safesched
pip install -e .
```

## 🚀 快速开始

### 批量调度任务（从 stdin）

```bash
cat tasks.txt | safesched python process.py {}
```

其中 `tasks.txt` 每行一个参数，`{}` 作为占位符被替换。

### 运行单个命令

自动选择最空闲的 GPU 并执行：

```bash
safesched run python train_model.py
```

### 指定 GPU 并配置并发

```bash
cat tasks.txt | safesched -g 0,1 -j 8 -t 3600 python process.py {}
```

参数说明：
- `-g 0,1`：指定 GPU 列表（默认自动检测）
- `-j 8`：最大并发任务数（默认 8×GPU 数 或 CPU 核心数 / 2）
- `-t 3600`：单个任务超时时间（秒，默认 1 小时）
- `-r 3`：失败重试次数（默认 2 次）
- `-v`：详细日志输出

### 更多例子

处理 10000 个视频文件，自动检测 GPU，并发 4 个任务，每个任务超时 30 分钟：

```bash
ls *.mp4 | safesched -j 4 -t 1800 python encode.py {}
```

使用 CPU 模式处理任务（无 GPU）：

```bash
cat tasks.txt | safesched python lightweight_task.py {}
```

## 🧪 开发与测试

### 安装开发依赖

```bash
git clone https://github.com/Misaka14766/safesched.git
cd safesched
pip install -e ".[dev]"
```

### 运行测试

```bash
# 运行所有测试
pytest -v

# 运行特定测试模块
pytest tests/test_monitor.py -v
pytest tests/test_cli.py -v

# 显示覆盖率（需要 pytest-cov）
pytest --cov=safesched tests/
```

## 🔄 CI/CD 工作流

项目使用 **GitHub Actions** 自动进行持续集成和发布：

| 工作流 | 触发条件 | 功能 |
|--------|--------|------|
| **Tests** | Push 到 `main` 或 `develop`，PR | 在 Ubuntu/Windows/macOS 上测试 Python 3.8~3.12 |
| **Publish** | 创建 Release 或推送 `v*` tag | 自动构建并发布到 PyPI |

### 发布新版本

1. **更新版本号**
   ```bash
   # 编辑 pyproject.toml，修改 version = "x.y.z"
   ```

2. **提交并推送**
   ```bash
   git add pyproject.toml
   git commit -m "Release v0.2.0"
   git push origin main
   ```

3. **创建 Release（推荐）或打 tag**
   ```bash
   # 方式 1: 在 GitHub 上创建 Release（更正式）
   
   # 方式 2: 使用 git tag
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **GitHub Actions 自动处理**
   - 构建分发包（.whl 和 .tar.gz）
   - 运行 twine 检查元数据
   - 上传到 PyPI

> ⚠️ **注意**：首次发布需要在 GitHub 仓库 Settings 中配置 `PYPI_API_TOKEN` secret，获取方式：
> 1. 在 https://pypi.org 注册账号
> 2. 创建 API Token（Account Settings → API tokens）
> 3. 在 GitHub 仓库 Settings → Secrets → New repository secret，添加 `PYPI_API_TOKEN`

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 👤 作者

- **Misaka14766**
- 📧 [misaka14766@gmail.com](mailto:misaka14766@gmail.com)
- 🔗 [GitHub](https://github.com/Misaka14766/safesched)

## 💬 反馈与贡献

欢迎提交 Issue 和 Pull Request！如有任何问题或建议，请通过以下方式联系：

- 📍 [GitHub Issues](https://github.com/Misaka14766/safesched/issues)
- 📧 邮箱：misaka14766@gmail.com

---

**Made with ❤️ by Misaka14766**
