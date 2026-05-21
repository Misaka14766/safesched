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

开发环境与测试

安装开发依赖（含测试工具）：

```bash
pip install -e ".[dev]"
```

运行测试（使用 pytest）：

```bash
pytest -v
```

运行特定测试文件：

```bash
pytest tests/test_monitor.py -v
pytest tests/test_cli.py -v
```

CI/CD

项目使用 GitHub Actions 自动进行：

- **Tests**: 在 Ubuntu、Windows、macOS 上对 Python 3.8~3.12 运行单元测试（`tests.yml`）
- **Publish**: 创建 Release 或推送 `v*` tag 时自动构建并发布到 PyPI（`publish.yml`）

要发布新版本：

1. 更新 `pyproject.toml` 中的 `version`
2. 提交更改并推送到 main
3. 创建 Release 或打 tag：
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
4. GitHub Actions 会自动构建并发布到 PyPI（需在仓库 Settings 中配置 `PYPI_API_TOKEN` secret）

贡献

欢迎 Issue 与 PR。若需联系请使用上方邮箱或在 GitHub 上发起讨论。
