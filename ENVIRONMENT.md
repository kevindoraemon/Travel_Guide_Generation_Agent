# 环境变量配置

项目不再在 `config.yml` 或 Python 源码中保存服务密钥。运行时需要以下三个环境变量：

- `OPENAI_API_KEY`：OpenAI 兼容模型服务密钥
- `TAVILY_API_KEY`：Tavily 搜索密钥
- `AMAP_MAPS_API_KEY`：高德地图 Web 服务密钥

Windows PowerShell 当前会话示例：

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:TAVILY_API_KEY = "your-key"
$env:AMAP_MAPS_API_KEY = "your-key"
```

Linux/macOS 示例：

```bash
export OPENAI_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
export AMAP_MAPS_API_KEY="your-key"
```

安装开发及测试依赖并执行测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`config.yml` 中的 `${NAME}` 仅是变量引用。变量未设置时会解析为空字符串，离线组件仍可使用；调用相应外部服务时需要先设置实际值。
