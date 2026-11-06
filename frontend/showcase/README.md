# 前端展示截图

截图分为两组，均由真实 Electron 前端窗口生成，并使用隔离的临时用户数据目录。

- `with-mock/`：带模拟数据，用于展示完整产品状态。
- `without-mock/`：不带模拟数据，用于展示首次启动和空状态。

重新生成：

```powershell
node tests/e2e/capture-showcase.mjs
```
