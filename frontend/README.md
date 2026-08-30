# Spacecraft — AI 推理训练平台

Electron + React + TypeScript 桌面客户端，用于 AI 模型推理与训练的本地化管理。

## 项目结构

```
Spacecraftfrontend/
├── index.html                        # Vite 入口 HTML
├── package.json                      # 项目配置、依赖、构建脚本
├── vite.config.ts                    # Vite 构建配置（React 插件、路径别名）
├── vitest.config.ts                  # Vitest 测试配置（jsdom 环境）
├── tsconfig.json                     # TypeScript 配置（renderer）
├── tsconfig.electron.json            # TypeScript 配置（main/preload）
├── tailwind.config.ts                # Tailwind CSS 主题（颜色、字体、动画）
├── postcss.config.js                 # PostCSS 配置（Tailwind + Autoprefixer）
├── .eslintrc.cjs                     # ESLint 配置（TS + React Hooks）
├── prettier.config.cjs               # Prettier 格式化配置
├── .prettierignore / .eslintignore   # 忽略目录
├── check_progress.py                 # 任务进度检查脚本
├── progress.json                     # 任务执行状态（152 任务）
├── scripts/
│   └── electron-dev.js               # Dev 模式启动 Electron 脚本
├── tests/
│   └── unit/                         # 单元测试（每个 task 对应一个文件）
├── docs/
│   └── plans/                        # 全部任务计划文档（11 模块 × 152 任务）
├── src/
│   ├── main/                         # Electron 主进程
│   │   ├── index.ts                  #   主入口：窗口创建、菜单、生命周期
│   │   ├── ipc/
│   │   │   ├── index.ts              #   旧版入口（兼容）
│   │   │   ├── register-all.ts       #   IPC handler 统一注册入口
│   │   │   ├── system.ipc.ts         #   窗口控制 + CUDA 状态 IPC
│   │   │   ├── dataset.ipc.ts        #   数据集 CRUD IPC
│   │   │   ├── inference.ipc.ts      #   推理任务 IPC（进度推送）
│   │   │   ├── training.ipc.ts       #   训练任务 IPC（配置/进度）
│   │   │   ├── model.ipc.ts          #   模型管理 IPC
│   │   │   └── statistics.ipc.ts     #   统计数据聚合 IPC
│   │   └── lib/
│   │       └── storage.ts            #   JSON 文件持久化层（read/write/update/delete）
│   ├── preload/
│   │   ├── index.ts                  #   contextBridge: 暴露 30+ 个 IPC 方法到渲染进程
│   │   └── index.d.ts                #   TypeScript 类型声明（Window.electronAPI）
│   ├── renderer/                     # React 渲染进程
│   │   ├── main.tsx                  #   React 入口（createRoot → render App）
│   │   ├── App.tsx                   #   根组件：HashRouter + AppShell + 7 条路由
│   │   ├── lib/
│   │   │   └── ipc-client.ts         #   渲染进程 IPC 调用封装（ipcCall<T> 泛型）
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── TitleBar.tsx      #     毛玻璃顶栏（44px、窗口拖拽、系统指示灯、窗口控制按钮）
│   │   │   │   ├── Sidebar.tsx       #     毛玻璃侧栏（240px、7 导航项、Logo、版本号）
│   │   │   │   └── AppShell.tsx      #     全局布局（TitleBar + Sidebar + main 内容区）
│   │   │   └── common/
│   │   │       ├── StatusDot.tsx     #     状态圆点（completed/running/failed/pending + 发光）
│   │   │       ├── GlassPanel.tsx    #     毛玻璃面板容器（标题 + actions + 内容）
│   │   │       ├── StatCard.tsx      #     统计卡片（图标/数值/变化量/hover 动效）
│   │   │       ├── ProgressBar.tsx   #     进度条（渐变 fill + 呼吸动画 + 4 色 variant）
│   │   │       ├── LogViewer.tsx     #     日志查看器（终端风格 + mono 字体 + 自动滚动）
│   │   │       ├── DataTable.tsx     #     通用数据表格（列配置/行点击/空状态/hover 高亮）
│   │   │       ├── EmptyState.tsx    #     空状态组件（图标 + 标题 + 描述 + 操作按钮）
│   │   │       └── PageHeader.tsx    #     页面标题（标题 + 描述 + actions 插槽）
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx         #     工作台（统计卡片 + 训练进度 + 推理任务表）
│   │   │   ├── DataCollection.tsx    #     数据采集（搜索/导入/类型标记/删除）
│   │   │   ├── Inference.tsx         #     推理模块（选数据/选模型/进度/结果指标）
│   │   │   ├── InferenceHistory.tsx  #     推理历史（筛选/详情/重跑/导出/删除）
│   │   │   ├── Training.tsx          #     训练模块（远程/本地切换/配置/进度/CUDA）
│   │   │   ├── ModelManagement.tsx   #     模型管理（搜索/导入/删除/默认/详情）
│   │   │   ├── Statistics.tsx        #     统计分析（6 图表占位 + 2×3 网格）
│   │   │   └── index.ts             #     页面模块导出
│   │   ├── stores/
│   │   │   ├── app.store.ts          #     全局应用状态（暗色模式/侧栏折叠）
│   │   │   ├── dashboard.store.ts    #     工作台状态（统计/CUDA/任务列表/刷新）
│   │   │   ├── dataset.store.ts      #     数据集状态（CRUD/搜索/选择/过滤）
│   │   │   ├── inference.store.ts    #     推理状态（活跃任务/队列/结果/进度）
│   │   │   ├── inference-history.store.ts  # 推理历史状态（列表/筛选/详情/分页）
│   │   │   ├── training.store.ts     #     训练状态（模式/配置/进度/系统状态）
│   │   │   ├── model.store.ts        #     模型状态（列表/搜索/过滤/选择/CRUD）
│   │   │   └── statistics.store.ts   #     统计状态（数据/日期范围/加载）
│   │   └── styles/
│   │       ├── theme.css             #     CSS 变量（背景/表面/文字/语义色/动画）+ shadcn 暗色
│   │       └── global.css            #     Tailwind 指令 + 滚动条 + 玻璃态工具类
│   └── shared/
│       └── types/
│           ├── index.ts              #     统一导出
│           ├── ipc.ts                #     IPC_CHANNELS 常量 + IpcChannelName 类型
│           ├── ipc-result.ts         #     IPCResult<T> 泛型 + IPCError 类型
│           ├── dataset.ts            #     Dataset 接口
│           ├── inference.ts          #     InferenceRequest/ProgressEvent/Result 类型
│           ├── training.ts           #     TrainingConfig/ProgressEvent/SystemStatus 类型
│           ├── model.ts              #     Model/ModelMetrics/ModelStatus 类型
│           └── statistics.ts         #     StatisticsQuery/StatisticsData 类型
```

## 核心文件说明

### 入口文件

| 文件 | 作用 |
|------|------|
| `index.html` | Vite 入口，加载 Google Fonts 和 `src/renderer/main.tsx` |
| `src/main/index.ts` | Electron 主进程入口：创建无边框 1440×1000 窗口、注册所有 IPC handlers |
| `src/renderer/main.tsx` | React 渲染入口：挂载 `<App />` 到 `#root` |
| `src/renderer/App.tsx` | HashRouter + AppShell + 7 条 `<Route>` 配置 |

### 布局组件

| 文件 | 尺寸 | 功能 |
|------|------|------|
| `TitleBar.tsx` | 44px 高 | 毛玻璃顶栏、`WebkitAppRegion: drag`、系统状态灯、窗口控制按钮 |
| `Sidebar.tsx` | 240px 宽 | 毛玻璃侧栏、Logo、7 个 NavLink、激活态 accent 指示条、版本号 |
| `AppShell.tsx` | 全屏 | flex-col 布局：TitleBar + flex-row(Sidebar + main) |

### 页面组件

| 文件 | 路由 | 核心功能 |
|------|------|----------|
| `Dashboard.tsx` | `#/` | 4 个统计卡片 + 训练进度 + 推理任务表格 |
| `DataCollection.tsx` | `#/data` | 搜索/导入/类型切换/删除，带确认弹窗 |
| `Inference.tsx` | `#/inference` | 选数据 + 选模型 → 开始推理 → 进度条 + 日志 + 6 项指标 |
| `InferenceHistory.tsx` | `#/inference-history` | 多维度筛选/详情抽屉/重跑/导出/删除 |
| `Training.tsx` | `#/training` | 远程/本地切换/超参数编辑/进度动画 |
| `ModelManagement.tsx` | `#/models` | 搜索/导入/删除/详情面板/指标展示 |
| `Statistics.tsx` | `#/statistics` | 2×3 图表网格（6 个图表占位） |

### 通用 UI 组件

| 组件 | Props | 样式 |
|------|-------|------|
| `StatusDot` | `status`, `label?` | 8px 圆点 + box-shadow 发光 + pulse 动画 |
| `GlassPanel` | `title?`, `actions?`, `children`, `className?` | 毛玻璃容器：border-radius 20px + blur(10px) |
| `StatCard` | `label`, `value`, `change?`, `icon` | hover 上浮 2px + shadow + mono 数字 |
| `ProgressBar` | `value`, `showGlow?`, `variant?` | accent 渐变 / 4 色 variant / 呼吸动画 |
| `LogViewer` | `logs`, `autoScroll?`, `maxHeight?` | 终端暗色背景 + 绿色文字 + JetBrains Mono |
| `DataTable<T>` | `columns`, `data`, `onRowClick?` | 表头大写/bold、行 hover 高亮、空状态 |
| `EmptyState` | `icon`, `title`, `description?`, `action?` | 图标 48px muted + 居中 + 操作按钮 |
| `PageHeader` | `title`, `description?`, `actions?` | h1 26px 700 + flex 水平布局 |

### Zustand 状态管理

| Store 文件 | 管理范围 | 核心 State |
|-----------|---------|-----------|
| `app.store.ts` | 全应用 | isDark, isSidebarCollapsed |
| `dashboard.store.ts` | 工作台 | stats(4 Cards), cudaStatus, trainingTasks, recentInferences, refreshDashboard |
| `dataset.store.ts` | 数据集 | datasets[], searchQuery, selectedDataset, CRUD + getFilteredDatasets |
| `inference.store.ts` | 推理 | activeTask, queue[], result, isRunning, addLog + push |
| `inference-history.store.ts` | 推理历史 | items[], filters, selectedItem, isDrawerOpen, page/pageSize |
| `training.store.ts` | 训练 | mode(remote/local), config, progress, systemStatus, isRunning |
| `model.store.ts` | 模型 | models[], selectedModel, searchQuery, sourceFilter, CRUD |
| `statistics.store.ts` | 统计 | data(6 sections), dateRange, isLoading |

### 共享类型

| 文件 | 导出 |
|------|------|
| `ipc.ts` | `IPC_CHANNELS`（30+ 通道常量 as const）、`IpcChannelName` 类型 |
| `ipc-result.ts` | `IPCResult<T>` 可辨识联合、`IPCError` 类型 |
| `dataset.ts` | `Dataset` 接口（id/name/samples/format/type/createdAt） |
| `inference.ts` | `InferenceRequest`, `InferenceProgressEvent`, `InferenceResult` |
| `training.ts` | `TrainingConfig`, `TrainingProgressEvent`, `SystemStatus` |
| `model.ts` | `Model`, `ModelMetrics`, `ModelStatus`, `ModelSource` |
| `statistics.ts` | `StatisticsQuery`, `StatisticsData`（6 个数据数组） |

### IPC 层

| 文件 | 功能 |
|------|------|
| `register-all.ts` | 统一调用所有模块 `registerXxxHandlers()` |
| `system.ipc.ts` | window:minimize/maximize/close + system:cuda-status（mock GPU） |
| `dataset.ipc.ts` | dataset:list/import/update-type/delete，使用 storage 层持久化 |
| `inference.ipc.ts` | inference:run（mock 进度推送）/run-batch/cancel |
| `training.ipc.ts` | training:start-local（mock epoch 循环）/start-remote/stop/config-save/config-load |
| `model.ipc.ts` | model:list/import（dialog 选文件夹）/delete/set-default |
| `statistics.ipc.ts` | statistics:query（生成 6 段 mock 聚合数据） |
| `storage.ts` | readJSON/writeJSON/appendJSON/updateJSON/deleteJSON（userData 目录） |
| `ipc-client.ts` | `ipcCall<T>(channel, ...args)` 渲染进程 IPC 调用封装 |

### Preload 脚本

| 文件 | 功能 |
|------|------|
| `preload/index.ts` | contextBridge 暴露 30+ 个方法（窗口/数据集/推理/训练/模型/系统/统计 + push 监听器） |
| `preload/index.d.ts` | 全局类型声明：`Window.electronAPI: ElectronAPI` |

### 样式系统

| 文件 | 内容 |
|------|------|
| `theme.css` | 方案 C 暗色主题：33 个 CSS 变量（背景/表面/文字/accent/语义/边框/圆角/尺寸/字体/动画）+ @keyframes pulse-glow + shadcn .dark 覆写 |
| `global.css` | @tailwind 指令 + body 全局样式 + webkit 滚动条 + ::selection + focus-visible + .glass/.glass-panel/.glass-card 工具类 + .text-gradient/.no-drag |
| `tailwind.config.ts` | 15 颜色映射 CSS 变量 + Inter/JetBrains Mono 字体 + 4 级圆角 + glass blur + pulse-glow 动画 |

### 构建配置

| 文件 | 说明 |
|------|------|
| `tsconfig.json` | Renderer TypeScript（ESNext/bundler/jsx react-jsx/@别名） |
| `tsconfig.electron.json` | Main+Preload TypeScript（commonjs/ES2022/dist-electron 输出） |
| `vite.config.ts` | React 插件 / port 5173(strict) / base './' / @ 别名 |
| `vitest.config.ts` | globals + jsdom 环境 + tests/**/*.test.{ts,tsx} |

### 测试

- **位置**: `tests/unit/`
- **命名**: `task-{N}-{描述}.test.{ts|tsx}`
- **运行**: `npm test`（vitest run）
- **当前**: 13 test files, 122 tests PASSED

## 启动

```bash
# 安装依赖
npm install

# 开发模式（Vite + Electron 同时启动）
npm run dev

# 运行测试
npm test
```

## 构建 .exe

```bash
npm run build
```

这个命令依次执行：
1. `vite build` — 打包 React 前端到 `dist/`
2. `tsc -p tsconfig.electron.json` — 编译主进程+preload 到 `dist-electron/`
3. `electron-builder --win --publish=never` — 生成可执行文件

### 构建产物

| 文件 | 路径 | 说明 |
|------|------|------|
| NSIS 安装包 | `out/Spacecraft Setup x.x.x.exe` | 可自定义安装路径 |
| Portable 便携版 | `out/Spacecraft x.x.x.exe` | 无需安装直接运行 |

### 构建前提

| 条件 | 状态 |
|------|------|
| `vite.config.ts` 设置 `base: './'` | ✅ 已配置 |
| `tsconfig.electron.json` 编译通过 | ✅ |
| `dist-electron/` 输出就绪 | ✅ |
| `package.json` 的 `build` 字段完整 | ✅ |
| `assets/icon.ico` 应用图标 | ⚠️ 可选，缺失不阻塞 |

### 可选：分步构建

```bash
npx vite build                        # 仅打包前端
npx tsc -p tsconfig.electron.json     # 仅编译主进程
npx electron-builder --win --publish=never  # 仅生成 exe
```

## 技术栈

- **框架**: Electron 40 + React 18 + TypeScript 5
- **构建**: Vite 6 + electron-builder 26
- **样式**: Tailwind CSS 3 + 方案 C 暗色主题
- **状态**: Zustand 5
- **路由**: React Router v7 (HashRouter)
- **图表**: ECharts 5 (统计页面)
- **图标**: Lucide React
- **测试**: Vitest + Testing Library + Playwright
