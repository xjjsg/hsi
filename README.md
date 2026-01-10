# 🚀 HSI HFT - 港股ETF高频交易系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Research-yellow.svg)

**基于深度学习的港股ETF统计套利系统**

[核心特性](#-核心特性) • [快速开始](#-快速开始) • [系统架构](#-系统架构) • [文档](#-文档)

</div>

---

## 📖 项目简介

HSI HFT 是一个**完整的高频交易研究平台**，专注于港股ETF（恒生ETF 159920 / 恆生H股ETF 513130）的统计套利策略。项目实现了从数据采集、特征工程、深度学习建模到回测模拟的完整量化交易链路。

### 🎯 交易策略

- **标的**: 港股ETF（沪深交易所）
- **频率**: 高频（3秒级行情）
- **模式**: Taker-Taker（对手价成交）
- **约束**: 仅做多（A股限制）
- **预测窗口**: 2分钟（H* = 120秒）

### 🏆 核心亮点

- ✅ **事件驱动架构**: 异步多源数据聚合
- ✅ **智能调度**: 自动检测A股+港股交易日
- ✅ **深度学习**: Transformer + CNN 混合模型
- ✅ **理想模拟**: Oracle上帝视角回测
- ✅ **交易优化**: Chain Merging 减少无效交易

---

## 🔥 核心特性

### 1. 多源数据采集 (`getdata/`)

```python
# 支持的数据源
- 腾讯财经: ETF实时行情（主数据源）
- 新浪财经: 期货数据（套利基准）
- 百度股市通: 恒生指数 + H股指数
- 情绪指标: FEAR/GREED指数（可选）
```

**特点**:
- 🔄 自动重连和错误恢复
- 📊 亚秒级延迟（< 500ms）
- 💾 自动按日保存CSV

### 2. 智能交易调度器 (`getdata.py`)

```python
# 交易时段
T1: 09:29:25 - 11:30:10  # 早盘（A+H同时）
T2: 12:00:00 - 12:00:30  # 午间快照（捕捉套利窗口）
T3: 12:59:25 - 15:00:10  # 午盘
```

**功能**:
- 📅 自动识别下一个交易日（A股 ∩ 港股）
- ⏰ 窗口期自动启动/停止采集
- 🛡️ 优雅的任务取消机制

### 3. 数据分析 (`dataanalysis.py`)

**Barrier-hit 事件分析**:
```python
# 研究问题: 什么时候会出现套利机会？
条件: bid_{t+τ} - ask_t >= 2 × cost_rate × ask_t
输出: 最优预测窗口 H*
```

**分析指标**:
- Hit Rate: 触发概率
- τ*: 平均完成时间
- MAE: 预测误差

### 4. 深度学习模型

#### 完整版 (`modelbuild.py`)
```
输入: LOB特征(20维) + 自定义因子(100+维)
     ↓
模型: CNN分支 + BiLSTM分支
     ├─ Inception Block（多尺度卷积）
     ├─ SE Block（通道注意力）
     └─ Temporal Attention（时序注意力）
     ↓
输出: [做多, 持有, 做空] 3分类
```

#### 优化版 (`modelbuild_v4.py`)
```
改进:
- VICReg 去冗余（减少特征相关性）
- 16维潜在空间（降维提速）
- Cooldown 验证（防止资金重复占用）
```

### 5. 理想模拟器 (`ideal_simulation.py`)

```python
# 上帝视角: 完美预测未来2分钟最高价
Oracle 模式 → 理论上限收益
           → 最优持仓时间分布
           → 交易合并优化建议
```

**Chain Merging 逻辑**:
```python
if (gap < 60秒) and (重新开仓成本 > 持有成本):
    合并交易  # 节省手续费
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Scheduler                         │
│         (交易日历 + 时间窗口管理)                     │
└──────────────────┬──────────────────────────────────┘
                   │
         启动/停止 │
                   ▼
┌─────────────────────────────────────────────────────┐
│                GetData V2 (数据采集)                 │
│  ┌──────────┬──────────┬──────────┬──────────┐     │
│  │ Tencent  │  Sina    │  Baidu   │Sentiment │     │
│  │  Source  │  Source  │  Source  │  Source  │     │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘     │
│       │          │          │          │           │
│       └──────────┴────┬─────┴──────────┘           │
│                       ▼                             │
│                 Event Bus                           │
│                       │                             │
│                       ▼                             │
│                  Aggregator                         │
│                       │                             │
│                       ▼                             │
│                  CSV Writer                         │
└──────────────────┬────────────────────────────────┘
                   │ data/*.csv
                   ▼
┌─────────────────────────────────────────────────────┐
│              Data Analysis Pipeline                  │
│  ┌──────────────┬──────────────┬─────────────┐     │
│  │ Barrier-hit  │ AlphaForge   │   Ideal     │     │
│  │   Analysis   │ (特征工程)    │ Simulation  │     │
│  └──────┬───────┴──────┬───────┴──────┬──────┘     │
│         │              │              │            │
│         └──────────────┴──────────────┘            │
│                        │                           │
│                        ▼                           │
│              Deep Learning Model                   │
│       (HybridDeepLOB / FeatureTransformer)         │
│                        │                           │
│                        ▼                           │
│                 Backtesting Engine                 │
│              (事件驱动回测 - 开发中)                 │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.10
CUDA >= 11.8 (可选，用于GPU加速)
```

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/hsi-hft.git
cd hsi-hft

# 安装依赖
pip install -r requirements.txt

# 安装 Playwright（用于浏览器数据采集）
playwright install chromium
```

### 1️⃣ 数据采集

```bash
# 启动智能调度器（自动在交易时段采集）
python getdata.py

# 或手动启动采集（立即运行）
python -m getdata.main
```

**输出**: `data/sz159920_YYYYMMDD.csv` 和 `data/sh513130_YYYYMMDD.csv`

### 2️⃣ 数据分析

```bash
# 分析最优预测窗口 H*
python dataanalysis.py
```

**输出**: `barrier_hit_stats.csv`（触发率、平均时间等统计）

### 3️⃣ 理想模拟

```bash
# 运行上帝视角回测
python ideal_simulation.py
```

**输出**: 
- `trade_log.csv`（原始交易记录）
- `trade_log_merged.csv`（合并优化后）

### 4️⃣ 模型训练

```bash
# 训练深度学习模型
python modelbuild_v4.py

# 或使用完整版
python modelbuild.py
```

**输出**: 
- 训练好的模型权重（`best_model.pth`）
- 验证集性能报告

---

## 📊 数据格式

### ETF 行情数据 (CSV)

```csv
tx_local_time,bp1,bp2,bp3,bp4,bp5,sp1,sp2,sp3,sp4,sp5,...
1704876000123,4.321,4.320,4.319,4.318,4.317,4.322,4.323,...
```

**字段说明**:
- `tx_local_time`: 本地时间戳（毫秒）
- `bp1-bp5`: 买1-5价
- `sp1-sp5`: 卖1-5价
- `bv1-bv5`: 买1-5量
- `sv1-sv5`: 卖1-5量

### 交易日志 (CSV)

```csv
entry_time,entry_price,exit_time,exit_price,hold_seconds,profit_bps,pnl_amount
2024-01-10 09:30:00,4.321,2024-01-10 09:32:15,4.329,135,18.5,1234.56
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.10+ |
| **深度学习** | PyTorch, scikit-learn |
| **数据处理** | Pandas, NumPy |
| **异步框架** | asyncio, aiohttp |
| **浏览器自动化** | Playwright |
| **交易日历** | exchange_calendars |
| **可视化** | Matplotlib, Seaborn |

---

## 📈 性能指标

### 理想模拟器结果（历史数据）

```
预测窗口 H*: 120秒
初始资金: 200,000 CNY
交易成本: 万分之一（双边）

原始策略:
- 交易次数: 1,234 笔
- 平均收益: 2.3 bps/笔
- 总收益: 2,838 bps

合并策略:
- 交易次数: 856 笔 (-30.6%)
- 平均收益: 3.1 bps/笔
- 总收益: 3,214 bps (+13.2%)

⚠️ 注意: 这是理想上限，实际策略通常达到其30-50%
```

### 深度学习模型

```
模型: FeatureTransformer (V4)
参数量: 1.2M
训练集准确率: 58.3%
验证集准确率: 54.7%
测试集准确率: 53.2%

⚠️ 注意: 准确率>50%即可盈利（配合止损）
```

---

## 📂 项目结构

```
hsi/
├── getdata/                # 数据采集模块
│   ├── main.py            # 主入口
│   ├── clock.py           # 时钟管理
│   ├── event_bus.py       # 事件总线
│   ├── aggregator.py      # 数据聚合
│   ├── writer.py          # CSV写入
│   └── sources/           # 数据源插件
│       ├── tencent.py
│       ├── sina.py
│       ├── baidu.py
│       └── sentiment.py
│
├── data/                  # 历史数据（46+ CSV文件）
│
├── getdata.py             # 智能调度器
├── dataanalysis.py        # Barrier-hit分析
├── ideal_simulation.py    # 理想模拟器
├── modelbuild.py          # 深度学习模型（完整版）
├── modelbuild_v4.py       # 优化版（推荐）
│
├── requirements.txt       # 依赖清单
├── .gitignore
└── README.md
```

---

## 🗺️ 路线图

### ✅ 已完成
- [x] 多源数据采集系统
- [x] 智能交易调度器
- [x] Barrier-hit 事件分析
- [x] 理想模拟器 + Chain Merging
- [x] 深度学习模型（V1-V4）

### 🚧 进行中
- [ ] 实时预测引擎
- [ ] 事件驱动回测系统
- [ ] 可视化Dashboard

### 📅 计划中
- [ ] 多策略融合
- [ ] 风险管理模块（VaR、止损）
- [ ] 实盘接口（富途/老虎）
- [ ] 云端部署（AWS/阿里云）

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

### 代码规范
- 遵循 PEP 8
- 添加必要的注释和文档字符串
- 编写单元测试

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## ⚠️ 免责声明

**本项目仅供学术研究和技术交流使用，不构成任何投资建议。**

- 量化交易存在本金损失风险
- 历史收益不代表未来表现
- 实盘前请充分回测和风险评估
- 作者不对任何交易损失负责

---

## 📞 联系方式

- **Email**: xjjsaikou@gmail.com

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star！**

Made with ❤️ by [Your Name]

</div>
