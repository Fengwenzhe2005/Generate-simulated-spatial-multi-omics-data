生成模拟的空间多组学数据

简介
- 本目录包含用于生成模拟空间多组学（RNA / ATAC / Protein）数据的脚本与示例数据。

快速开始
- 安装依赖：pip install -r requirements.txt
- 运行脚本：python simulations/sim.py
- 或在笔记本中交互运行：simulations/generate_omics.ipynb

目录结构
- simulations/: 核心模拟脚本与示例数据
- utils/: 预处理与辅助函数
- requirements.txt: 依赖列表

备注
- 示例输出位于：simulations/simulations/dataset_triple_omics/
- 推荐 Python 3.8+ 环境