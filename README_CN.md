# OpenEvolve 中文使用指南

<div align="center">

<img src="openevolve-logo.png" alt="OpenEvolve Logo" width="400">

**🧬 最先进的开源代码进化工具**

*将你的大语言模型变成自主的代码优化器，发现突破性算法*

</div>

---

## 什么是 OpenEvolve？

OpenEvolve 是一个使用大语言模型（LLM）自动优化和进化代码的工具。它通过进化算法，让 AI 自动发现更好的算法实现，无需人工干预。

### 核心特点

- **自主发现**：AI 自动发现新算法，无需人工指导
- **实际效果**：在真实硬件上实现 2-3 倍性能提升
- **研究级**：完整的可复现性、评估流程和科学严谨性
- **多语言支持**：Python、Rust、R、Metal 着色器等

### OpenEvolve vs 手动优化

| 方面 | 手动优化 | OpenEvolve |
|------|---------|------------|
| **学习时间** | 需要数周理解领域 | 几分钟即可开始 |
| **解决方案质量** | 取决于专家水平 | 持续探索新方法 |
| **时间投入** | 每个优化需要数天到数周 | 数小时完成完整进化 |
| **可复现性** | 难以复现 | 完全确定性复现 |
| **扩展性** | 受限于人力 | 跨岛屿并行进化 |

---

## 🚀 快速开始

### 安装

```bash
# 使用 pip 安装
pip install openevolve
```

### 环境配置

首先需要配置 LLM API 密钥。OpenEvolve 支持任何 OpenAI 兼容的 API：

```bash
# 使用 OpenAI
export OPENAI_API_KEY="sk-..."

# 或使用 Google Gemini（免费额度可用）
# 获取 API 密钥：https://aistudio.google.com/apikey
export OPENAI_API_KEY="your-gemini-api-key"
```

### 运行第一个示例

```bash
# 运行函数最小化示例
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

这个示例会从简单的随机搜索算法进化到模拟退火算法，性能提升 100 倍！

### 从 Checkpoint 恢复

OpenEvolve 会定期保存 checkpoint，允许你中断后继续演化。使用 `--checkpoint` 参数从之前的 checkpoint 恢复：

```bash
# 从指定的 checkpoint 恢复，继续运行 50 次迭代
python openevolve-run.py problems/erdos_475/initial_program.py \
  problems/erdos_475/evaluator.py \
  --config problems/erdos_475/config.yaml \
  --checkpoint problems/erdos_475/openevolve_output/checkpoints/checkpoint_335 \
  --iterations 50
```

---

## ⚙️ 配置参数

### Island 模型参数

Island 模型将程序分成多个独立的"岛屿"独立进化，定期迁移交换优秀程序，防止过早收敛。

**相关参数**

```yaml
database:
  num_islands: 6              # 岛屿数量
  migration_interval: 50     # 迁移间隔（每 N 代迁移一次）
  migration_rate: 0.1        # 迁移率（迁移每个岛屿前 10% 的优秀程序）
```

**参数调整效果**

| 参数 | 调大 | 调小 |
|------|------|------|
| `num_islands` | ✅ 更多多样性<br>❌ 收敛更慢<br>❌ 计算开销更大 | ✅ 收敛更快<br>❌ 可能过早收敛<br>❌ 多样性降低 |
| `migration_interval` | ✅ 保持岛屿独立性<br>✅ 多样性更高<br>❌ 知识共享慢 | ✅ 知识共享快<br>❌ 岛屿可能很快变得相似<br>❌ 多样性降低 |
| `migration_rate` | ✅ 好解传播快<br>❌ 多样性降低<br>❌ 岛屿可能同质化 | ✅ 保持多样性<br>❌ 知识共享慢<br>❌ 好解传播慢 |

**推荐值**

- **简单问题**：`num_islands: 3-5`，`migration_interval: 20-30`，`migration_rate: 0.15`
- **中等问题**：`num_islands: 5-7`，`migration_interval: 50`，`migration_rate: 0.1`
- **复杂问题**：`num_islands: 7-10`，`migration_interval: 50-100`，`migration_rate: 0.05-0.1`
