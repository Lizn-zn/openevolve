# MCTS和规则划分快速开始指南

## 概述

MCTS（蒙特卡洛树搜索）和规则划分功能已完全集成到OpenEvolve框架中。本指南将帮助你快速启用和使用这些功能。

## 快速启用

### 1. 创建规则程序文件

在问题目录下创建 `rule.py` 文件，例如 `problems/erdos_475/rule.py`:

```python
def apply(search_output):
    """
    应用规则到search program的输出
    
    Args:
        search_output: Search program的输出（如 (p, elements)）
        
    Returns:
        bool: 单规则
        或
        tuple[bool, ...]: 多规则
    """
    p, elements = search_output
    
    # 单规则示例：按p的大小分类
    return p > 50
    
    # 多规则示例（取消注释使用）：
    # return (p > 50, len(elements) > 10, p % 2 == 1)
```

### 2. 更新配置文件

在 `config.yaml` 中添加以下配置：

```yaml
# 规则划分配置
rule_partition:
  enabled: true
  rule_program_path: "problems/erdos_475/rule.py"
  use_with_map_elites: false

# MCTS配置
mcts:
  enabled: true
  exploration_constant: 1.414  # sqrt(2)
  simulations_per_iteration: 10
  reward_function: "best_fitness"  # 选项: "best_fitness", "average_fitness", "max_improvement"
  use_rule_partition: true
  sampling_ratio: 0.7  # 70%使用MCTS采样，30%使用原有策略
```

### 3. 运行进化

```bash
python openevolve-run.py \
  problems/erdos_475/initial_program.py \
  problems/erdos_475/evaluator.py \
  --config problems/erdos_475/config.yaml
```

## 配置参数说明

### rule_partition

- `enabled`: 是否启用规则划分（默认: false）
- `rule_program_path`: 规则程序文件路径
- `use_with_map_elites`: 是否与MAP-Elites并行使用（默认: false）

### mcts

- `enabled`: 是否启用MCTS（默认: false）
- `exploration_constant`: UCB1探索常数（默认: 1.414，即√2）
- `simulations_per_iteration`: 每次迭代的MCTS模拟次数（默认: 10）
- `reward_function`: 奖励函数类型
  - `"best_fitness"`: 使用区域最佳fitness
  - `"average_fitness"`: 使用区域平均fitness
  - `"max_improvement"`: 使用改进值（当前简化为best_fitness）
- `use_rule_partition`: 必须与rule_partition.enabled=true配合使用（默认: true）
- `sampling_ratio`: MCTS采样比例，0.0-1.0（默认: 0.7）

## 工作原理

### 数据流

1. **程序评估**: Evaluator执行search program，生成artifacts（包含search output）
2. **规则分类**: 从artifacts提取search output，应用规则进行分类
3. **区域管理**: 程序被添加到对应区域，更新区域统计
4. **MCTS选择**: 使用MCTS算法选择要探索的区域
5. **程序采样**: 从选中区域采样parent program
6. **MCTS更新**: 根据新程序的fitness更新MCTS树

### MCTS算法

MCTS使用UCB1公式平衡探索与利用：

```
UCB1 = (average_reward) + C * sqrt(ln(parent_visits) / visits)
```

- **探索项**: 鼓励访问未充分探索的区域
- **利用项**: 偏向高奖励的区域
- **C值**: 控制探索/利用平衡（默认√2）

## 示例：多规则划分

```python
# rule.py
def apply(search_output):
    p, elements = search_output
    return (
        p > 50,              # 规则1: p是否大于50
        len(elements) > 10,  # 规则2: 元素数量是否大于10
        p % 2 == 1          # 规则3: p是否为奇数
    )
```

这将创建 2³ = 8 个区域：
- `(True, True, True)`: p>50, |elements|>10, p为奇数
- `(True, True, False)`: p>50, |elements|>10, p为偶数
- ... 等等

## 注意事项

1. **规则设计**: 规则应该有意义，能够有效划分搜索空间
2. **区域数量**: 多规则会产生指数级区域（2^n），注意控制规则数量
3. **Artifacts要求**: Search program的输出必须通过artifacts传递（evaluator需要返回artifacts）
4. **性能开销**: MCTS和规则执行会增加计算开销，需要权衡

## 故障排除

### 问题：程序没有被分类

**原因**: Artifacts中缺少search output

**解决**: 确保evaluator返回的artifacts包含search output：
- 对于erdos_475问题：artifacts应包含 `"p"` 和 `"elements"`
- 或包含 `"search_output"` 键

### 问题：MCTS没有选择区域

**原因**: 没有程序被分类到区域，或MCTS配置不正确

**解决**: 
1. 检查规则程序是否正确加载
2. 检查artifacts是否正确传递
3. 确保 `mcts.use_rule_partition: true`

### 问题：采样总是使用原有策略

**原因**: `sampling_ratio` 设置过低，或MCTS未启用

**解决**: 
1. 检查 `mcts.enabled: true`
2. 增加 `sampling_ratio` 值（如0.8或0.9）

## 高级用法

### 动态调整sampling_ratio

可以在运行过程中根据性能动态调整sampling_ratio，但需要修改代码。

### 自定义奖励函数

修改 `mcts_explorer.py` 中的 `_compute_reward` 方法，添加自定义奖励计算逻辑。

### 与MAP-Elites结合

设置 `use_with_map_elites: true` 可以同时使用规则划分和MAP-Elites（需要进一步开发）。

## 参考

- 详细文档: `MCTS_EXPANSION.md`
- 测试用例: `tests/test_mcts_rule_partition.py`
- 示例规则: `problems/erdos_475/rule.py`

