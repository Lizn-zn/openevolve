# 启用MCTS后的行为分析

## 当前状态

### 配置文件状态
- `rule_partition` 和 `mcts` 配置目前都是**注释状态**（未启用）
- `rule.py` 文件已存在：`problems/erdos_475/rule.py`
- 规则定义：`p > 50`（单规则，将空间划分为2个区域）

## 如果现在启用MCTS会发生什么？

### 1. 初始化阶段

#### 1.1 Controller初始化 (`openevolve/controller.py`)
```
✅ 检测到 rule_partition.enabled: true
   └─> 加载 RuleProgram("problems/erdos_475/rule.py")
   └─> 创建 RulePartition
   └─> 日志: "Initialized rule partition with problems/erdos_475/rule.py"

✅ 检测到 mcts.enabled: true
   └─> 检查 rule_partition 是否已初始化
   └─> 创建 MCTSExplorer(exploration_constant=1.414, reward_function="best_fitness")
   └─> 日志: "Initialized MCTS explorer (C=1.414)"

✅ 将 rule_partition 和 mcts_explorer 传递给 ProgramDatabase
```

**可能的错误情况**：
- ❌ 如果 `rule_program_path` 路径错误 → 记录错误，继续运行（不使用规则划分）
- ❌ 如果 `rule.py` 文件不存在 → 记录错误，继续运行
- ❌ 如果 `rule.py` 没有 `apply` 函数 → 抛出异常，程序可能无法启动

### 2. 迭代执行阶段

#### 2.1 采样阶段（每次迭代）

**原有行为**（MCTS未启用）：
```
database.sample()
  └─> _sample_parent()
      └─> 根据 exploration_ratio/exploitation_ratio 选择策略
      └─> 从当前island采样
```

**启用MCTS后的行为**：
```
database.sample()
  └─> 检查: use_mcts = random() < 0.7 (sampling_ratio)
  
  情况A: use_mcts = True (70%概率)
    └─> _sample_with_mcts()
        └─> mcts_explorer.select_region(partition, simulations=10)
            └─> 运行10次MCTS模拟
            └─> 使用UCB1选择最佳区域
            └─> 返回选中的区域ID（如 (True,) 或 (False,)）
        └─> sample_from_region(region_id)
            └─> 从选中区域的程序列表中随机采样
            └─> 返回parent program
    
  情况B: use_mcts = False (30%概率)
    └─> _sample_parent()
        └─> 使用原有策略（MAP-Elites + Island模型）
```

#### 2.2 程序评估阶段

**原有行为**：
```
evaluator.evaluate_program(child_code)
  └─> 执行程序，获取metrics
  └─> 返回评估结果
```

**启用MCTS后**（无变化，但artifacts很重要）：
```
evaluator.evaluate_program(child_code)
  └─> 执行程序，获取metrics
  └─> 执行 run_search() 获取 (p, elements)
  └─> 返回评估结果 + artifacts {"p": p, "elements": elements}
```

#### 2.3 程序分类阶段（新增）

**在 iteration.py 中**：
```
artifacts = evaluator.get_pending_artifacts(child_id)
  └─> 提取 search_output = (artifacts["p"], artifacts["elements"])

rule_partition.classify(search_output)
  └─> rule_program.apply((p, elements))
      └─> 执行规则: return p > 50
      └─> 返回: True 或 False
  └─> 标准化: (True,) 或 (False,)
  └─> 返回 region_id

rule_partition.add_program(region_id, child_id, fitness)
  └─> 将程序添加到对应区域
  └─> 更新区域统计（best_fitness, average_fitness等）
  └─> 存储 region_id 到 program.metadata["rule_region"]
```

#### 2.4 MCTS更新阶段（新增）

```
mcts_explorer.update(region_id, reward=fitness)
  └─> 获取或创建该区域的MCTS节点
  └─> 更新节点统计（visits++, total_reward += fitness）
  └─> 反向传播更新路径上的所有节点
```

### 3. 空间划分结果

#### 规则: `p > 50`

**区域划分**：
- **区域 (True,)**: 包含所有 `p > 50` 的程序
  - 例如: p=53, 59, 61, 67, 71, 73, 79, 83, 89, 97...
- **区域 (False,)**: 包含所有 `p <= 50` 的程序
  - 例如: p=5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47

#### MCTS行为

**初始阶段**（两个区域都没有程序）：
- MCTS会随机选择区域（因为UCB1值都是inf）
- 随着程序被添加到区域，MCTS开始学习

**学习阶段**：
- 如果区域(True,)中的程序fitness更高：
  - MCTS会倾向于选择(True,)区域
  - UCB1公式会平衡探索(False,)和利用(True,)
- 如果区域(False,)中的程序fitness更高：
  - MCTS会倾向于选择(False,)区域

**稳定阶段**：
- MCTS树会收敛到最优策略
- 高fitness区域会被更频繁地探索
- 但exploration_constant确保仍会探索低fitness区域

### 4. 实际效果

#### 优势
1. **智能探索**: MCTS会优先探索有潜力的区域（高fitness）
2. **平衡探索**: 不会完全忽略低fitness区域
3. **自适应**: 随着进化过程，MCTS会学习哪些区域更有价值

#### 潜在问题
1. **初期不稳定**: 前几个迭代中，MCTS可能随机选择（因为数据不足）
2. **区域不平衡**: 如果某个区域一直没有好程序，可能被忽略
3. **计算开销**: 每次采样需要运行10次MCTS模拟（simulations_per_iteration=10）

### 5. 日志输出示例

启用MCTS后，你会看到类似的日志：

```
INFO: Initialized rule partition with problems/erdos_475/rule.py
INFO: Initialized MCTS explorer (C=1.414)
DEBUG: MCTS selected region (True,) (UCB1=1.234)
DEBUG: Classified program abc123 to region (True,)
DEBUG: Updated MCTS node for region (True,) with reward 0.8523
```

### 6. 性能影响

#### 额外开销
- **每次采样**: ~10次MCTS模拟（如果使用MCTS）
- **每次添加程序**: 规则分类 + 区域更新（~1ms）
- **内存**: 每个区域维护程序列表和统计（可忽略）

#### 预期改进
- **探索效率**: 可能提高10-30%（取决于问题特性）
- **收敛速度**: 可能更快找到好解（如果规则设计合理）

### 7. 如何验证MCTS是否工作

#### 检查日志
```bash
grep "MCTS selected region" logs/*.log
grep "Classified program.*to region" logs/*.log
grep "Updated MCTS node" logs/*.log
```

#### 检查区域统计
可以在代码中添加调试输出：
```python
# 在迭代中打印区域统计
if database.rule_partition:
    summary = database.rule_partition.get_summary()
    print(f"Regions: {summary}")
```

### 8. 故障排除

#### 问题：MCTS没有选择区域
- **检查**: `mcts.enabled: true` 和 `rule_partition.enabled: true`
- **检查**: `sampling_ratio` 是否太低（尝试0.9）

#### 问题：程序没有被分类
- **检查**: artifacts中是否包含 `"p"` 和 `"elements"`
- **检查**: evaluator是否正确返回artifacts

#### 问题：总是选择同一个区域
- **正常现象**: 如果该区域确实有更好的程序
- **调整**: 增加 `exploration_constant`（如2.0）来增加探索

## 总结

启用MCTS后：
1. ✅ **会工作**: 所有功能已集成，代码路径完整
2. ✅ **会分类**: 程序会根据规则自动分类到区域
3. ✅ **会学习**: MCTS会学习哪些区域更有价值
4. ✅ **会平衡**: 在探索和利用之间保持平衡
5. ⚠️ **有开销**: 增加少量计算开销（可接受）
6. ⚠️ **需配置**: 确保rule.py路径正确，artifacts正确返回

**建议**: 先在小规模测试中启用，观察日志和性能，确认一切正常后再用于大规模运行。

