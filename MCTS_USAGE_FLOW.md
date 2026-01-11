# MCTS和规则划分的使用流程详解

## 核心问题：规则确定区域后，如何使用这些区域？

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 程序评估阶段（Evaluator）                                      │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
    Search Program执行 → 返回 (p, elements)
                          │
                          ▼
    Evaluator评估 → 返回 metrics + artifacts
                          │
                          ▼
    artifacts = {"p": 53, "elements": [1,2,3], "score": 0.85}
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 规则分类阶段（Rule Partition）                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
    从artifacts提取: search_output = (53, [1,2,3])
                          │
                          ▼
    应用规则: rule.apply((53, [1,2,3]))
                          │
                          ▼
    规则执行: p > 50 → 53 > 50 → True
                          │
                          ▼
    标准化: region_id = (True,)
                          │
                          ▼
    添加到区域:
    - region_programs[(True,)] = [..., "program_123"]
    - region_stats[(True,)].update("program_123", fitness=0.85)
    - program.metadata["rule_region"] = (True,)
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 区域存储结构                                                   │
└─────────────────────────────────────────────────────────────────┘

RulePartition内部数据结构：

region_programs = {
    (True,):  ["prog1", "prog2", "prog3", ...],  # p > 50 的所有程序
    (False,): ["prog4", "prog5", ...]            # p <= 50 的所有程序
}

region_stats = {
    (True,):  RegionStats(
        program_count=15,
        best_fitness=0.92,
        average_fitness=0.78,
        exploration_count=8
    ),
    (False,): RegionStats(
        program_count=5,
        best_fitness=0.65,
        average_fitness=0.58,
        exploration_count=2
    )
}

                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MCTS选择阶段（下次迭代采样时）                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
    迭代开始，需要采样parent program
                          │
                          ▼
    检查: use_mcts = random() < 0.7 (70%概率)
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
     True (70%)                          False (30%)
        │                                   │
        ▼                                   ▼
┌──────────────────┐              ┌──────────────────┐
│ MCTS采样路径      │              │ 原有采样路径      │
└──────────────────┘              └──────────────────┘
        │                                   │
        ▼                                   ▼
mcts_explorer.select_region()      _sample_parent()
        │                          (MAP-Elites + Island)
        ▼
┌─────────────────────────────────────────────────────────┐
│ MCTS选择算法（运行10次模拟）                              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
对于每个区域，计算UCB1值：
        
UCB1(True,) = (avg_reward) + 1.414 * sqrt(ln(root_visits) / visits)
            = (0.78) + 1.414 * sqrt(ln(20) / 8)
            = 0.78 + 1.414 * 0.52
            = 1.52

UCB1(False,) = (0.58) + 1.414 * sqrt(ln(20) / 2)
             = 0.58 + 1.414 * 1.73
             = 3.03  ← 更高！因为访问次数少，探索价值高
        │
        ▼
选择UCB1值最高的区域: (False,) ← 被选中
        │
        ▼
返回 region_id = (False,)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 从选中区域采样程序                                      │
└─────────────────────────────────────────────────────────┘
        │
        ▼
sample_from_region((False,))
        │
        ▼
获取区域程序列表: region_programs[(False,)] = ["prog4", "prog5", ...]
        │
        ▼
随机选择一个: parent_id = "prog4"
        │
        ▼
返回 parent_program = programs["prog4"]
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ 6. 使用parent生成child                                    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
LLM基于parent生成新的child program
        │
        ▼
评估child → 分类到区域 → 更新MCTS
        │
        ▼
循环回到步骤1
```

## 详细步骤说明

### 步骤1: 程序分类（添加程序时）

**位置**: `database.py` 的 `add()` 方法 → `_classify_and_add_to_region()`

**过程**:
```python
# 1. 获取artifacts
artifacts = {"p": 53, "elements": [1, 2, 3]}

# 2. 提取search output
search_output = (artifacts["p"], artifacts["elements"])  # (53, [1,2,3])

# 3. 应用规则
region_id = rule_partition.classify(search_output)
# 内部: rule.apply((53, [1,2,3])) → p > 50 → True → (True,)

# 4. 添加到区域
rule_partition.add_program((True,), "program_123", fitness=0.85)
# 结果:
#   - region_programs[(True,)].append("program_123")
#   - region_stats[(True,)].update("program_123", 0.85)
#   - program.metadata["rule_region"] = (True,)
```

**结果**: 程序被存储到对应区域的列表中

### 步骤2: MCTS选择区域（采样时）

**位置**: `mcts_explorer.py` 的 `select_region()` 方法

**过程**:
```python
# 1. 获取所有已探索的区域
all_regions = [(True,), (False,)]

# 2. 为每个区域创建/获取MCTS节点
node_True = MCTSNode(region_id=(True,))
node_False = MCTSNode(region_id=(False,))

# 3. 运行10次MCTS模拟
for _ in range(10):
    # 3.1 选择节点（使用UCB1）
    selected = _select(root)  # 选择UCB1值最高的子节点
    
    # 3.2 计算奖励（从区域统计获取）
    reward = _compute_reward(partition, selected.region_id)
    # 对于(True,): reward = region_stats[(True,)].best_fitness = 0.92
    # 对于(False,): reward = region_stats[(False,)].best_fitness = 0.65
    
    # 3.3 反向传播更新
    _backpropagate(selected, reward)
    # 更新节点: visits++, total_reward += reward

# 4. 选择最佳子节点（基于UCB1）
best_region = root.get_best_child(exploration_constant)
# UCB1计算:
#   (True,):  (0.92) + 1.414 * sqrt(ln(10)/5) = 1.23
#   (False,): (0.65) + 1.414 * sqrt(ln(10)/2) = 2.15  ← 更高！
# 返回: (False,)
```

**结果**: MCTS选择了一个区域，比如 `(False,)`

### 步骤3: 从区域采样程序

**位置**: `database.py` 的 `sample_from_region()` 方法

**过程**:
```python
# 1. 获取选中区域的程序列表
region_id = (False,)
region_programs = rule_partition.get_region_programs((False,))
# 返回: ["prog4", "prog5", "prog6", ...]

# 2. 过滤有效程序（确保程序还在数据库中）
valid_programs = [pid for pid in region_programs if pid in self.programs]

# 3. 随机选择一个
parent_id = random.choice(valid_programs)  # 例如: "prog4"

# 4. 返回程序
return self.programs["prog4"]
```

**结果**: 从选中区域中随机选择了一个parent program

### 步骤4: 使用parent生成child

**位置**: `iteration.py` 的 `run_iteration_with_shared_db()`

**过程**:
```python
# 1. 使用parent生成child
parent = sample_from_region((False,))  # 从(False,)区域采样
child_code = llm.generate(parent.code)

# 2. 评估child
child_metrics = evaluator.evaluate(child_code)
# evaluator内部执行:
#   - 运行child的run_search() → 返回 (p, elements)
#   - 评估结果 → 返回metrics + artifacts

# 3. 分类child（重复步骤1）
artifacts = {"p": 31, "elements": [1,2,3]}  # 假设p=31
region_id = rule_partition.classify((31, [1,2,3]))  # → (False,)
rule_partition.add_program((False,), "child_123", fitness)

# 4. 更新MCTS
mcts_explorer.update((False,), reward=fitness)
# 更新(False,)区域的MCTS节点统计
```

**结果**: child被分类并添加到区域，MCTS统计被更新

## 关键数据结构

### RulePartition 存储

```python
# 区域程序映射
region_programs = {
    (True,):  ["prog1", "prog2", "prog3"],  # 所有p>50的程序ID
    (False,): ["prog4", "prog5"]            # 所有p<=50的程序ID
}

# 区域统计
region_stats = {
    (True,): RegionStats(
        program_count=3,
        best_fitness=0.92,
        best_program_id="prog1",
        average_fitness=0.85,
        exploration_count=5
    ),
    (False,): RegionStats(...)
}

# 程序到区域的映射
program_to_region = {
    "prog1": (True,),
    "prog2": (True,),
    "prog3": (True,),
    "prog4": (False,),
    "prog5": (False,)
}
```

### MCTS树结构

```
Root Node (visits=20, total_reward=15.0)
│
├─ Node (True,)  (visits=12, total_reward=11.0, avg=0.92)
│  └─ 代表区域(True,)，包含12个程序的统计
│
└─ Node (False,) (visits=8, total_reward=4.0, avg=0.50)
   └─ 代表区域(False,)，包含8个程序的统计
```

## 实际运行示例

### 迭代1: 第一个程序

```
1. 初始程序被评估
   - run_search() → (53, [1,2,3])
   - artifacts = {"p": 53, "elements": [1,2,3]}
   
2. 分类
   - rule.apply((53, [1,2,3])) → True
   - region_id = (True,)
   - 添加到region_programs[(True,)] = ["initial_prog"]
   - region_stats[(True,)] = {best_fitness: 0.75, count: 1}

3. MCTS树
   - Root: visits=0
   - (True,): visits=0 (未访问)
```

### 迭代2: 采样（使用MCTS）

```
1. MCTS选择区域
   - 两个区域都未访问，UCB1都是inf
   - 随机选择或按顺序选择，假设选择(True,)
   
2. 从区域采样
   - region_programs[(True,)] = ["initial_prog"]
   - 返回 initial_prog 作为parent
   
3. 生成child
   - LLM基于initial_prog生成新代码
   - 评估 → (59, [2,3,4]) → region_id = (True,)
   - 添加到(True,)区域
   
4. 更新MCTS
   - (True,)节点: visits=1, total_reward=0.80
```

### 迭代3-10: 学习过程

```
随着程序不断添加：
- (True,)区域: 15个程序，best_fitness=0.92
- (False,)区域: 3个程序，best_fitness=0.65

MCTS计算UCB1:
- (True,):  0.92 + 1.414*sqrt(ln(18)/15) = 1.15  (利用)
- (False,): 0.65 + 1.414*sqrt(ln(18)/3) = 2.10  (探索) ← 更高！

MCTS选择(False,)区域进行探索
→ 从(False,)采样parent
→ 生成child（可能也是p<=50）
→ 如果child的fitness更高，会更新(False,)的统计
```

### 迭代100+: 稳定阶段

```
MCTS已经学习到：
- (True,)区域通常有更好的程序（best_fitness=0.95）
- (False,)区域程序较差（best_fitness=0.70）

但UCB1仍然会：
- 大部分时间选择(True,)（利用高fitness）
- 偶尔选择(False,)（探索，防止遗漏好解）
```

## 关键理解点

### 1. 区域是"程序集合"

每个区域维护一个程序ID列表：
- `region_programs[(True,)]` = 所有p>50的程序ID列表
- `region_programs[(False,)]` = 所有p<=50的程序ID列表

### 2. MCTS选择"区域"，不是"程序"

```
MCTS的作用：
  输入: 所有区域的统计信息
  输出: 选择一个区域（如(True,)或(False,)）
  
然后从选中区域的程序列表中随机采样一个程序
```

### 3. 区域统计驱动MCTS选择

MCTS根据区域的统计信息（best_fitness, average_fitness）来决定选择哪个区域：
- 高fitness区域 → 更频繁被选择（利用）
- 低访问次数区域 → 偶尔被选择（探索）

### 4. 采样是"从区域中随机选"

MCTS选择区域后，从该区域的程序列表中**随机**选择一个作为parent：
- 不是选择区域中最好的程序
- 不是选择区域中最差的程序
- 是**随机选择**，保持多样性

## 为什么这样设计？

### 优势

1. **智能探索**: MCTS会优先探索有潜力的区域
2. **保持多样性**: 从区域中随机采样，避免过早收敛
3. **自适应**: 随着进化过程，MCTS会学习哪些区域更有价值
4. **平衡**: UCB1确保不会完全忽略低fitness区域

### 与MAP-Elites的区别

- **MAP-Elites**: 基于程序的特征（complexity, diversity）划分
- **规则划分**: 基于程序的**输出**（search output）划分
- **MCTS**: 智能选择要探索的区域

两者可以结合使用（通过`use_with_map_elites`配置）。

## 总结

**规则确定区域后的使用流程**：

1. **存储**: 程序被分类并存储到对应区域的列表中
2. **选择**: MCTS根据区域统计选择要探索的区域
3. **采样**: 从选中区域的程序列表中随机采样parent
4. **生成**: 使用parent生成child
5. **更新**: child被分类并更新区域统计和MCTS树
6. **循环**: 重复步骤2-5

核心思想：**MCTS选择"在哪里探索"（哪个区域），然后从该区域中随机采样程序进行进化**。

