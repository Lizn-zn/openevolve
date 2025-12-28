# MCTS扩展：基于规则的空间划分与探索

## 概述

本文档描述在OpenEvolve框架中实现基于规则的空间划分（Rule-Based Partitioning）和蒙特卡洛树搜索（MCTS）探索的扩展功能。该功能允许用户定义自定义规则来划分搜索空间，并使用MCTS算法智能地探索不同区域。

## 核心概念

### 1. Search Program（搜索程序）
- **定义**: 被进化的程序，包含`run_search()`函数
- **输出**: 搜索结果，例如在Erdos 475问题中返回`(p, elements)`
- **位置**: 如`problems/erdos_475/initial_program.py`

### 2. Rule Program（规则程序）
- **定义**: 用户定义的程序，用于对search program的输出进行分类
- **输入**: Search program的输出（如`(p, elements)`）
- **输出**: `True/False`或`tuple[bool, ...]`（支持多规则）
- **作用**: 将搜索空间划分为不同的区域

### 3. 空间划分（Space Partitioning）
- **单规则**: 将空间划分为两个区域（`True`区域和`False`区域）
- **多规则**: 通过规则组合形成多个区域（如`(True, True)`, `(True, False)`, `(False, True)`, `(False, False)`）
- **每个区域维护**:
  - 该区域的程序列表
  - 该区域的最佳fitness score
  - 该区域的探索次数和统计信息

### 4. MCTS（蒙特卡洛树搜索）
- **节点**: 代表一个由规则划分的区域
- **动作**: 选择要探索的区域
- **奖励**: 基于该区域中程序的fitness score
- **策略**: 使用UCB1等算法平衡探索与利用

## 架构设计

### 模块结构

```
openevolve/
├── rule_program.py          # 规则program加载和执行
├── rule_partition.py         # 基于规则的空间划分
├── mcts_explorer.py          # MCTS探索器
├── database.py               # 修改：添加规则划分支持
├── iteration.py              # 修改：集成MCTS选择
└── config.py                 # 修改：添加规则和MCTS配置
```

### 核心模块说明

#### 1. RuleProgram (`openevolve/rule_program.py`)

**职责**: 加载和执行规则program

**主要接口**:
```python
class RuleProgram:
    def __init__(self, rule_program_path: str):
        """加载规则program"""
        
    def apply(self, search_output: Any) -> Union[bool, Tuple[bool, ...]]:
        """
        应用规则到search program的输出
        
        Args:
            search_output: Search program的输出
            
        Returns:
            bool或tuple[bool, ...]: 规则分类结果
        """
```

**实现要点**:
- 使用`importlib`动态加载规则program文件
- 规则program必须实现`apply(search_output)`函数
- 支持单规则（返回`bool`）和多规则（返回`tuple[bool, ...]`）
- 错误处理：规则执行失败时返回默认分类

#### 2. RulePartition (`openevolve/rule_partition.py`)

**职责**: 基于规则划分空间并管理各区域

**主要接口**:
```python
class RulePartition:
    def __init__(self, rule_program: RuleProgram):
        """初始化空间划分"""
        
    def classify(self, search_output: Any) -> Tuple[bool, ...]:
        """对search output进行分类，返回区域标识"""
        
    def add_program(self, region_id: Tuple[bool, ...], program_id: str, fitness: float):
        """将程序添加到指定区域"""
        
    def get_region_programs(self, region_id: Tuple[bool, ...]) -> List[str]:
        """获取指定区域的程序列表"""
        
    def get_region_stats(self, region_id: Tuple[bool, ...]) -> Dict:
        """获取指定区域的统计信息"""
        
    def get_all_regions(self) -> List[Tuple[bool, ...]]:
        """获取所有已探索的区域"""
```

**实现要点**:
- 使用字典存储各区域的程序：`{region_id: [program_ids...]}`
- 维护每个区域的统计信息：最佳fitness、程序数量、探索次数等
- 支持动态区域创建（首次遇到新区域时自动创建）
- 区域标识使用tuple以便支持多规则

#### 3. MCTSExplorer (`openevolve/mcts_explorer.py`)

**职责**: 实现MCTS算法来选择要探索的区域

**主要接口**:
```python
class MCTSNode:
    """MCTS树节点，代表一个区域"""
    region_id: Tuple[bool, ...]
    visits: int
    total_reward: float
    children: List['MCTSNode']
    parent: Optional['MCTSNode']

class MCTSExplorer:
    def __init__(self, exploration_constant: float = 1.414):
        """初始化MCTS探索器"""
        
    def select_region(self, partition: RulePartition) -> Tuple[bool, ...]:
        """
        使用MCTS选择要探索的区域
        
        Args:
            partition: 规则划分对象
            
        Returns:
            要探索的区域标识
        """
        
    def update(self, region_id: Tuple[bool, ...], reward: float):
        """
        更新MCTS树，记录探索结果
        
        Args:
            region_id: 探索的区域
            reward: 获得的奖励（基于fitness score）
        """
```

**MCTS算法流程**:
1. **Selection（选择）**: 从根节点开始，使用UCB1公式选择子节点
   ```
   UCB1(node) = (node.total_reward / node.visits) + 
                C * sqrt(ln(parent.visits) / node.visits)
   ```
2. **Expansion（扩展）**: 如果节点未被完全探索，添加新的子节点（新区域）
3. **Simulation（模拟）**: 在选中区域进行rollout，采样程序并评估
4. **Backpropagation（回传）**: 将模拟结果反向传播，更新路径上所有节点的统计

**实现要点**:
- 使用UCB1平衡探索与利用
- 支持自定义exploration constant（默认√2）
- 维护MCTS树结构，支持多轮探索
- 奖励计算：基于区域的fitness score（如最佳fitness或平均fitness）

#### 4. 数据库扩展 (`openevolve/database.py`)

**修改点**:
- 在`ProgramDatabase`类中添加：
  - `rule_partition: Optional[RulePartition]`属性
  - `mcts_explorer: Optional[MCTSExplorer]`属性
- 修改`add()`方法：
  - 如果启用规则划分，执行search program获取输出
  - 应用规则进行分类
  - 将程序添加到对应区域
  - 更新MCTS统计
- 添加新方法：
  - `sample_from_region(region_id)`: 从指定区域采样程序
  - `get_region_for_program(program_id)`: 获取程序所属区域

#### 5. 迭代逻辑扩展 (`openevolve/iteration.py`)

**修改点**:
- 修改`run_iteration_with_shared_db()`:
  - 如果启用MCTS+规则划分：
    1. MCTS选择要探索的区域
    2. 从选中区域采样parent program
    3. 生成child program
    4. 评估child program
    5. 应用规则分类child program
    6. 更新MCTS统计
  - 否则使用原有采样策略

## 配置扩展

### Config类修改 (`openevolve/config.py`)

添加新的配置类：

```python
@dataclass
class RulePartitionConfig:
    """规则划分配置"""
    enabled: bool = False
    rule_program_path: Optional[str] = None
    use_with_map_elites: bool = False  # 是否与MAP-Elites结合使用

@dataclass
class MCTSConfig:
    """MCTS配置"""
    enabled: bool = False
    exploration_constant: float = 1.414  # sqrt(2)
    simulations_per_iteration: int = 10
    reward_function: str = "best_fitness"  # "best_fitness" | "average_fitness" | "max_improvement"
    use_rule_partition: bool = True  # 是否使用规则划分
    sampling_ratio: float = 0.7  # MCTS采样比例（剩余使用原有策略）
```

在`Config`类中添加：
```python
rule_partition: RulePartitionConfig = field(default_factory=RulePartitionConfig)
mcts: MCTSConfig = field(default_factory=MCTSConfig)
```

### 配置文件示例 (`config.yaml`)

```yaml
# 规则划分配置
rule_partition:
  enabled: true
  rule_program_path: "problems/erdos_475/rule.py"
  use_with_map_elites: false  # 如果为true，规则划分与MAP-Elites并行使用

# MCTS配置
mcts:
  enabled: true
  exploration_constant: 1.414  # UCB1探索常数
  simulations_per_iteration: 10  # 每次迭代的模拟次数
  reward_function: "best_fitness"  # 奖励函数类型
  use_rule_partition: true  # 必须与rule_partition.enabled=true配合使用
  sampling_ratio: 0.7  # 70%使用MCTS采样，30%使用原有策略
```

## 规则Program编写指南

### 规则Program接口

规则program必须实现`apply()`函数：

```python
def apply(search_output):
    """
    应用规则到search program的输出
    
    Args:
        search_output: Search program的输出（类型取决于具体问题）
        
    Returns:
        bool: 单规则情况
        或
        tuple[bool, ...]: 多规则情况，每个bool对应一个规则
    """
    # 实现规则逻辑
    pass
```

### 示例1: 单规则（Erdos 475问题）

```python
# problems/erdos_475/rule.py
def apply(search_output):
    """
    规则: p是否大于50
    """
    p, elements = search_output
    return p > 50
```

这将空间划分为两个区域：
- `(True,)`: p > 50的区域
- `(False,)`: p <= 50的区域

### 示例2: 多规则（Erdos 475问题）

```python
# problems/erdos_475/rule.py
def apply(search_output):
    """
    规则1: p是否大于50
    规则2: elements数量是否大于10
    """
    p, elements = search_output
    rule1 = p > 50
    rule2 = len(elements) > 10
    return (rule1, rule2)
```

这将空间划分为四个区域：
- `(True, True)`: p > 50 且 |elements| > 10
- `(True, False)`: p > 50 且 |elements| <= 10
- `(False, True)`: p <= 50 且 |elements| > 10
- `(False, False)`: p <= 50 且 |elements| <= 10

### 示例3: 复杂规则（函数最小化问题）

```python
# problems/function_minimization/rule.py
def apply(search_output):
    """
    规则1: x是否在[-2, 2]范围内
    规则2: y是否在[-2, 2]范围内
    规则3: 函数值是否小于-1.0
    """
    x, y, value = search_output
    rule1 = -2 <= x <= 2
    rule2 = -2 <= y <= 2
    rule3 = value < -1.0
    return (rule1, rule2, rule3)
```

这将空间划分为8个区域（2³ = 8）。

## 数据流

### 完整迭代流程

```
1. MCTS选择区域
   └─> MCTSExplorer.select_region(partition)
       └─> 返回 region_id

2. 从选中区域采样parent
   └─> database.sample_from_region(region_id)
       └─> 返回 parent_program

3. LLM生成child program
   └─> llm_ensemble.generate_with_context(...)
       └─> 返回 child_code

4. 评估child program
   └─> evaluator.evaluate_program(child_code)
       └─> 返回 metrics

5. 执行search program获取输出
   └─> 执行child_code中的run_search()
       └─> 返回 search_output

6. 应用规则分类
   └─> rule_partition.classify(search_output)
       └─> 返回 child_region_id

7. 将child添加到对应区域
   └─> rule_partition.add_program(child_region_id, child_id, fitness)

8. 更新MCTS统计
   └─> mcts_explorer.update(child_region_id, reward)
       └─> 更新MCTS树节点统计

9. 将child添加到database
   └─> database.add(child_program)
```

### 关键决策点

1. **规则执行时机**:
   - 选项A: 在evaluator中执行（评估时分类）
   - 选项B: 在iteration中执行（生成后立即分类）
   - **推荐**: 选项B，因为需要search program的输出

2. **MCTS与现有采样策略的关系**:
   - 选项A: MCTS完全控制parent选择
   - 选项B: MCTS与现有策略混合（通过`sampling_ratio`配置）
   - **推荐**: 选项B，提供灵活性

3. **规则划分与MAP-Elites的关系**:
   - 选项A: 完全替代MAP-Elites
   - 选项B: 与MAP-Elites并行使用
   - 选项C: 规则作为额外的feature dimension
   - **推荐**: 选项A（初始实现），后续可扩展支持选项B/C

## 实现步骤

### Phase 1: 基础框架
1. ✅ 创建`RuleProgram`类，实现规则加载和执行
2. ✅ 创建`RulePartition`类，实现空间划分和管理
3. ✅ 扩展`Config`类，添加规则和MCTS配置

### Phase 2: MCTS实现
4. ✅ 创建`MCTSNode`和`MCTSExplorer`类
5. ✅ 实现UCB1选择算法
6. ✅ 实现MCTS的四个阶段（Selection, Expansion, Simulation, Backpropagation）

### Phase 3: 集成
7. ✅ 修改`ProgramDatabase`，添加规则划分支持
8. ✅ 修改`iteration.py`，集成MCTS选择逻辑
9. ✅ 添加区域采样方法

### Phase 4: 测试和优化
10. ✅ 编写单元测试
11. ✅ 编写集成测试
12. ✅ 性能优化和调试

## 使用示例

### 基本使用

1. **创建规则program** (`problems/erdos_475/rule.py`):
```python
def apply(search_output):
    p, elements = search_output
    return p > 50
```

2. **配置** (`problems/erdos_475/config.yaml`):
```yaml
rule_partition:
  enabled: true
  rule_program_path: "problems/erdos_475/rule.py"

mcts:
  enabled: true
  exploration_constant: 1.414
  simulations_per_iteration: 10
  use_rule_partition: true
  sampling_ratio: 0.7
```

3. **运行**:
```bash
python openevolve-run.py \
  problems/erdos_475/initial_program.py \
  problems/erdos_475/evaluator.py \
  --config problems/erdos_475/config.yaml
```

### 高级使用：多规则

```python
# rule.py
def apply(search_output):
    p, elements = search_output
    return (
        p > 50,              # 规则1
        len(elements) > 10,  # 规则2
        p % 2 == 1          # 规则3
    )
```

这将创建2³ = 8个区域。

## 优势与适用场景

### 优势

1. **灵活性**: 规则可自定义，适应不同问题的特性
2. **智能探索**: MCTS平衡探索与利用，避免过早收敛
3. **可扩展性**: 可与现有MAP-Elites框架结合
4. **问题导向**: 规则可以针对问题特性设计，提高搜索效率

### 适用场景

1. **有明确分类标准的问题**: 如Erdos 475问题，可以根据p的大小、elements数量等分类
2. **多目标优化**: 不同规则可以对应不同的优化目标
3. **约束优化**: 规则可以表示约束条件，MCTS可以探索满足不同约束的区域
4. **探索未知区域**: MCTS倾向于探索未充分探索的区域，有助于发现新的解

## 注意事项

1. **规则设计**: 规则应该有意义，能够有效划分搜索空间
2. **区域数量**: 多规则会产生指数级区域（2^n），注意控制规则数量
3. **性能开销**: MCTS和规则执行会增加计算开销，需要权衡
4. **规则稳定性**: 规则应该对相似的输入产生稳定的输出

## 未来扩展

1. **动态规则**: 规则可以根据进化过程动态调整
2. **规则进化**: 使用进化算法优化规则本身
3. **混合策略**: 更灵活的MCTS与MAP-Elites结合策略
4. **可视化**: 可视化规则划分的空间和MCTS探索过程

## 参考资源

- [MCTS算法详解](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [UCB1公式](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [MAP-Elites算法](https://arxiv.org/abs/1504.04909)

