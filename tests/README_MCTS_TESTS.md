# MCTS和规则划分测试说明

## 测试文件

- `test_mcts_rule_partition.py`: 完整的测试套件，包含所有模块的单元测试和集成测试

## 运行测试

### 方法1: 使用pytest（推荐）

```bash
pytest tests/test_mcts_rule_partition.py -v
```

### 方法2: 使用unittest

```bash
python -m unittest tests.test_mcts_rule_partition -v
```

### 方法3: 使用测试脚本

```bash
./run_mcts_test.sh
```

## 测试覆盖

### 1. TestRuleProgram
- ✅ 单规则测试（返回bool）
- ✅ 多规则测试（返回tuple）
- ✅ 错误处理和默认值

### 2. TestRulePartition
- ✅ 分类功能
- ✅ 程序添加到区域
- ✅ 区域统计信息
- ✅ 多规则区域管理

### 3. TestMCTSExplorer
- ✅ MCTS节点创建和更新
- ✅ UCB1值计算
- ✅ 区域选择
- ✅ MCTS更新机制

### 4. TestIntegration
- ✅ 与ProgramDatabase的集成
- ✅ 自动分类功能
- ✅ 从区域采样
- ✅ MCTS采样

## 示例规则文件

`problems/erdos_475/rule.py` 是一个示例规则program文件，展示了如何编写规则：

```python
def apply(search_output):
    p, elements = search_output
    return p > 50  # 单规则
    # 或
    # return (p > 50, len(elements) > 10)  # 多规则
```

## 注意事项

1. 测试使用临时文件，测试完成后会自动清理
2. 某些测试需要artifacts数据，测试中会模拟这些数据
3. 集成测试需要完整的配置环境

## 故障排除

如果测试失败：

1. 检查依赖是否安装：`openevolve.rule_program`, `openevolve.rule_partition`, `openevolve.mcts_explorer`
2. 确保Python路径正确
3. 检查临时目录权限

