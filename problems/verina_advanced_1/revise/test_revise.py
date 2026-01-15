#!/usr/bin/env python3
"""
Revise 模块测试脚本

测试 LLM 修复和 sorry 替换功能。
"""

import sys
from pathlib import Path

# 添加路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from revise import revise_proof, llm_revise, sorry_revise, verify_lean_code, is_code_valid
from revise.config import get_config, reload_config


# 测试用例：一段有编译错误的 Lean 代码
TEST_CODE_WITH_ERROR = """import Mathlib
set_option maxHeartbeats 0

namespace verina_advanced_1

-- EVOLVE-BLOCK-START

lemma lemma1_nonempty (nums : List Int) : nums.length > 0 := by
  have hlen : nums.length = 0 ∨ nums.length > 0 := by
    exact lt_or_eq_of_le (Nat.zero_le _ ) |> Or.symm |> Or_flip ?_
  sorry

theorem FindSingleNumber_spec_satisfied (nums: List Int) : True := by
  have h := lemma1_nonempty nums
  exact True.intro

-- EVOLVE-BLOCK-END

end verina_advanced_1
"""

# 对应的错误消息
TEST_ERROR_MESSAGES = [
    "{'line': 10, 'column': 56}: unknown identifier 'Or_flip'",
    "{'line': 8, 'column': 23}: unsolved goals",
]


def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试 1: 配置加载")
    print("=" * 60)
    
    config = get_config()
    print(f"use_llm: {config.use_llm}")
    print(f"llm.model: {config.llm.model}")
    print(f"llm.api_base: {config.llm.api_base}")
    print(f"llm.use_azure_ad: {config.llm.use_azure_ad}")
    print(f"fallback_to_sorry: {config.fallback_to_sorry}")
    print(f"max_retries: {config.max_retries}")
    print("✅ 配置加载成功\n")
    return True


def test_sorry_revise():
    """测试 sorry 替换"""
    print("=" * 60)
    print("测试 2: Sorry 替换")
    print("=" * 60)
    
    revised_code, success, revised_names = sorry_revise(TEST_CODE_WITH_ERROR, TEST_ERROR_MESSAGES)
    
    print(f"成功: {success}")
    print(f"修复的定理: {revised_names}")
    
    if success:
        print("\n修复后的代码 (部分):")
        # 只显示 EVOLVE-BLOCK 部分
        lines = revised_code.split('\n')
        in_block = False
        for line in lines:
            if 'EVOLVE-BLOCK-START' in line:
                in_block = True
            if in_block:
                print(line)
            if 'EVOLVE-BLOCK-END' in line:
                break
        print("✅ Sorry 替换成功\n")
    else:
        print("❌ Sorry 替换失败\n")
    
    return success


def test_llm_revise():
    """测试 LLM 修复"""
    print("=" * 60)
    print("测试 3: LLM 智能修复")
    print("=" * 60)
    
    config = get_config()
    if not config.use_llm:
        print("⏭️ LLM 修复已禁用，跳过测试\n")
        return True
    
    print("正在调用 LLM API...")
    revised_code, success = llm_revise(TEST_CODE_WITH_ERROR, TEST_ERROR_MESSAGES)
    
    print(f"成功: {success}")
    
    if success:
        print("\n修复后的代码 (部分):")
        lines = revised_code.split('\n')
        in_block = False
        for line in lines:
            if 'EVOLVE-BLOCK-START' in line:
                in_block = True
            if in_block:
                print(line)
            if 'EVOLVE-BLOCK-END' in line:
                break
        print("✅ LLM 修复成功\n")
    else:
        print("❌ LLM 修复失败（将回退到 sorry 替换）\n")
    
    return True  # LLM 失败不算测试失败


def test_full_revise():
    """测试完整的 revise_proof 流程"""
    print("=" * 60)
    print("测试 4: 完整 revise_proof 流程")
    print("=" * 60)
    
    print("正在调用 revise_proof...")
    # 现在 revise_proof 内部使用 revise 模块自带的 verify_lean_code
    revised_code, success, info = revise_proof(
        TEST_CODE_WITH_ERROR, 
        TEST_ERROR_MESSAGES,
    )
    
    print(f"成功: {success}")
    print(f"修复方式: {info}")
    
    if success:
        # 验证修复后的代码
        print("\n验证修复后的代码...")
        result = verify_lean_code(revised_code)
        is_valid = result.get("is_valid_with_sorry", False)
        print(f"Lean 验证结果: {'✅ 编译通过' if is_valid else '❌ 编译失败'}")
        
        if is_valid:
            print("\n修复后的代码 (部分):")
            lines = revised_code.split('\n')
            in_block = False
            for line in lines:
                if 'EVOLVE-BLOCK-START' in line:
                    in_block = True
                if in_block:
                    print(line)
                if 'EVOLVE-BLOCK-END' in line:
                    break
            print("✅ 完整流程测试成功\n")
            return True
    
    print("❌ 完整流程测试失败\n")
    return False


def test_lean_server():
    """测试 Lean 服务器连接"""
    print("=" * 60)
    print("测试 0: Lean 服务器连接")
    print("=" * 60)
    
    config = get_config()
    print(f"Lean 服务器地址: {config.lean_server.url}")
    
    try:
        # 尝试发送一个简单的验证请求
        test_code = "-- test\n#check Nat"
        result = verify_lean_code(test_code)
        if "error" not in result or result.get("is_valid_with_sorry"):
            print("✅ Lean 服务器连接正常\n")
            return True
    except Exception as e:
        print(f"连接错误: {e}")
    
    print("❌ Lean 服务器未运行或无法连接")
    print(f"请确保 Lean 服务器在 {config.lean_server.url} 运行\n")
    return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Revise 模块测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 测试 0: Lean 服务器
    if not test_lean_server():
        print("⚠️ Lean 服务器未运行，跳过需要验证的测试\n")
        skip_verification = True
    else:
        skip_verification = False
    
    # 测试 1: 配置
    results.append(("配置加载", test_config()))
    
    # 测试 2: Sorry 替换
    results.append(("Sorry 替换", test_sorry_revise()))
    
    # 测试 3: LLM 修复
    results.append(("LLM 修复", test_llm_revise()))
    
    # 测试 4: 完整流程（需要 Lean 服务器）
    if not skip_verification:
        results.append(("完整流程", test_full_revise()))
    else:
        print("=" * 60)
        print("测试 4: 完整流程 (跳过 - 需要 Lean 服务器)")
        print("=" * 60 + "\n")
    
    # 汇总
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n总体结果: {'✅ 全部通过' if all_passed else '❌ 有测试失败'}\n")
    
    return 0 if all_passed else 1


def test_end_to_end():
    """
    端到端测试：真实的 verify -> revise -> verify 流程
    
    模拟真实场景：
    1. 有一段编译错误的 Lean 代码
    2. 用 Lean server 验证，获取错误信息
    3. 用 revise 模块修复
    4. 再次用 Lean server 验证
    """
    print("\n" + "=" * 60)
    print("端到端测试: Verify -> Revise -> Verify")
    print("=" * 60 + "\n")
    
    # 一段有编译错误的 Lean 代码
    broken_code = """import Mathlib
set_option maxHeartbeats 0

namespace verina_advanced_1

-- EVOLVE-BLOCK-START

-- 这个 lemma 有语法错误：Or_flip 不存在
lemma lemma1_nonempty (nums : List Int) : nums.length > 0 := by
  have hlen : nums.length = 0 ∨ nums.length > 0 := by
    exact lt_or_eq_of_le (Nat.zero_le _ ) |> Or.symm |> Or_flip ?_
  sorry

-- 这个 theorem 依赖上面的 lemma
theorem FindSingleNumber_spec_satisfied (nums: List Int) : True := by
  have h := lemma1_nonempty nums
  exact True.intro

-- EVOLVE-BLOCK-END

end verina_advanced_1
"""
    
    print("=" * 40)
    print("Step 1: 原始代码 (有编译错误)")
    print("=" * 40)
    # 只打印 EVOLVE-BLOCK 部分
    for line in broken_code.split('\n'):
        if 'EVOLVE-BLOCK' in line or ('lemma' in line.lower() or 'theorem' in line.lower() or 'sorry' in line or 'exact' in line):
            print(line)
    
    print("\n" + "=" * 40)
    print("Step 2: 第一次验证 (获取错误信息)")
    print("=" * 40)
    
    from revise.lean_verify import verify_lean_code, check_verification_result
    
    result1 = verify_lean_code(broken_code)
    is_valid_no_sorry, is_valid_with_sorry, errors = check_verification_result(result1)
    
    print(f"编译通过 (无 sorry): {is_valid_no_sorry}")
    print(f"编译通过 (有 sorry): {is_valid_with_sorry}")
    print(f"错误数量: {len(errors)}")
    
    if errors:
        print("\n错误信息:")
        for i, err in enumerate(errors[:5], 1):
            # 简化错误信息显示
            err_str = str(err)
            if len(err_str) > 100:
                err_str = err_str[:100] + "..."
            print(f"  {i}. {err_str}")
    
    if is_valid_with_sorry:
        print("\n⚠️ 代码已经可以编译，不需要修复")
        return True
    
    print("\n" + "=" * 40)
    print("Step 3: 调用 revise 模块修复")
    print("=" * 40)
    
    print("正在修复...")
    revised_code, success, info = revise_proof(broken_code, errors)
    
    print(f"修复成功: {success}")
    print(f"修复方式: {info}")
    
    if not success:
        print("❌ 修复失败")
        return False
    
    print("\n修复后的代码 (EVOLVE-BLOCK 部分):")
    in_block = False
    for line in revised_code.split('\n'):
        if 'EVOLVE-BLOCK-START' in line:
            in_block = True
        if in_block:
            print(line)
        if 'EVOLVE-BLOCK-END' in line:
            break
    
    print("\n" + "=" * 40)
    print("Step 4: 第二次验证 (检查修复结果)")
    print("=" * 40)
    
    result2 = verify_lean_code(revised_code)
    is_valid_no_sorry2, is_valid_with_sorry2, errors2 = check_verification_result(result2)
    
    print(f"编译通过 (无 sorry): {is_valid_no_sorry2}")
    print(f"编译通过 (有 sorry): {is_valid_with_sorry2}")
    print(f"错误数量: {len(errors2)}")
    
    if is_valid_with_sorry2:
        print("\n✅ 修复成功！代码现在可以编译通过")
        
        # 额外信息：检查 sorry 数量
        sorry_count = revised_code.count('sorry')
        print(f"剩余 sorry 数量: {sorry_count}")
        
        return True
    else:
        print("\n❌ 修复后仍然有编译错误:")
        for i, err in enumerate(errors2[:3], 1):
            print(f"  {i}. {str(err)[:80]}...")
        return False


def test_end_to_end_with_real_file():
    """
    端到端测试：使用真实的 checkpoint 文件
    """
    print("\n" + "=" * 60)
    print("端到端测试: 使用真实 checkpoint 文件")
    print("=" * 60 + "\n")
    
    # 尝试找一个真实的 checkpoint 文件
    import os
    checkpoint_base = Path(__file__).parent.parent / "openevolve_output" / "checkpoints"
    
    if not checkpoint_base.exists():
        print(f"⏭️ Checkpoint 目录不存在: {checkpoint_base}")
        return True
    
    # 找最新的 checkpoint
    checkpoints = sorted([d for d in checkpoint_base.iterdir() if d.is_dir()])
    if not checkpoints:
        print("⏭️ 没有找到 checkpoint")
        return True
    
    latest = checkpoints[-1]
    program_file = latest / "best_program.lean"
    
    if not program_file.exists():
        print(f"⏭️ 程序文件不存在: {program_file}")
        return True
    
    print(f"使用 checkpoint: {latest.name}")
    
    with open(program_file, 'r') as f:
        code = f.read()
    
    print(f"代码长度: {len(code)} 字符")
    print(f"代码行数: {len(code.split(chr(10)))} 行")
    
    # 验证
    from revise.lean_verify import verify_lean_code, check_verification_result
    
    print("\n验证中...")
    result = verify_lean_code(code)
    is_valid_no_sorry, is_valid_with_sorry, errors = check_verification_result(result)
    
    print(f"编译通过 (无 sorry): {is_valid_no_sorry}")
    print(f"编译通过 (有 sorry): {is_valid_with_sorry}")
    print(f"错误数量: {len(errors)}")
    
    if is_valid_with_sorry:
        print("✅ 代码编译通过，无需修复")
        return True
    
    if errors:
        print("\n尝试修复...")
        revised_code, success, info = revise_proof(code, errors)
        print(f"修复结果: {'成功' if success else '失败'}, 方式: {info}")
        
        if success:
            result2 = verify_lean_code(revised_code)
            _, is_valid2, _ = check_verification_result(result2)
            print(f"修复后编译: {'✅ 通过' if is_valid2 else '❌ 失败'}")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Revise 模块测试")
    parser.add_argument("--e2e", action="store_true", help="只运行端到端测试")
    parser.add_argument("--real", action="store_true", help="使用真实 checkpoint 测试")
    args = parser.parse_args()
    
    if args.e2e:
        success = test_end_to_end()
        sys.exit(0 if success else 1)
    elif args.real:
        success = test_end_to_end_with_real_file()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())

