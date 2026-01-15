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


if __name__ == "__main__":
    sys.exit(main())

