#!/usr/bin/env python3
"""
Revise 模块端到端测试

测试流程: Lean 代码 -> 验证 -> 修复 -> 再验证
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from revise import revise_proof
from revise.lean_verify import verify_lean_code, check_verification_result


# 测试用的 Lean 代码（有编译错误）
TEST_LEAN_CODE = """import Mathlib
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


def test_revise(lean_code: str) -> bool:
    """
    端到端测试：验证 -> 修复 -> 验证
    
    Args:
        lean_code: 待测试的 Lean 代码
        
    Returns:
        是否修复成功
    """
    print("=" * 50)
    print("Step 1: 验证原始代码")
    print("=" * 50)
    
    result = verify_lean_code(lean_code)
    _, is_valid, errors = check_verification_result(result)
    
    print(f"编译通过: {is_valid}")
    print(f"错误数量: {len(errors)}")
    
    if is_valid:
        print("✅ 代码已经可以编译，无需修复")
        return True
    
    if errors:
        print("\n错误信息:")
        for i, err in enumerate(errors[:3], 1):
            err_str = str(err)[:80]
            print(f"  {i}. {err_str}")
    
    print("\n" + "=" * 50)
    print("Step 2: 调用 revise 修复")
    print("=" * 50)
    
    revised_code, success, info = revise_proof(lean_code, errors)
    
    print(f"修复成功: {success}")
    print(f"修复方式: {info}")
    
    if not success:
        print("❌ 修复失败")
        return False
    
    print("\n" + "=" * 50)
    print("Step 3: 验证修复后的代码")
    print("=" * 50)
    
    result2 = verify_lean_code(revised_code)
    _, is_valid2, errors2 = check_verification_result(result2)
    
    print(f"编译通过: {is_valid2}")
    print(f"错误数量: {len(errors2)}")
    print(f"修复后的代码: {revised_code}")
    
    if is_valid2:
        print(f"剩余 sorry 数量: {revised_code.count('sorry')}")
        print("\n✅ 修复成功！")
        return True
    else:
        print("\n❌ 修复后仍有错误")
        return False


if __name__ == "__main__":
    # 可以通过命令行传入文件路径，否则使用默认测试代码
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        if file_path.exists():
            lean_code = file_path.read_text()
            print(f"使用文件: {file_path}\n")
        else:
            print(f"文件不存在: {file_path}")
            sys.exit(1)
    else:
        lean_code = TEST_LEAN_CODE
        print("使用默认测试代码\n")
    
    success = test_revise(lean_code)
    sys.exit(0 if success else 1)
