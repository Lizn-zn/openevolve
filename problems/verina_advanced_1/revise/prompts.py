"""
Revise 模块的 Prompt 模板

提供两种策略:
1. Fix: 尝试真正修复语法错误，不使用 sorry
2. Eliminate: 使用最小化 sorry 填充来消除编译错误
"""

# =============================================================================
# Fix 策略 - 尝试真正修复语法错误
# =============================================================================

FIX_SYSTEM_PROMPT = """You are a Lean 4 expert. Your job is to FIX syntax and type errors WITHOUT using `sorry`.

Rules:
1. ONLY modify code between `-- EVOLVE-BLOCK-START` and `-- EVOLVE-BLOCK-END`
2. Actually FIX the errors - do NOT use `sorry`
3. Analyze the error messages carefully to understand what went wrong
4. Keep all working code intact

Output SEARCH/REPLACE blocks:
<<<<<<< SEARCH
exact code to find
=======
replacement
>>>>>>> REPLACE"""


FIX_USER_PROMPT_TEMPLATE = """Fix the compilation errors shown below. Do NOT use `sorry`.

CRITICAL RULES:
- Actually FIX the errors by correcting the code logic
- Do NOT use `sorry` - find the real solution
- Analyze the error messages to understand what went wrong

COMMON FIXES:
- "type mismatch" → Check the types and adjust the expression accordingly
- "unknown identifier" → Check if the name is spelled correctly or needs to be imported
- "failed to synthesize" → Provide the missing instance or adjust the types
- "unsolved goals" → The tactic didn't fully solve the goal, try a different approach

{code}

Errors:

{errors}

Output ONLY SEARCH/REPLACE blocks. Fix the errors by correcting the code, NOT by adding sorry."""


# =============================================================================
# Eliminate 策略 - 使用最小化 sorry 填充
# =============================================================================

ELIMINATE_SYSTEM_PROMPT = """You are a Lean 4 syntax repair tool. Make code compile with minimal `sorry` insertions.

Rules:
1. ONLY modify code between `-- EVOLVE-BLOCK-START` and `-- EVOLVE-BLOCK-END`
2. DO NOT try to fix proofs - just make them compile
3. Look at the ERROR LINE NUMBERS and fix those specific lines
4. Keep all working code intact

Output SEARCH/REPLACE blocks:
<<<<<<< SEARCH
exact code to find
=======
replacement
>>>>>>> REPLACE"""


ELIMINATE_USER_PROMPT_TEMPLATE = """Fix ONLY the compilation errors shown below. Do NOT modify working code.

CRITICAL RULES:
- DO NOT attempt to fix or rewrite proofs! Your ONLY job is to make code compile using `sorry`.
- NEVER replace a failing tactic with a different proof attempt - just use `sorry`.

HOW TO FIX:
- "unsolved goals" error → APPEND `sorry` after the tactic (e.g., `simp [x]` → `simp [x]; sorry`)
- Other errors (failed to synthesize, type mismatch) → This is CRITICAL:
  * You MUST replace the ENTIRE proof body from the error line to the end of that lemma/theorem with just `sorry`
  * Do NOT keep any code after the `sorry` - all subsequent tactics in that proof become dead code
- Multiple goals from one tactic → Add multiple `sorry`s as needed

{code}

Errors:

{errors}

Output ONLY SEARCH/REPLACE blocks. Fix ONLY the erroring lines, keep everything else unchanged."""


