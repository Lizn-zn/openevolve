"""
Utilities for code parsing, diffing, and manipulation
"""

import re
from typing import Dict, List, Optional, Tuple, Union


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    # Pattern to match comment-style EVOLVE-BLOCK markers
    # Supports: # (Python/Shell), -- (Lean/Haskell/SQL), // (C/Java/JS), /* (C-style block)
    start_pattern = re.compile(r'^\s*(#|--|//|/\*)\s*EVOLVE-BLOCK-START')
    end_pattern = re.compile(r'^\s*(#|--|//|/\*|\*)\s*EVOLVE-BLOCK-END')

    for i, line in enumerate(lines):
        if start_pattern.match(line):
            in_block = True
            start_line = i
            block_content = []
        elif end_pattern.match(line) and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = original_code.split("\n")
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            if result_lines[i : i + len(search_lines)] == search_lines:
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(
    diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def validate_changes_within_evolve_block(
    original_code: str, 
    modified_code: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all changes are within EVOLVE-BLOCK markers.
    
    Args:
        original_code: Original source code with EVOLVE-BLOCK markers
        modified_code: Modified code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all changes are within EVOLVE-BLOCK
        - error_message: Description of violation if invalid, None otherwise
    """
    # Parse EVOLVE-BLOCK ranges from original code
    evolve_blocks = parse_evolve_blocks(original_code)
    
    if not evolve_blocks:
        # No EVOLVE-BLOCK markers found, allow all changes
        return True, None
    
    # Get line ranges that are allowed to be modified
    # Include the marker lines themselves (start_line and end_line)
    allowed_ranges = []
    for start_line, end_line, _ in evolve_blocks:
        # Allow modification from start_line (marker) to end_line (marker) inclusive
        allowed_ranges.append((start_line, end_line))
    
    def is_line_in_allowed_range(line_num: int) -> bool:
        for start, end in allowed_ranges:
            if start <= line_num <= end:
                return True
        return False
    
    # Compare original and modified code line by line
    original_lines = original_code.split("\n")
    modified_lines = modified_code.split("\n")
    
    # Find which lines were modified
    violations = []
    
    # Use difflib to find changes
    import difflib
    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        
        # For 'replace', 'delete', 'insert' operations
        if tag in ('replace', 'delete'):
            # Check if the original lines being modified are in allowed range
            for line_num in range(i1, i2):
                if not is_line_in_allowed_range(line_num):
                    line_content = original_lines[line_num] if line_num < len(original_lines) else ""
                    violations.append(
                        f"Line {line_num + 1} modified outside EVOLVE-BLOCK: {line_content[:50]}..."
                    )
        
        if tag == 'insert':
            # For insertions, check if the insertion point is within allowed range
            # Use the line before/after the insertion point
            insert_point = i1
            if not is_line_in_allowed_range(insert_point) and not is_line_in_allowed_range(insert_point - 1):
                violations.append(
                    f"Code inserted at line {insert_point + 1} outside EVOLVE-BLOCK"
                )
    
    if violations:
        error_msg = "Changes detected outside EVOLVE-BLOCK:\n" + "\n".join(violations[:5])
        if len(violations) > 5:
            error_msg += f"\n... and {len(violations) - 5} more violations"
        return False, error_msg
    
    return True, None


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    # Check Lean first (before Python, since both use 'import' and 'def')
    if re.search(r"^(import Mathlib|namespace|theorem|lemma|#check|#eval|set_option)", code, re.MULTILINE) or \
       re.search(r":=\s*by\s", code):
        return "lean"
    elif re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"
