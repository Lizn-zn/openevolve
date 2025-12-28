"""
Rule Program module for OpenEvolve

This module provides functionality to load and execute rule programs that
classify search program outputs for space partitioning.
"""

import importlib.util
import logging
import os
import sys
from typing import Any, Tuple, Union

logger = logging.getLogger(__name__)


class RuleProgram:
    """
    Loads and executes rule programs for space partitioning.
    
    A rule program is a Python module that defines an `apply()` function
    which takes the output of a search program and returns a boolean or
    tuple of booleans for classification.
    
    Example:
        # Single rule
        def apply(search_output):
            p, elements = search_output
            return p > 50
        
        # Multiple rules
        def apply(search_output):
            p, elements = search_output
            return (p > 50, len(elements) > 10)
    """

    def __init__(self, rule_program_path: str):
        """
        Initialize RuleProgram by loading the rule program file.
        
        Args:
            rule_program_path: Path to the rule program file
            
        Raises:
            ValueError: If rule program file doesn't exist
            ImportError: If failed to load the module
            AttributeError: If module doesn't contain 'apply' function
        """
        if not os.path.exists(rule_program_path):
            raise ValueError(f"Rule program file {rule_program_path} not found")

        self.rule_program_path = os.path.abspath(rule_program_path)
        self._load_rule_function()

    def _load_rule_function(self) -> None:
        """Load the apply function from the rule program file"""
        try:
            # Add the rule program's directory to Python path for local imports
            rule_dir = os.path.dirname(self.rule_program_path)
            if rule_dir not in sys.path:
                sys.path.insert(0, rule_dir)
                logger.debug(f"Added {rule_dir} to Python path for local imports")

            # Load the module
            spec = importlib.util.spec_from_file_location(
                "rule_module", self.rule_program_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Failed to load spec from {self.rule_program_path}"
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules["rule_module"] = module
            spec.loader.exec_module(module)

            # Check if apply function exists
            if not hasattr(module, "apply"):
                raise AttributeError(
                    f"Rule program file {self.rule_program_path} does not contain an 'apply' function"
                )

            self.apply_function = module.apply

            # Validate that apply is callable
            if not callable(self.apply_function):
                raise TypeError(
                    f"'apply' in {self.rule_program_path} is not callable"
                )

            logger.info(
                f"Successfully loaded rule program from {self.rule_program_path}"
            )

        except Exception as e:
            logger.error(f"Error loading rule program: {str(e)}")
            raise

    def apply(
        self, search_output: Any, default_result: Union[bool, Tuple[bool, ...]] = None
    ) -> Union[bool, Tuple[bool, ...]]:
        """
        Apply the rule to search program output.
        
        Args:
            search_output: Output from the search program (e.g., (p, elements))
            default_result: Default result to return if rule execution fails.
                           If None, raises exception. For single rule, use bool.
                           For multiple rules, use tuple[bool, ...].
        
        Returns:
            bool: Single rule result
            tuple[bool, ...]: Multiple rules result
            
        Raises:
            Exception: If rule execution fails and default_result is None
        """
        try:
            result = self.apply_function(search_output)

            # Validate return type
            if not isinstance(result, (bool, tuple)):
                raise TypeError(
                    f"Rule apply() must return bool or tuple[bool, ...], got {type(result)}"
                )

            # If tuple, validate all elements are bool
            if isinstance(result, tuple):
                if not all(isinstance(x, bool) for x in result):
                    raise TypeError(
                        f"Rule apply() tuple must contain only bool values, got {result}"
                    )
                if len(result) == 0:
                    raise ValueError("Rule apply() tuple cannot be empty")

            logger.debug(
                f"Rule applied successfully: {result} (type: {type(result).__name__})"
            )
            return result

        except Exception as e:
            if default_result is not None:
                logger.warning(
                    f"Rule execution failed: {e}. Using default result: {default_result}"
                )
                return default_result
            else:
                logger.error(f"Rule execution failed: {e}")
                raise

    def get_num_rules(self) -> int:
        """
        Get the number of rules (1 for single rule, >1 for multiple rules).
        
        This is determined by calling apply with a test input and checking
        the return type. If the rule program hasn't been tested yet, returns None.
        
        Note: This method requires a test call to determine the number of rules.
        For a more reliable approach, check the return type after calling apply().
        
        Returns:
            int: Number of rules (1 for bool, >1 for tuple length)
            None: If cannot determine (rule hasn't been called yet)
        """
        # We can't determine this without calling apply, so we return None
        # The caller should check the return type after calling apply()
        return None

    def __repr__(self) -> str:
        return f"RuleProgram(path={self.rule_program_path})"

 