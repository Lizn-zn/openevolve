"""
Rule-based space partitioning module for OpenEvolve

This module provides functionality to partition the search space based on
rule programs that classify search program outputs.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from openevolve.rule_program import RuleProgram

logger = logging.getLogger(__name__)


class RegionStats:
    """Statistics for a region in the partitioned space"""

    def __init__(self):
        self.program_count: int = 0
        self.best_fitness: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.total_fitness: float = 0.0
        self.average_fitness: float = 0.0
        self.exploration_count: int = 0  # Number of times this region was explored

    def update(self, program_id: str, fitness: float) -> None:
        """Update statistics with a new program"""
        self.program_count += 1
        self.total_fitness += fitness
        self.average_fitness = self.total_fitness / self.program_count

        # Update best fitness
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_program_id = program_id

    def increment_exploration(self) -> None:
        """Increment exploration count"""
        self.exploration_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "program_count": self.program_count,
            "best_fitness": self.best_fitness,
            "best_program_id": self.best_program_id,
            "average_fitness": self.average_fitness,
            "exploration_count": self.exploration_count,
        }

    def __repr__(self) -> str:
        return (
            f"RegionStats(count={self.program_count}, "
            f"best_fitness={self.best_fitness:.4f if self.best_fitness else None}, "
            f"explorations={self.exploration_count})"
        )


class RulePartition:
    """
    Manages space partitioning based on rule programs.
    
    Partitions the search space into regions based on rule classification
    results. Each region maintains a list of programs and statistics.
    
    Example:
        rule_program = RuleProgram("rule.py")
        partition = RulePartition(rule_program)
        
        # Classify search output
        region_id = partition.classify((53, [1, 2, 3]))
        
        # Add program to region
        partition.add_program(region_id, "program_123", 0.85)
        
        # Get region statistics
        stats = partition.get_region_stats(region_id)
    """

    def __init__(self, rule_program: RuleProgram):
        """
        Initialize RulePartition with a rule program.
        
        Args:
            rule_program: RuleProgram instance for classification
        """
        self.rule_program = rule_program

        # Store programs by region: {region_id: [program_ids...]}
        self.region_programs: Dict[Tuple[bool, ...], List[str]] = defaultdict(list)

        # Store statistics for each region
        self.region_stats: Dict[Tuple[bool, ...], RegionStats] = defaultdict(
            RegionStats
        )

        # Track which program belongs to which region
        self.program_to_region: Dict[str, Tuple[bool, ...]] = {}

        logger.info("Initialized RulePartition")

    def _normalize_region_id(
        self, result: Any
    ) -> Tuple[bool, ...]:
        """
        Normalize rule result to tuple format.
        
        Converts bool to (bool,) for consistency.
        
        Args:
            result: Rule apply() result (bool or tuple[bool, ...])
            
        Returns:
            tuple[bool, ...]: Normalized region identifier
        """
        if isinstance(result, bool):
            return (result,)
        elif isinstance(result, tuple):
            return result
        else:
            raise TypeError(
                f"Rule result must be bool or tuple[bool, ...], got {type(result)}"
            )

    def classify(self, search_output: Any) -> Tuple[bool, ...]:
        """
        Classify search output into a region.
        
        Args:
            search_output: Output from the search program (e.g., (p, elements))
            
        Returns:
            tuple[bool, ...]: Region identifier
            
        Raises:
            Exception: If rule execution fails
        """
        try:
            result = self.rule_program.apply(search_output)
            region_id = self._normalize_region_id(result)
            
            logger.debug(f"Classified search output to region {region_id}")
            return region_id
        except Exception as e:
            logger.error(f"Failed to classify search output: {e}")
            raise

    def add_program(
        self,
        region_id: Tuple[bool, ...],
        program_id: str,
        fitness: float,
    ) -> None:
        """
        Add a program to a region.
        
        Args:
            region_id: Region identifier (tuple[bool, ...])
            program_id: Program identifier
            fitness: Fitness score of the program
        """
        # Normalize region_id
        region_id = self._normalize_region_id(region_id)

        # Add program to region (avoid duplicates)
        if program_id not in self.region_programs[region_id]:
            self.region_programs[region_id].append(program_id)
        else:
            logger.debug(
                f"Program {program_id} already in region {region_id}, updating stats only"
            )

        # Update statistics
        self.region_stats[region_id].update(program_id, fitness)

        # Track program-to-region mapping
        self.program_to_region[program_id] = region_id

        logger.debug(
            f"Added program {program_id} to region {region_id} "
            f"(fitness={fitness:.4f}, region_size={len(self.region_programs[region_id])})"
        )

    def remove_program(self, program_id: str) -> bool:
        """
        Remove a program from its region.
        
        Args:
            program_id: Program identifier to remove
            
        Returns:
            bool: True if program was found and removed, False otherwise
        """
        if program_id not in self.program_to_region:
            return False

        region_id = self.program_to_region[program_id]

        # Remove from region programs list
        if program_id in self.region_programs[region_id]:
            self.region_programs[region_id].remove(program_id)

        # Remove from mapping
        del self.program_to_region[program_id]

        # Note: We don't update stats here as it would require recalculating
        # all statistics. Stats are maintained for historical tracking.

        logger.debug(f"Removed program {program_id} from region {region_id}")
        return True

    def get_region_programs(self, region_id: Tuple[bool, ...]) -> List[str]:
        """
        Get list of program IDs in a region.
        
        Args:
            region_id: Region identifier (tuple[bool, ...])
            
        Returns:
            List of program IDs in the region
        """
        region_id = self._normalize_region_id(region_id)
        return self.region_programs.get(region_id, []).copy()

    def get_region_stats(self, region_id: Tuple[bool, ...]) -> Dict[str, Any]:
        """
        Get statistics for a region.
        
        Args:
            region_id: Region identifier (tuple[bool, ...])
            
        Returns:
            Dictionary containing region statistics
        """
        region_id = self._normalize_region_id(region_id)
        stats = self.region_stats.get(region_id, RegionStats())
        return stats.to_dict()

    def get_all_regions(self) -> List[Tuple[bool, ...]]:
        """
        Get all explored regions.
        
        Returns:
            List of region identifiers (tuples)
        """
        return list(self.region_programs.keys())

    def get_region_for_program(self, program_id: str) -> Optional[Tuple[bool, ...]]:
        """
        Get the region that contains a program.
        
        Args:
            program_id: Program identifier
            
        Returns:
            Region identifier if program exists, None otherwise
        """
        return self.program_to_region.get(program_id)

    def increment_exploration(self, region_id: Tuple[bool, ...]) -> None:
        """
        Increment exploration count for a region.
        
        Args:
            region_id: Region identifier
        """
        region_id = self._normalize_region_id(region_id)
        self.region_stats[region_id].increment_exploration()

    def get_total_programs(self) -> int:
        """
        Get total number of programs across all regions.
        
        Returns:
            Total program count
        """
        return len(self.program_to_region)

    def get_region_count(self) -> int:
        """
        Get number of explored regions.
        
        Returns:
            Number of regions
        """
        return len(self.region_programs)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the partition state.
        
        Returns:
            Dictionary with partition summary
        """
        summary = {
            "total_regions": self.get_region_count(),
            "total_programs": self.get_total_programs(),
            "regions": {},
        }

        for region_id in self.get_all_regions():
            stats = self.get_region_stats(region_id)
            summary["regions"][str(region_id)] = {
                "program_count": stats["program_count"],
                "best_fitness": stats["best_fitness"],
                "average_fitness": stats["average_fitness"],
                "exploration_count": stats["exploration_count"],
            }

        return summary

    def clear(self) -> None:
        """Clear all regions and programs"""
        self.region_programs.clear()
        self.region_stats.clear()
        self.program_to_region.clear()
        logger.info("Cleared all regions and programs")

    def __repr__(self) -> str:
        return (
            f"RulePartition(regions={self.get_region_count()}, "
            f"programs={self.get_total_programs()})"
        )

