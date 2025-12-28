"""
Monte Carlo Tree Search (MCTS) explorer for OpenEvolve

This module implements MCTS algorithm to intelligently explore regions
in the rule-partitioned search space.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from openevolve.rule_partition import RulePartition

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    A node in the MCTS tree representing a region.
    
    Each node corresponds to a region in the partitioned space and maintains
    statistics for MCTS selection.
    """

    def __init__(
        self,
        region_id: Tuple[bool, ...],
        parent: Optional["MCTSNode"] = None,
    ):
        """
        Initialize MCTS node.
        
        Args:
            region_id: Region identifier (tuple[bool, ...])
            parent: Parent node (None for root)
        """
        self.region_id = region_id
        self.parent = parent
        self.children: List["MCTSNode"] = []

        # MCTS statistics
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.average_reward: float = 0.0

    def add_child(self, child: "MCTSNode") -> None:
        """Add a child node"""
        self.children.append(child)

    def update(self, reward: float) -> None:
        """
        Update node statistics with a new reward.
        
        Args:
            reward: Reward value from exploration
        """
        self.visits += 1
        self.total_reward += reward
        self.average_reward = self.total_reward / self.visits

    def ucb1(self, exploration_constant: float, parent_visits: int) -> float:
        """
        Calculate UCB1 value for node selection.
        
        UCB1 = (average_reward) + C * sqrt(ln(parent_visits) / visits)
        
        Args:
            exploration_constant: Exploration constant (C)
            parent_visits: Number of visits to parent node
            
        Returns:
            UCB1 value
        """
        if self.visits == 0:
            # Unvisited nodes get infinite UCB1 value
            return float("inf")

        exploitation = self.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )

        return exploitation + exploration

    def is_fully_expanded(self, all_regions: List[Tuple[bool, ...]]) -> bool:
        """
        Check if node is fully expanded (all possible children exist).
        
        Args:
            all_regions: List of all available regions
            
        Returns:
            True if all children are expanded
        """
        # A node is fully expanded if it has children for all regions
        # In our case, each region is a direct child of root
        return len(self.children) >= len(all_regions)

    def get_best_child(self, exploration_constant: float) -> Optional["MCTSNode"]:
        """
        Get child with highest UCB1 value.
        
        Args:
            exploration_constant: Exploration constant for UCB1
            
        Returns:
            Best child node, or None if no children
        """
        if not self.children:
            return None

        best_child = max(
            self.children,
            key=lambda child: child.ucb1(exploration_constant, self.visits),
        )
        return best_child

    def __repr__(self) -> str:
        return (
            f"MCTSNode(region={self.region_id}, "
            f"visits={self.visits}, "
            f"avg_reward={self.average_reward:.4f})"
        )


class MCTSExplorer:
    """
    Monte Carlo Tree Search explorer for region selection.
    
    Uses MCTS algorithm to balance exploration and exploitation when
    selecting regions to explore in the partitioned search space.
    
    Example:
        explorer = MCTSExplorer(exploration_constant=1.414)
        region_id = explorer.select_region(partition)
        # ... explore region ...
        explorer.update(region_id, reward=0.85)
    """

    def __init__(
        self,
        exploration_constant: float = 1.414,
        reward_function: str = "best_fitness",
    ):
        """
        Initialize MCTS explorer.
        
        Args:
            exploration_constant: Exploration constant for UCB1 (default: sqrt(2))
            reward_function: How to compute reward from region stats.
                            Options: "best_fitness", "average_fitness", "max_improvement"
        """
        self.exploration_constant = exploration_constant
        self.reward_function = reward_function

        # Root node (represents all regions)
        self.root: Optional[MCTSNode] = None

        # Map from region_id to node for quick lookup
        self.region_to_node: Dict[Tuple[bool, ...], MCTSNode] = {}

        logger.info(
            f"Initialized MCTSExplorer (C={exploration_constant}, "
            f"reward={reward_function})"
        )

    def _create_root_if_needed(self) -> None:
        """Create root node if it doesn't exist"""
        if self.root is None:
            # Root node has no region_id (represents all regions)
            self.root = MCTSNode(region_id=(), parent=None)

    def _get_or_create_node(
        self, region_id: Tuple[bool, ...]
    ) -> MCTSNode:
        """
        Get existing node or create new one for a region.
        
        Args:
            region_id: Region identifier
            
        Returns:
            MCTSNode for the region
        """
        if region_id in self.region_to_node:
            return self.region_to_node[region_id]

        # Create new node
        node = MCTSNode(region_id=region_id, parent=self.root)
        self.region_to_node[region_id] = node

        # Add as child of root
        if self.root:
            self.root.add_child(node)

        logger.debug(f"Created new MCTS node for region {region_id}")
        return node

    def _compute_reward(
        self, partition: RulePartition, region_id: Tuple[bool, ...]
    ) -> float:
        """
        Compute reward for a region based on reward function.
        
        Args:
            partition: RulePartition instance
            region_id: Region identifier
            
        Returns:
            Reward value (0.0 to 1.0)
        """
        stats = partition.get_region_stats(region_id)

        if self.reward_function == "best_fitness":
            # Use best fitness as reward
            best_fitness = stats.get("best_fitness")
            if best_fitness is None:
                return 0.0
            # Normalize to [0, 1] (assuming fitness is already in [0, 1])
            return max(0.0, min(1.0, best_fitness))

        elif self.reward_function == "average_fitness":
            # Use average fitness as reward
            avg_fitness = stats.get("average_fitness", 0.0)
            return max(0.0, min(1.0, avg_fitness))

        elif self.reward_function == "max_improvement":
            # Use improvement over baseline (simplified: use best_fitness)
            best_fitness = stats.get("best_fitness")
            if best_fitness is None:
                return 0.0
            return max(0.0, min(1.0, best_fitness))

        else:
            logger.warning(
                f"Unknown reward function: {self.reward_function}, "
                f"using best_fitness"
            )
            best_fitness = stats.get("best_fitness")
            return max(0.0, min(1.0, best_fitness)) if best_fitness else 0.0

    def select_region(
        self, partition: RulePartition, simulations: int = 1
    ) -> Tuple[bool, ...]:
        """
        Select a region to explore using MCTS.
        
        Args:
            partition: RulePartition instance
            simulations: Number of MCTS simulations to run (default: 1)
            
        Returns:
            Selected region identifier
        """
        self._create_root_if_needed()

        all_regions = partition.get_all_regions()

        # If no regions exist yet, we can't select anything
        if not all_regions:
            logger.warning("No regions available for MCTS selection")
            return ()

        # Ensure all regions have nodes
        for region_id in all_regions:
            self._get_or_create_node(region_id)

        # Run MCTS simulations
        for _ in range(simulations):
            # Selection: Select path from root to leaf
            selected_node = self._select(self.root, all_regions)

            # Expansion: If node is not fully expanded, expand it
            # (In our case, all regions are direct children of root,
            # so expansion happens automatically when we create nodes)

            # Simulation: Get reward from region statistics
            reward = self._compute_reward(partition, selected_node.region_id)

            # Backpropagation: Update path from selected node to root
            self._backpropagate(selected_node, reward)

        # Select best child of root based on UCB1
        best_child = self.root.get_best_child(self.exploration_constant)

        if best_child is None:
            # Fallback: random selection
            logger.warning("No best child found, using random selection")
            return random.choice(all_regions)

        logger.debug(
            f"MCTS selected region {best_child.region_id} "
            f"(UCB1={best_child.ucb1(self.exploration_constant, self.root.visits):.4f})"
        )

        return best_child.region_id

    def _select(
        self, node: MCTSNode, all_regions: List[Tuple[bool, ...]]
    ) -> MCTSNode:
        """
        Selection phase: Select path from node to leaf.
        
        In our case, all regions are direct children of root, so we
        simply select the best child of root.
        
        Args:
            node: Starting node (usually root)
            all_regions: List of all available regions
            
        Returns:
            Selected leaf node (region node)
        """
        # If node has no children, return it (shouldn't happen for root)
        if not node.children:
            return node

        # Select best child using UCB1
        best_child = node.get_best_child(self.exploration_constant)

        if best_child is None:
            # Fallback: return first child or node itself
            return node.children[0] if node.children else node

        # In our flat structure, best_child is already a leaf (region node)
        return best_child

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagation phase: Update statistics from node to root.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward value to propagate
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def update(
        self, region_id: Tuple[bool, ...], reward: Optional[float] = None, partition: Optional[RulePartition] = None
    ) -> None:
        """
        Update MCTS tree with exploration result.
        
        Args:
            region_id: Explored region
            reward: Reward obtained from exploration (if None, will compute from partition)
            partition: Optional RulePartition to compute reward from stats (used if reward is None)
        """
        self._create_root_if_needed()

        # Compute reward if not provided but partition is available
        if reward is None:
            if partition is not None:
                reward = self._compute_reward(partition, region_id)
            else:
                raise ValueError("Either reward or partition must be provided")

        # Get or create node for region
        node = self._get_or_create_node(region_id)

        # Update node and backpropagate
        self._backpropagate(node, reward)

        logger.debug(
            f"Updated MCTS node for region {region_id} with reward {reward:.4f}"
        )

    def get_node_stats(self, region_id: Tuple[bool, ...]) -> Optional[Dict]:
        """
        Get statistics for a region's MCTS node.
        
        Args:
            region_id: Region identifier
            
        Returns:
            Dictionary with node statistics, or None if node doesn't exist
        """
        node = self.region_to_node.get(region_id)
        if node is None:
            return None

        return {
            "region_id": node.region_id,
            "visits": node.visits,
            "total_reward": node.total_reward,
            "average_reward": node.average_reward,
            "ucb1": (
                node.ucb1(self.exploration_constant, self.root.visits)
                if self.root
                else 0.0
            ),
        }

    def get_summary(self) -> Dict:
        """
        Get summary of MCTS tree state.
        
        Returns:
            Dictionary with MCTS summary
        """
        summary = {
            "root_visits": self.root.visits if self.root else 0,
            "num_nodes": len(self.region_to_node),
            "exploration_constant": self.exploration_constant,
            "reward_function": self.reward_function,
            "nodes": {},
        }

        for region_id, node in self.region_to_node.items():
            summary["nodes"][str(region_id)] = {
                "visits": node.visits,
                "average_reward": node.average_reward,
                "total_reward": node.total_reward,
            }

        return summary

    def clear(self) -> None:
        """Clear MCTS tree"""
        self.root = None
        self.region_to_node.clear()
        logger.info("Cleared MCTS tree")

    def __repr__(self) -> str:
        return (
            f"MCTSExplorer(nodes={len(self.region_to_node)}, "
            f"root_visits={self.root.visits if self.root else 0})"
        )

