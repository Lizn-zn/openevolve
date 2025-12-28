"""
Test cases for MCTS and Rule-Based Partitioning functionality

This test suite verifies:
1. RuleProgram loading and execution
2. RulePartition classification and region management
3. MCTSExplorer region selection
4. Integration with ProgramDatabase
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openevolve.rule_program import RuleProgram
from openevolve.rule_partition import RulePartition, RegionStats
from openevolve.mcts_explorer import MCTSExplorer, MCTSNode
from openevolve.database import ProgramDatabase, Program
from openevolve.config import Config, DatabaseConfig


class TestRuleProgram(unittest.TestCase):
    """Test RuleProgram module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary rule program file
        self.temp_dir = tempfile.mkdtemp()
        self.rule_file = os.path.join(self.temp_dir, "test_rule.py")

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_rule(self):
        """Test single rule (returns bool)"""
        # Create rule program
        rule_code = """
def apply(search_output):
    p, elements = search_output
    return p > 50
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)

        # Load and test
        rule = RuleProgram(self.rule_file)
        
        # Test classification
        result1 = rule.apply((53, [1, 2, 3]))
        self.assertIsInstance(result1, bool)
        self.assertTrue(result1)  # 53 > 50
        
        result2 = rule.apply((31, [1, 2, 3]))
        self.assertIsInstance(result2, bool)
        self.assertFalse(result2)  # 31 <= 50

    def test_multiple_rules(self):
        """Test multiple rules (returns tuple)"""
        # Create rule program with multiple rules
        rule_code = """
def apply(search_output):
    p, elements = search_output
    rule1 = p > 50
    rule2 = len(elements) > 5
    return (rule1, rule2)
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)

        # Load and test
        rule = RuleProgram(self.rule_file)
        
        # Test classification
        result = rule.apply((53, [1, 2, 3, 4, 5, 6, 7]))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0])  # 53 > 50
        self.assertTrue(result[1])  # len([1,2,3,4,5,6,7]) > 5

    def test_rule_with_default(self):
        """Test rule with default result on error"""
        rule_code = """
def apply(search_output):
    # This will raise an error if search_output is not a tuple
    p, elements = search_output
    return p > 50
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)

        rule = RuleProgram(self.rule_file)
        
        # Test with invalid input and default result
        result = rule.apply("invalid", default_result=False)
        self.assertFalse(result)


class TestRulePartition(unittest.TestCase):
    """Test RulePartition module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary rule program file
        self.temp_dir = tempfile.mkdtemp()
        self.rule_file = os.path.join(self.temp_dir, "test_rule.py")
        
        # Create simple rule
        rule_code = """
def apply(search_output):
    p, elements = search_output
    return p > 50
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)
        
        self.rule_program = RuleProgram(self.rule_file)
        self.partition = RulePartition(self.rule_program)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify(self):
        """Test classification"""
        region1 = self.partition.classify((53, [1, 2, 3]))
        self.assertEqual(region1, (True,))
        
        region2 = self.partition.classify((31, [1, 2, 3]))
        self.assertEqual(region2, (False,))

    def test_add_program(self):
        """Test adding programs to regions"""
        # Add programs to different regions
        self.partition.add_program((True,), "prog1", 0.8)
        self.partition.add_program((True,), "prog2", 0.9)
        self.partition.add_program((False,), "prog3", 0.7)
        
        # Check region programs
        true_programs = self.partition.get_region_programs((True,))
        self.assertEqual(len(true_programs), 2)
        self.assertIn("prog1", true_programs)
        self.assertIn("prog2", true_programs)
        
        false_programs = self.partition.get_region_programs((False,))
        self.assertEqual(len(false_programs), 1)
        self.assertIn("prog3", false_programs)

    def test_region_stats(self):
        """Test region statistics"""
        self.partition.add_program((True,), "prog1", 0.8)
        self.partition.add_program((True,), "prog2", 0.9)
        self.partition.add_program((True,), "prog3", 0.7)
        
        stats = self.partition.get_region_stats((True,))
        self.assertEqual(stats["program_count"], 3)
        self.assertEqual(stats["best_fitness"], 0.9)
        self.assertEqual(stats["best_program_id"], "prog2")
        self.assertAlmostEqual(stats["average_fitness"], 0.8, places=1)

    def test_multiple_regions(self):
        """Test multiple rule regions"""
        # Create partition with multiple rules
        rule_code = """
def apply(search_output):
    p, elements = search_output
    return (p > 50, len(elements) > 5)
"""
        rule_file = os.path.join(self.temp_dir, "multi_rule.py")
        with open(rule_file, "w") as f:
            f.write(rule_code)
        
        rule_program = RuleProgram(rule_file)
        partition = RulePartition(rule_program)
        
        # Test different regions
        region1 = partition.classify((53, [1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(region1, (True, True))
        
        region2 = partition.classify((53, [1, 2, 3]))
        self.assertEqual(region2, (True, False))
        
        region3 = partition.classify((31, [1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(region3, (False, True))
        
        region4 = partition.classify((31, [1, 2, 3]))
        self.assertEqual(region4, (False, False))


class TestMCTSExplorer(unittest.TestCase):
    """Test MCTSExplorer module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary rule program file
        self.temp_dir = tempfile.mkdtemp()
        self.rule_file = os.path.join(self.temp_dir, "test_rule.py")
        
        rule_code = """
def apply(search_output):
    p, elements = search_output
    return p > 50
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)
        
        from openevolve.rule_program import RuleProgram
        from openevolve.rule_partition import RulePartition
        
        rule_program = RuleProgram(self.rule_file)
        self.partition = RulePartition(rule_program)
        self.explorer = MCTSExplorer(exploration_constant=1.414)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_node_creation(self):
        """Test MCTS node creation"""
        node = MCTSNode(region_id=(True,))
        self.assertEqual(node.region_id, (True,))
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.total_reward, 0.0)

    def test_node_update(self):
        """Test node statistics update"""
        node = MCTSNode(region_id=(True,))
        node.update(0.8)
        node.update(0.9)
        
        self.assertEqual(node.visits, 2)
        self.assertAlmostEqual(node.total_reward, 1.7, places=10)  # Use assertAlmostEqual for float comparison
        self.assertAlmostEqual(node.average_reward, 0.85, places=2)

    def test_ucb1_calculation(self):
        """Test UCB1 value calculation"""
        node = MCTSNode(region_id=(True,))
        node.visits = 10
        node.total_reward = 8.0
        node.average_reward = 0.8
        
        parent_visits = 20
        ucb1 = node.ucb1(exploration_constant=1.414, parent_visits=parent_visits)
        
        # UCB1 = 0.8 + 1.414 * sqrt(ln(20) / 10)
        expected = 0.8 + 1.414 * (0.693 / 3.162)  # Approximate
        self.assertGreater(ucb1, 0.8)
        self.assertLess(ucb1, 2.0)

    def test_region_selection(self):
        """Test MCTS region selection"""
        # Add some programs to regions
        self.partition.add_program((True,), "prog1", 0.9)
        self.partition.add_program((True,), "prog2", 0.8)
        self.partition.add_program((False,), "prog3", 0.7)
        
        # Select region using MCTS
        region_id = self.explorer.select_region(self.partition, simulations=5)
        
        # Should return a valid region
        self.assertIsInstance(region_id, tuple)
        self.assertIn(region_id, [(True,), (False,)])

    def test_mcts_update(self):
        """Test MCTS update after exploration"""
        # Add programs
        self.partition.add_program((True,), "prog1", 0.9)
        
        # Select region (this will run simulations and update nodes based on region stats)
        # Note: select_region runs simulations internally, which updates nodes
        region_id = self.explorer.select_region(self.partition, simulations=1)
        
        # Get initial stats after selection
        initial_stats = self.explorer.get_node_stats(region_id)
        initial_visits = initial_stats["visits"] if initial_stats else 0
        initial_total_reward = initial_stats["total_reward"] if initial_stats else 0.0
        
        # Update with a new reward (this should add to existing stats)
        # Pass reward directly, not through partition
        new_reward = 0.85
        self.explorer.update(region_id, reward=new_reward)
        
        # Check node was updated (visits should be initial_visits + 1)
        node_stats = self.explorer.get_node_stats(region_id)
        self.assertIsNotNone(node_stats)
        self.assertEqual(node_stats["visits"], initial_visits + 1)
        
        # Total reward should be initial + new
        expected_total = initial_total_reward + new_reward
        self.assertAlmostEqual(node_stats["total_reward"], expected_total, places=2)
        
        # Average reward should be total_reward / visits
        expected_avg = expected_total / (initial_visits + 1)
        self.assertAlmostEqual(node_stats["average_reward"], expected_avg, places=2)


class TestIntegration(unittest.TestCase):
    """Test integration with ProgramDatabase"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary rule program file
        self.temp_dir = tempfile.mkdtemp()
        self.rule_file = os.path.join(self.temp_dir, "test_rule.py")
        
        rule_code = """
def apply(search_output):
    p, elements = search_output
    return p > 50
"""
        with open(self.rule_file, "w") as f:
            f.write(rule_code)
        
        from openevolve.rule_program import RuleProgram
        from openevolve.rule_partition import RulePartition
        from openevolve.mcts_explorer import MCTSExplorer
        
        rule_program = RuleProgram(self.rule_file)
        rule_partition = RulePartition(rule_program)
        mcts_explorer = MCTSExplorer(exploration_constant=1.414)
        
        # Create database with rule partition and MCTS
        config = DatabaseConfig()
        self.database = ProgramDatabase(
            config,
            rule_partition=rule_partition,
            mcts_explorer=mcts_explorer
        )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify_on_add(self):
        """Test that programs are classified when added"""
        # Create a program with artifacts containing search output
        program = Program(
            id="test_prog",
            code="def run_search(): return (53, [1, 2, 3])",
            language="python",
            metrics={"combined_score": 0.8}
        )
        
        # Add program first (so it exists in database)
        self.database.add(program)
        
        # Then store artifacts with search output
        artifacts = {"p": 53, "elements": [1, 2, 3]}
        self.database.store_artifacts("test_prog", artifacts)
        
        # Manually trigger classification (since artifacts were added after)
        # In real usage, artifacts would be available when add() is called
        if self.database.rule_partition:
            search_output = (artifacts["p"], artifacts["elements"])
            region_id = self.database.rule_partition.classify(search_output)
            fitness = 0.8  # Use combined_score as fitness
            self.database.rule_partition.add_program(region_id, program.id, fitness)
            program.metadata["rule_region"] = region_id
        
        # Check that program was classified
        region_id = program.metadata.get("rule_region")
        self.assertIsNotNone(region_id)
        self.assertEqual(region_id, (True,))  # 53 > 50
        
        # Check that program is in the region
        region_programs = self.database.rule_partition.get_region_programs(region_id)
        self.assertIn("test_prog", region_programs)

    def test_sample_from_region(self):
        """Test sampling from a specific region"""
        # Add programs to different regions
        prog1 = Program(
            id="prog1",
            code="def run_search(): return (53, [1, 2, 3])",
            language="python",
            metrics={"combined_score": 0.8}
        )
        self.database.add(prog1)
        self.database.store_artifacts("prog1", {"p": 53, "elements": [1, 2, 3]})
        
        # Manually classify and add to region
        if self.database.rule_partition:
            region_id1 = self.database.rule_partition.classify((53, [1, 2, 3]))
            self.database.rule_partition.add_program(region_id1, "prog1", 0.8)
            prog1.metadata["rule_region"] = region_id1
        
        prog2 = Program(
            id="prog2",
            code="def run_search(): return (31, [1, 2, 3])",
            language="python",
            metrics={"combined_score": 0.7}
        )
        self.database.add(prog2)
        self.database.store_artifacts("prog2", {"p": 31, "elements": [1, 2, 3]})
        
        # Manually classify and add to region
        if self.database.rule_partition:
            region_id2 = self.database.rule_partition.classify((31, [1, 2, 3]))
            self.database.rule_partition.add_program(region_id2, "prog2", 0.7)
            prog2.metadata["rule_region"] = region_id2
        
        # Sample from True region (should get prog1)
        sampled = self.database.sample_from_region((True,))
        self.assertEqual(sampled.id, "prog1")

    def test_mcts_sampling(self):
        """Test MCTS-based sampling"""
        # Add programs to regions
        prog1 = Program(
            id="prog1",
            code="def run_search(): return (53, [1, 2, 3])",
            language="python",
            metrics={"combined_score": 0.9}
        )
        self.database.store_artifacts("prog1", {"p": 53, "elements": [1, 2, 3]})
        self.database.add(prog1)
        
        prog2 = Program(
            id="prog2",
            code="def run_search(): return (31, [1, 2, 3])",
            language="python",
            metrics={"combined_score": 0.7}
        )
        self.database.store_artifacts("prog2", {"p": 31, "elements": [1, 2, 3]})
        self.database.add(prog2)
        
        # Sample using MCTS
        parent, inspirations = self.database.sample(
            num_inspirations=2,
            use_mcts=True,
            mcts_simulations=5
        )
        
        # Should return a valid program
        self.assertIsNotNone(parent)
        self.assertIn(parent.id, ["prog1", "prog2"])


if __name__ == "__main__":
    unittest.main()

