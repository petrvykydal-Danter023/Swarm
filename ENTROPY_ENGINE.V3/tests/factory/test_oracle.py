import unittest
from factory.oracle.generator import HardStatesGenerator
from factory.oracle.expert import PrivilegedOracle

class TestOracle(unittest.TestCase):
    def setUp(self):
        self.config = {}
        self.generator = HardStatesGenerator(self.config)
        self.oracle = PrivilegedOracle(self.config)

    def test_generator_yields_scenarios(self):
        """Test if generator produces expected hard scenarios."""
        scenarios = list(self.generator.generate_hard_states(num_per_type=2))
        self.assertTrue(len(scenarios) > 0)
        self.assertIn("type", scenarios[0])
        self.assertIn("narrow_passage", [s["type"] for s in scenarios])

    def test_oracle_act_structure(self):
        """Test if oracle returns actions with correct structure."""
        mock_state = {} # Mock state
        action = self.oracle.act(mock_state)
        self.assertIn("motor", action)
        self.assertIn("comm", action)

if __name__ == '__main__':
    unittest.main()
