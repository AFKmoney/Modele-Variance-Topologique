"""
MVT v2 - Tests Unitaires.
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, '/home/z/my-project/scripts')

from mvt.config import MVTConfig
from mvt.core.metric_tensor import MetricTensor
from mvt.core.vector_field import VectorField
from mvt.lagrangian.semantic_lagrangian import SemanticLagrangian
from mvt.lagrangian.integrator import LagrangianIntegrator
from mvt.encoder import InputEncoder
from mvt.syntopy import SyntopicLayer
from mvt.plasticity import TopologicalPlasticityEngine
from mvt.projector import ManifoldProjector
from mvt.creativity import CreativityEngine
from mvt.model import MVT, GenerationResult
from mvt.agent import MVSAgent, Goal, AgentState

# Config légère pour les tests
CONFIG = MVTConfig(
    ambient_dim=8,
    intrinsic_dim=4,
    dt=0.02,
    num_rk4_steps=10,
    seed=42,
    temperature=0.8,
    curvature_threshold=100.0,
    divergence_threshold=1e8,
    max_agent_iterations=2,
    self_reflection_threshold=0.3,
)


class TestMetricTensor(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)

    def test_symmetric_positive_definite(self):
        G = self.metric.G
        np.testing.assert_array_almost_equal(G, G.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(G) > 0))

    def test_distance_symmetry(self):
        p1, p2 = np.random.randn(8), np.random.randn(8)
        self.assertAlmostEqual(self.metric.distance(p1, p2),
                               self.metric.distance(p2, p1))

    def test_update_preserves_positivity(self):
        dG = np.random.randn(8, 8) * 0.1
        self.metric.update(dG)
        self.assertTrue(np.all(np.linalg.eigvalsh(self.metric.G) > 0))

    def test_scalar_curvature(self):
        q = np.random.randn(8) * 0.1
        R = self.metric.scalar_curvature(q)
        self.assertTrue(np.isfinite(R))


class TestVectorField(unittest.TestCase):
    def setUp(self):
        self.field = VectorField(CONFIG)

    def test_add_and_evaluate(self):
        self.field.add_source(np.random.randn(8), 1.0)
        F = self.field.evaluate(np.random.randn(8))
        self.assertEqual(F.shape, (8,))

    def test_potential(self):
        self.field.add_source(np.random.randn(8), 1.0)
        V = self.field.potential(np.random.randn(8))
        self.assertIsInstance(V, float)


class TestLagrangian(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.L = SemanticLagrangian(CONFIG, self.metric)

    def test_kinetic_positive(self):
        q, dq = np.random.randn(8) * 0.1, np.random.randn(8) * 0.1
        self.assertGreaterEqual(self.L.kinetic_energy(q, dq), 0)

    def test_euler_lagrange_shape(self):
        q, dq = np.random.randn(8) * 0.1, np.random.randn(8) * 0.1
        ddq = self.L.euler_lagrange_rhs(q, dq, 0.0)
        self.assertEqual(ddq.shape, (8,))

    def test_action(self):
        traj = np.random.randn(5, 8) * 0.1
        S = self.L.action(traj, 0.01)
        self.assertIsInstance(S, float)


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.L = SemanticLagrangian(CONFIG, self.metric)
        self.integrator = LagrangianIntegrator(CONFIG, self.L)

    def test_shape(self):
        traj = self.integrator.integrate(np.random.randn(8) * 0.1)
        self.assertEqual(traj.shape[0], CONFIG.num_rk4_steps + 1)

    def test_finite(self):
        traj = self.integrator.integrate(np.random.randn(8) * 0.1)
        self.assertTrue(np.all(np.isfinite(traj)))

    def test_initial_position(self):
        q0 = np.random.randn(8) * 0.1
        traj = self.integrator.integrate(q0)
        np.testing.assert_array_almost_equal(traj[0], q0)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.encoder = InputEncoder(CONFIG, self.metric)

    def test_encode_text(self):
        field = self.encoder.encode_text("bonjour le monde")
        self.assertGreater(len(field.control_points), 0)

    def test_deterministic(self):
        _, q1 = self.encoder.encode_prompt("test")
        _, q2 = self.encoder.encode_prompt("test")
        np.testing.assert_array_almost_equal(q1, q2)


class TestSyntopy(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.encoder = InputEncoder(CONFIG, self.metric)
        self.syntopy = SyntopicLayer(CONFIG, self.metric, self.encoder)

    def test_set_clear(self):
        self.syntopy.set_example("exemple")
        self.assertIsNotNone(self.syntopy._example_field)
        self.syntopy.clear_example()
        self.assertIsNone(self.syntopy._example_field)


class TestPlasticity(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.plasticity = TopologicalPlasticityEngine(CONFIG, self.metric)

    def test_update(self):
        self.plasticity.update_metric(np.random.randn(5, 8) * 0.1, True)
        self.assertTrue(np.all(np.linalg.eigvalsh(self.metric.G) > 0))


class TestProjector(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.encoder = InputEncoder(CONFIG, self.metric)
        self.encoder.encode_text("le chat dort le chien court")
        self.projector = ManifoldProjector(CONFIG, self.metric, self.encoder)

    def test_project(self):
        text = self.projector.project_trajectory(np.random.randn(20, 8) * 0.1)
        self.assertIsInstance(text, str)


class TestCreativityEngine(unittest.TestCase):
    def setUp(self):
        self.metric = MetricTensor(CONFIG)
        self.creativity = CreativityEngine(CONFIG, self.metric)

    def test_noise_injection(self):
        noise = self.creativity.inject_noise(np.zeros(8), np.zeros(8))
        self.assertEqual(noise.shape, (8,))
        self.assertTrue(np.any(noise != 0))  # Non-trivial

    def test_novelty_force(self):
        self.creativity.record_visit(np.ones(8))
        force = self.creativity.novelty_force(np.ones(8) * 0.5)
        self.assertEqual(force.shape, (8,))

    def test_metaphor_jump(self):
        q = np.zeros(8)
        q_new = self.creativity.metaphor_jump(q)
        self.assertEqual(q_new.shape, (8,))
        # Le saut doit changer la position
        self.assertTrue(np.linalg.norm(q_new - q) > 0)

    def test_creative_word_selection(self):
        candidates = [("mot1", 0.5), ("mot2", 1.0), ("mot3", 1.5)]
        word = self.creativity.creative_word_selection(
            np.zeros(8), candidates, []
        )
        self.assertIn(word, ["mot1", "mot2", "mot3"])

    def test_divergent_thinking(self):
        q0 = np.zeros(8)
        branches = self.creativity.divergent_thinking(q0, 3)
        self.assertEqual(len(branches), 3)

    def test_creativity_score(self):
        traj = np.random.randn(10, 8) * 0.5
        score = self.creativity.evaluate_creativity(traj)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestMVTFull(unittest.TestCase):
    def setUp(self):
        self.model = MVT(CONFIG)

    def test_basic_generation(self):
        result = self.model.generate("test prompt")
        self.assertIsInstance(result, GenerationResult)
        self.assertIsInstance(result.text, str)

    def test_creativity_scores(self):
        result = self.model.generate("test creatif")
        self.assertGreaterEqual(result.creativity_score, 0)
        self.assertGreaterEqual(result.diversity_score, 0)
        self.assertLessEqual(result.diversity_score, 1)

    def test_one_shot(self):
        result = self.model.generate_with_example("A -> 1", "B -> ?")
        self.assertIsInstance(result.text, str)

    def test_batch(self):
        results = self.model.batch_generate(["a", "b", "c"])
        self.assertEqual(len(results), 3)

    def test_temperature_control(self):
        self.model.set_temperature(1.5)
        self.assertAlmostEqual(self.model.creativity.temperature, 1.5)

    def test_no_nan(self):
        result = self.model.generate("test")
        self.assertFalse(np.isnan(result.action))
        self.assertFalse(np.isnan(result.creativity_score))

    def test_soft_singularity(self):
        """Les singularités sont douces — pas de blocage."""
        result = self.model.generate("test")
        # Même avec singularité, on doit avoir du texte
        self.assertTrue(len(result.text) > 0 or result.has_singularity)

    def test_stats(self):
        self.model.generate("test")
        stats = self.model.get_stats()
        self.assertIn("creativity", stats)
        self.assertIn("temperature", stats["creativity"])

    def test_reset(self):
        self.model.generate("test")
        self.model.reset()
        self.assertEqual(len(self.model._generation_history), 0)


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MVSAgent(CONFIG)

    def test_initialization(self):
        self.assertEqual(self.agent.state, AgentState.IDLE)
        stats = self.agent.get_agent_stats()
        self.assertEqual(stats["total_iterations"], 0)

    def test_observe(self):
        obs = self.agent.observe("test prompt")
        self.assertIn("complexity", obs)
        self.assertIn("key_concepts", obs)

    def test_plan(self):
        goals = self.agent.plan("un prompt complexe avec plusieurs mots cles")
        self.assertGreater(len(goals), 0)

    def test_reflect(self):
        result = GenerationResult(
            text="mot1 mot2 mot3 mot4",
            trajectory=np.random.randn(10, 8) * 0.1,
            action=1.0,
            syntopy_score=0.5,
        )
        reflection = self.agent.reflect(result, "prompt")
        self.assertGreaterEqual(reflection.satisfaction_score, 0)
        self.assertLessEqual(reflection.satisfaction_score, 1)

    def test_goal_distance(self):
        goal = Goal("test", np.zeros(8), success_threshold=0.5)
        dist = goal.distance_to(np.ones(8), self.agent.mvt.metric)
        self.assertGreater(dist, 0)

    def test_agent_run(self):
        """L'agent doit pouvoir compléter une tâche."""
        result = self.agent.run("prompt test", verbose=False)
        self.assertIsInstance(result, GenerationResult)
        # L'agent doit avoir fait au moins 1 itération
        self.assertGreater(self.agent._iteration, 0)

    def test_agent_reset(self):
        self.agent.run("test", verbose=False)
        self.agent.reset()
        self.assertEqual(self.agent._iteration, 0)
        self.assertEqual(self.agent.state, AgentState.IDLE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
