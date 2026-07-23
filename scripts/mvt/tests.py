"""
MVT - Tests Unitaires.
========================
Tests exhaustifs pour tous les composants du MVT.
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
from mvt.model import MVT, GenerationResult


class TestMetricTensor(unittest.TestCase):
    """Tests du tenseur métrique dynamique G(t)."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.metric = MetricTensor(self.config)

    def test_initialization(self):
        """Le tenseur doit être symétrique défini positif."""
        G = self.metric.G
        N = self.config.ambient_dim
        self.assertEqual(G.shape, (N, N))
        np.testing.assert_array_almost_equal(G, G.T, decimal=10)

        # Vérifier la définie positivité
        eigvals = np.linalg.eigvalsh(G)
        self.assertTrue(np.all(eigvals > 0))

    def test_distance(self):
        """La distance doit être positive et respecter la symétrie."""
        p1 = np.random.randn(self.config.ambient_dim)
        p2 = np.random.randn(self.config.ambient_dim)

        d12 = self.metric.distance(p1, p2)
        d21 = self.metric.distance(p2, p1)

        self.assertGreater(d12, 0)
        self.assertAlmostEqual(d12, d21, places=10)

    def test_inner_product(self):
        """Le produit scalaire doit être symétrique."""
        v1 = np.random.randn(self.config.ambient_dim)
        v2 = np.random.randn(self.config.ambient_dim)

        ip12 = self.metric.inner_product(v1, v2)
        ip21 = self.metric.inner_product(v2, v1)
        self.assertAlmostEqual(ip12, ip21, places=10)

    def test_update_preserves_positivity(self):
        """La mise à jour doit préserver la définie positivité."""
        dG = np.random.randn(self.config.ambient_dim, self.config.ambient_dim) * 0.1
        self.metric.update(dG)

        eigvals = np.linalg.eigvalsh(self.metric.G)
        self.assertTrue(np.all(eigvals > 0))

    def test_scalar_curvature(self):
        """La courbure scalaire doit être calculable."""
        q = np.random.randn(self.config.ambient_dim) * 0.1
        R = self.metric.scalar_curvature(q)
        self.assertIsInstance(R, float)
        self.assertTrue(np.isfinite(R))

    def test_history_tracking(self):
        """L'historique doit se mettre à jour."""
        initial_len = len(self.metric._history)
        dG = np.random.randn(self.config.ambient_dim, self.config.ambient_dim) * 0.01
        self.metric.update(dG)
        self.assertEqual(len(self.metric._history), initial_len + 1)


class TestVectorField(unittest.TestCase):
    """Tests du champ vectoriel continu."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.field = VectorField(self.config)

    def test_add_source(self):
        """L'ajout d'une source doit créer un attracteur."""
        pos = np.random.randn(self.config.ambient_dim)
        self.field.add_source(pos, strength=2.0)
        self.assertEqual(len(self.field.control_points), 1)
        self.assertEqual(len(self.field.strengths), 1)

    def test_evaluation(self):
        """L'évaluation doit retourner un vecteur de la bonne dimension."""
        pos = np.random.randn(self.config.ambient_dim)
        self.field.add_source(pos, strength=1.0)

        q = np.random.randn(self.config.ambient_dim)
        F = self.field.evaluate(q)
        self.assertEqual(F.shape, (self.config.ambient_dim,))

    def test_potential(self):
        """Le potentiel doit être scalaire."""
        pos = np.random.randn(self.config.ambient_dim)
        self.field.add_source(pos, strength=1.0)

        q = np.random.randn(self.config.ambient_dim)
        V = self.field.potential(q)
        self.assertIsInstance(V, float)

    def test_gradient(self):
        """Le gradient doit avoir la bonne dimension."""
        pos = np.random.randn(self.config.ambient_dim)
        self.field.add_source(pos, strength=1.0)

        q = np.random.randn(self.config.ambient_dim)
        grad = self.field.gradient(q)
        self.assertEqual(grad.shape, (self.config.ambient_dim,))


class TestSemanticLagrangian(unittest.TestCase):
    """Tests du lagrangien sémantique."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.metric = MetricTensor(self.config)
        self.lagrangian = SemanticLagrangian(self.config, self.metric)

    def test_kinetic_energy_positive(self):
        """L'énergie cinétique doit être non négative."""
        q = np.random.randn(self.config.ambient_dim) * 0.1
        dq = np.random.randn(self.config.ambient_dim) * 0.1

        T = self.lagrangian.kinetic_energy(q, dq)
        self.assertGreaterEqual(T, 0)

    def test_potential_energy(self):
        """L'énergie potentielle doit être calculable."""
        q = np.random.randn(self.config.ambient_dim) * 0.1
        V = self.lagrangian.potential_energy(q)
        self.assertIsInstance(V, float)

    def test_lagrangian_value(self):
        """Le lagrangien L = T - V doit être calculable."""
        q = np.random.randn(self.config.ambient_dim) * 0.1
        dq = np.random.randn(self.config.ambient_dim) * 0.1

        L = self.lagrangian.lagrangian(q, dq)
        self.assertIsInstance(L, float)

    def test_euler_lagrange_rhs(self):
        """Le RHS d'Euler-Lagrange doit avoir la bonne dimension."""
        q = np.random.randn(self.config.ambient_dim) * 0.1
        dq = np.random.randn(self.config.ambient_dim) * 0.1

        ddq = self.lagrangian.euler_lagrange_rhs(q, dq, 0.0)
        self.assertEqual(ddq.shape, (self.config.ambient_dim,))

    def test_action_calculation(self):
        """L'action doit être calculable le long d'une trajectoire."""
        trajectory = np.random.randn(10, self.config.ambient_dim) * 0.1
        S = self.lagrangian.action(trajectory, dt=0.01)
        self.assertIsInstance(S, float)


class TestIntegrator(unittest.TestCase):
    """Tests de l'intégrateur RK4."""

    def setUp(self):
        self.config = MVTConfig(
            ambient_dim=16,
            dt=0.01,
            num_rk4_steps=50,
            seed=42,
        )
        self.metric = MetricTensor(self.config)
        self.lagrangian = SemanticLagrangian(self.config, self.metric)
        self.integrator = LagrangianIntegrator(self.config, self.lagrangian)

    def test_integration_shape(self):
        """La trajectoire doit avoir la bonne forme."""
        q0 = np.random.randn(self.config.ambient_dim) * 0.1
        trajectory = self.integrator.integrate(q0)
        self.assertEqual(trajectory.shape[0], self.config.num_rk4_steps + 1)
        self.assertEqual(trajectory.shape[1], self.config.ambient_dim)

    def test_trajectory_finite(self):
        """Toutes les valeurs doivent être finies."""
        q0 = np.random.randn(self.config.ambient_dim) * 0.1
        trajectory = self.integrator.integrate(q0)
        self.assertTrue(np.all(np.isfinite(trajectory)))

    def test_initial_position(self):
        """La position initiale doit être préservée."""
        q0 = np.random.randn(self.config.ambient_dim) * 0.1
        trajectory = self.integrator.integrate(q0)
        np.testing.assert_array_almost_equal(trajectory[0], q0)


class TestEncoder(unittest.TestCase):
    """Tests de l'encodeur."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.metric = MetricTensor(self.config)
        self.encoder = InputEncoder(self.config, self.metric)

    def test_encode_text(self):
        """L'encodage doit créer un champ avec des points de contrôle."""
        text = "Bonjour le monde"
        field = self.encoder.encode_text(text)
        self.assertGreater(len(field.control_points), 0)

    def test_encode_prompt(self):
        """L'encodage d'un prompt doit retourner un champ et une position."""
        text = "L'intelligence artificielle"
        field, q0 = self.encoder.encode_prompt(text)
        self.assertIsInstance(field, VectorField)
        self.assertEqual(q0.shape, (self.config.ambient_dim,))

    def test_deterministic(self):
        """L'encodage doit être déterministe."""
        text = "test determinisme"
        _, q1 = self.encoder.encode_prompt(text)
        _, q2 = self.encoder.encode_prompt(text)
        np.testing.assert_array_almost_equal(q1, q2)

    def test_similarity(self):
        """La similarité entre textes doit être entre 0 et 1."""
        sim = self.encoder.text_similarity("chat noir", "chat blanc")
        self.assertGreaterEqual(sim, 0)
        self.assertLessEqual(sim, 1)


class TestSyntopy(unittest.TestCase):
    """Tests de la couche de syntopie."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, syntopy_strength=1.0, seed=42)
        self.metric = MetricTensor(self.config)
        self.encoder = InputEncoder(self.config, self.metric)
        self.syntopy = SyntopicLayer(self.config, self.metric, self.encoder)

    def test_set_example(self):
        """La définition d'un exemple doit fonctionner."""
        self.syntopy.set_example("Exemple de syntopie")
        self.assertIsNotNone(self.syntopy._example_field)

    def test_clear_example(self):
        """L'effacement de l'exemple doit fonctionner."""
        self.syntopy.set_example("Exemple")
        self.syntopy.clear_example()
        self.assertIsNone(self.syntopy._example_field)

    def test_syntopy_score(self):
        """Le score de syntopie doit être entre 0 et 1."""
        self.syntopy.set_example("Exemple de test")
        trajectory = np.random.randn(10, self.config.ambient_dim) * 0.1
        score = self.syntopy.compute_syntopy_score(trajectory)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestPlasticity(unittest.TestCase):
    """Tests du moteur de plasticité."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.metric = MetricTensor(self.config)
        self.plasticity = TopologicalPlasticityEngine(self.config, self.metric)

    def test_update_metric(self):
        """La mise à jour doit préserver la définie positivité."""
        trajectory = np.random.randn(10, self.config.ambient_dim) * 0.1
        self.plasticity.update_metric(trajectory, success=True)
        eigvals = np.linalg.eigvalsh(self.metric.G)
        self.assertTrue(np.all(eigvals > 0))

    def test_singularity_detection(self):
        """La détection de singularité doit fonctionner."""
        trajectory = np.random.randn(10, self.config.ambient_dim) * 0.1
        has_sing, step = self.plasticity.detect_singularity(trajectory)
        self.assertIsInstance(has_sing, bool)

    def test_plasticity_stats(self):
        """Les stats doivent être un dict avec les clés attendues."""
        stats = self.plasticity.get_plasticity_stats()
        expected_keys = {
            "total_erosions", "total_sedimentations",
            "total_barriers", "total_channels",
        }
        self.assertTrue(expected_keys.issubset(stats.keys()))


class TestProjector(unittest.TestCase):
    """Tests du projecteur."""

    def setUp(self):
        self.config = MVTConfig(ambient_dim=16, seed=42)
        self.metric = MetricTensor(self.config)
        self.encoder = InputEncoder(self.config, self.metric)
        # Pré-encoder quelques mots
        self.encoder.encode_text("le chat mange la souris le chien court")
        self.projector = ManifoldProjector(self.config, self.metric, self.encoder)

    def test_project_trajectory(self):
        """La projection doit retourner du texte."""
        trajectory = np.random.randn(20, self.config.ambient_dim) * 0.1
        text = self.projector.project_trajectory(trajectory)
        self.assertIsInstance(text, str)

    def test_singularity_declaration(self):
        """La déclaration de singularité doit fonctionner."""
        q = np.zeros(self.config.ambient_dim)
        text = self.projector.singularity_declaration(q)
        self.assertIn("Singularité", text)


class TestMVTFull(unittest.TestCase):
    """Tests d'intégration du modèle complet."""

    def setUp(self):
        self.config = MVTConfig(
            ambient_dim=16,
            dt=0.02,
            num_rk4_steps=30,
            seed=42,
        )
        self.model = MVT(self.config)

    def test_initialization(self):
        """Le modèle doit s'initialiser correctement."""
        stats = self.model.get_stats()
        self.assertIn("metric", stats)
        self.assertIn("plasticity", stats)

    def test_basic_generation(self):
        """La génération basique doit fonctionner."""
        result = self.model.generate("Test de génération")
        self.assertIsInstance(result, GenerationResult)
        self.assertIsInstance(result.text, str)

    def test_one_shot_generation(self):
        """La génération one-shot doit fonctionner."""
        result = self.model.generate_with_example(
            "Exemple : A → 1, B → 2",
            "C → ?"
        )
        self.assertIsInstance(result, GenerationResult)

    def test_batch_generation(self):
        """La génération par lot doit fonctionner."""
        results = self.model.batch_generate(["Test 1", "Test 2", "Test 3"])
        self.assertEqual(len(results), 3)

    def test_stats_after_generation(self):
        """Les stats doivent se mettre à jour après génération."""
        self.model.generate("Test")
        stats = self.model.get_stats()
        self.assertGreater(stats["generation"]["total_generations"], 0)

    def test_reset(self):
        """Le reset doit fonctionner."""
        self.model.generate("Test")
        self.model.reset()
        stats = self.model.get_stats()
        self.assertEqual(stats["generation"]["total_generations"], 0)

    def test_trajectory_shape(self):
        """La trajectoire retournée doit avoir la bonne forme."""
        result = self.model.generate("Test", return_trajectory=True)
        self.assertEqual(
            result.trajectory.shape[1],
            self.config.ambient_dim
        )

    def test_no_nan(self):
        """Aucune valeur NaN ne doit apparaître."""
        result = self.model.generate("Test")
        self.assertFalse(np.isnan(result.action))
        self.assertFalse(np.isnan(result.kinetic_energy))
        self.assertFalse(np.isnan(result.potential_energy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
