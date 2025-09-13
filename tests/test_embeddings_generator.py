from unittest import TestCase

from src.probability_generator import ProbabilityGenerator
from src.embeddings_generator import EmbeddingsGenerator


class TestGenerator(TestCase):
    def test_generation(self):
        embeddings = [-1, 0, 1, 2, -1]
        p = ProbabilityGenerator(embeddings).probability_matrix()
        g = EmbeddingsGenerator(probability=p)
        generated = g.generate()
        self.assertEqual(generated, [0, 1, 2])
