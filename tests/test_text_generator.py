from unittest import TestCase

from src.probability_generator import ProbabilityGenerator
from src.text_embedder import TextEmbedder
from src.text_generator import TextGenerator


class TestTextGenerator(TestCase):
    def test_generation(self):
        embeddings = [0, 1, 2]
        vocab = {
            "a": 0,
            "b": 1,
            "c": 2,
        }
        p = ProbabilityGenerator(embeddings).probability_matrix()
        g = TextGenerator(
            probability=p,
            vocabulary=vocab,
        )
        generated = g.generate()
        self.assertEqual(generated, "a b c")
