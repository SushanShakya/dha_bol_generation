import torch
from unittest import TestCase
from src.probability_generator import ProbabilityGenerator


class TestProbabilityGenerator(TestCase):

    def setUp(self):
        self.embeddings = [0, 1, 2]
        self.g = ProbabilityGenerator(self.embeddings)

    def test_padding(self):
        """
        Test to see if start and end is added to
        the values
        """
        padded = self.g.pad()
        self.assertEqual(padded, [-1, 0, 1, 2, -1])

    def test_bigraph(self):
        """
        Test to see if bigraph generated is correct
        """
        bigraph = self.g.bigraph()
        self.assertEqual(bigraph, [(-1, 0), (0, 1), (1, 2), (2, -1)])

    def test_probabilty_matrix(self):
        matrix = self.g.probability_matrix()
        self.assertTrue(
            matrix.equal(
                torch.tensor(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                    ]
                )
            )
        )
