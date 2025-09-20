from unittest import TestCase
import torch.nn.functional as F

from src.nn import NN
from src.vocab_generator import VocabGenerator


class TestNN(TestCase):
    def test_init_weights(self):
        vocab = ["a", "b", "c"]
        nn = NN(vocab)
        # self.assertEqual(nn.weights.shape, (4, 1))
        self.assertEqual(nn.weights.shape, (4, 4))

    def test_compute(self):
        vocab = ["a", "b", "c"]
        nn = NN(vocab)
        inputs = [0, -1]
        output = nn.compute(inputs)

        self.assertEqual(output.shape, (2, 4))

    def test_gradient_descent(self):
        vocab = ["a", "b", "c"]
        nn = NN(vocab)
        inputs = [-1, 0]
        expected_outputs = [0, 1]
        for _ in range(10):
            loss = nn.gradient_descent(inputs, expected_outputs)

        print(loss)


# [[1, 0, 0, 0]
#  [0, 1, 0, 0]] (2 x 4)
# [[a, e],
# [b, f],
# [c, g],
# [d, h]] (4 x 2)

# Matrix Multiplication
#
# [[a, e],
# [b, f]] (2 x 1)
