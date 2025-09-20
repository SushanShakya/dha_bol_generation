import torch
import torch.nn.functional as F


class NN:
    def __init__(self, vocab):
        self.vocab = vocab
        self.weights = self.init_weights()

    def init_weights(self):
        d = len(self.vocab) + 1
        return torch.randn((d, d), requires_grad=True)

    def compute(self, inputs):
        l = len(self.vocab) + 1
        inputs = list(map(lambda a: l - 1 if a == -1 else a, inputs))
        inputs = F.one_hot(torch.tensor(inputs), l).float()
        logits = inputs @ self.weights
        counts = logits.exp()
        return counts / counts.sum(1, keepdim=True)

    def gradient_descent(self, inputs, expected_outputs):
        prob = self.compute(inputs)
        predicted_probs = prob[range(len(expected_outputs)), expected_outputs]
        log_likelyhood = predicted_probs.log()
        neg_log_likelyhood = -(log_likelyhood).mean()
        # + 0.01 * (self.weights**2).mean()

        self.weights.grad = None
        neg_log_likelyhood.backward()

        self.weights.data += -0.1 * self.weights.grad

        return neg_log_likelyhood

    def random_walk(self):
        start = [-1]
        result = []

        for _ in range(10):
            prob = self.compute(start)
            print(start)
            print(prob)
            prediction = torch.multinomial(prob, num_samples=1, replacement=True).item()
            if prediction == -1 or prediction == 3:
                break
            result.append(prediction)
            start = [prediction]

        return result
