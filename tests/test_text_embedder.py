from unittest import TestCase

from src.text_embedder import TextEmbedder


class TestTextEmbedder(TestCase):

    def test_stoi(self):
        vocab = ["a", "b", "c"]
        a = TextEmbedder(vocab)
        self.assertEqual(
            a.stoi,
            {"a": 0, "b": 1, "c": 2},
        )

    def test_itos(self):
        vocab = ["a", "b", "c"]
        a = TextEmbedder(vocab)
        self.assertEqual(
            a.itos,
            {0: "a", 1: "b", 2: "c"},
        )

    def test_embed(self):
        text = "धाँ"
        e = TextEmbedder(["धाँ", "दि"])
        embedding = e.embed(text)
        self.assertEqual(len(embedding), 3)

        text = "धाँ दि"
        embedding = e.embed(text)
        self.assertEqual(len(embedding), 4)

    def test_unembed(self):
        e = TextEmbedder(["धाँ", "दि"])
        embedding = [0, 1, 1, 0]
        result = e.unembed(embedding)

        self.assertEqual(
            result,
            ["धाँ", "दि", "दि", "धाँ"],
        )
