from unittest import TestCase

from src.text_embedder import TextEmbedder


class TestTextEmbedder(TestCase):

    def test_clean_text(self):
        inputs = [
            "a b c\na b c",
            "a b c\n\na b c",
            "a b c\n\n\na b c",
            "a b c\n\n\n\na b c",
        ]
        output = "a b c a b c"
        e = TextEmbedder("")
        for i in inputs:
            cleaned = e.clean_text(i)
            self.assertEqual(cleaned, output)

    def test_vocab_generation(self):
        file = "datasets/sample.dataset"
        e = TextEmbedder(file)
        vocab = e.vocabulary()
        self.assertEqual(len(vocab.keys()), 11)

    def test_embed(self):
        text = "धाँ"
        file = "datasets/sample.dataset"
        e = TextEmbedder(file)
        embedding = e.create_embeddings(text)
        self.assertEqual(len(embedding), 1)

        text = "धाँ दि"
        embedding = e.create_embeddings(text)
        self.assertEqual(len(embedding), 2)
