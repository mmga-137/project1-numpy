"""Small independent checks; these do not need the course checkpoint or images."""
import unittest

import numpy as np

from image_encoder import ImageEncoder, MultiHeadAttention


class EncoderArrayTests(unittest.TestCase):
    def test_patchify_rejects_wrong_shape_even_with_same_pixel_count(self):
        encoder = ImageEncoder.__new__(ImageEncoder)
        for shape in ((1, 3, 112, 448), (1, 224, 224, 3), (3, 224, 224)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, r'\[B,3,224,224\]'):
                    encoder.patchify(np.zeros(shape, dtype=np.float32))

    def test_patchify_rejects_wrong_dtype(self):
        encoder = ImageEncoder.__new__(ImageEncoder)
        for dtype in (np.float64, np.uint8):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(TypeError, 'float32'):
                    encoder.patchify(np.zeros((1, 3, 224, 224), dtype=dtype))

    def test_patchify_rejects_non_array_input(self):
        with self.assertRaisesRegex(TypeError, 'NumPy array'):
            ImageEncoder.__new__(ImageEncoder).patchify([1, 2, 3])

    def test_patch_order_matches_explicit_image_slices(self):
        # A non-contiguous input also catches assumptions about memory layout.
        storage = np.arange(2 * 3 * 224 * 448, dtype=np.float32).reshape(2, 3, 224, 448)
        images = storage[..., ::2]
        actual = ImageEncoder.__new__(ImageEncoder).patchify(images)
        expected = np.stack([
            np.stack([
                image[:, top:top + 32, left:left + 32].flatten()
                for top in range(0, 224, 32)
                for left in range(0, 224, 32)
            ])
            for image in images
        ])
        np.testing.assert_array_equal(actual, expected)

    def make_attention(self):
        rng = np.random.default_rng(24)
        weights = {
            'toy.in_proj_weight': rng.normal(0, 0.2, (24, 8)).astype(np.float32),
            'toy.in_proj_bias': rng.normal(0, 0.1, 24).astype(np.float32),
            'toy.out_proj.weight': rng.normal(0, 0.2, (8, 8)).astype(np.float32),
            'toy.out_proj.bias': rng.normal(0, 0.1, 8).astype(np.float32),
        }
        return MultiHeadAttention(weights, 'toy', heads=2)

    def test_head_split_preserves_each_token_feature_group(self):
        attention = self.make_attention()
        x = np.arange(3 * 5 * 8, dtype=np.float32).reshape(3, 5, 8)
        actual = attention.split_heads(x)
        self.assertEqual(actual.shape, (3, 2, 5, 4))
        for batch in range(3):
            for head in range(2):
                for token in range(5):
                    np.testing.assert_array_equal(
                        actual[batch, head, token], x[batch, token, head * 4:(head + 1) * 4]
                    )

    def test_attention_matches_independent_head_loop(self):
        attention = self.make_attention()
        x = np.random.default_rng(18).normal(size=(3, 5, 8)).astype(np.float32)
        qkv = x @ attention.qkv_weight.T + attention.qkv_bias
        for causal in (False, True):
            with self.subTest(causal=causal):
                expected = np.empty_like(x)
                for batch in range(3):
                    for token in range(5):
                        joined = []
                        for head in range(2):
                            start = head * 4
                            query = qkv[batch, token, start:start + 4]
                            # For the causal oracle, future rows do not enter the sum.
                            end = token + 1 if causal else 5
                            keys = qkv[batch, :end, 8 + start:12 + start]
                            values = qkv[batch, :end, 16 + start:20 + start]
                            scores = np.array([np.dot(query, key) / 2 for key in keys])
                            probabilities = np.exp(scores - scores.max())
                            probabilities /= probabilities.sum()
                            value = sum(p * row for p, row in zip(probabilities, values))
                            joined.extend(value)
                        expected[batch, token] = (
                            np.asarray(joined) @ attention.out_weight.T + attention.out_bias
                        )
                actual = attention(x, causal=causal)
                self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
