from unittest import TestCase
import torch

from ..attention import Attention
from .. import config


class TestAttention(TestCase):

    def test_shat(self):
        queries = torch.tensor([[[0., 0.], [0., 0.], [0., 0.], [1., 0.5]]])
        keys = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 0.]]])
        values = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])

        attn = Attention(config.SHAT)
        output = attn(queries, keys, values)
        torch.testing.assert_allclose(output[0, 0, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 1, :].tolist(), [.5, .5, 0.])
        torch.testing.assert_allclose(output[0, 2, :].tolist(), [1/3, 1/3, 1/3])
        torch.testing.assert_allclose(output[0, 3, :].tolist(), [0.3129638433456421, 0.3129638433456421, 0.21975961327552795])

    def test_ahat(self):
        queries = torch.tensor([[[0., 0.], [0., 0.], [0., 0.], [1., 0.5]]])
        keys = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 0.]]])
        values = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])

        attn = Attention(config.AHAT)
        output = attn(queries, keys, values)
        torch.testing.assert_allclose(output[0, 0, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 1, :].tolist(), [.5, .5, 0.])
        torch.testing.assert_allclose(output[0, 2, :].tolist(), [1/3, 1/3, 1/3])
        torch.testing.assert_allclose(output[0, 3, :].tolist(), [.5, .5, 0.])

    def test_luhat(self):
        queries = torch.tensor([[[0., 0.], [0., 0.], [0., 0.], [1., 0.5]]])
        keys = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 0.]]])
        values = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])

        attn = Attention(config.LUHAT)
        output = attn(queries, keys, values)
        torch.testing.assert_allclose(output[0, 0, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 1, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 2, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 3, :].tolist(), [1., 0., 0.])

    def test_ruhat(self):
        queries = torch.tensor([[[0., 0.], [0., 0.], [0., 0.], [1., 0.5]]])
        keys = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 0.]]])
        values = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])

        attn = Attention(config.RUHAT)
        output = attn(queries, keys, values)
        torch.testing.assert_allclose(output[0, 0, :].tolist(), [1., 0., 0.])
        torch.testing.assert_allclose(output[0, 1, :].tolist(), [0., 1., 0.])
        torch.testing.assert_allclose(output[0, 2, :].tolist(), [0., 0., 1.])
        torch.testing.assert_allclose(output[0, 3, :].tolist(), [0., 1., 0.])


if __name__ == '__main__':
    unittest.main()