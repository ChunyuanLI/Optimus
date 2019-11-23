import math
import torch
import torch.nn as nn

from ..utils import log_sum_exp

class EncoderBase(nn.Module):
    """docstring for EncoderBase"""
    def __init__(self):
        super(EncoderBase, self).__init__()

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)
        Returns: the tensors required to parameterize a distribution.
        E.g. for Gaussian encoder it returns the mean and variance tensors
        """

        raise NotImplementedError

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        raise NotImplementedError

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        raise NotImplementedError


    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        raise NotImplementedError

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """

        raise NotImplementedError