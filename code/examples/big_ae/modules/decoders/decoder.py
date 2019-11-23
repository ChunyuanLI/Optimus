import torch
import torch.nn as nn


class DecoderBase(nn.Module):
    """docstring for Decoder"""
    def __init__(self):
        super(DecoderBase, self).__init__()


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def decode(self, x, z):
        """
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns: Tensor1
            Tensor1: the output logits with size (batch_size * n_sample, seq_len, vocab_size)
        """

        raise NotImplementedError

    def reconstruct_error(self, x, z):
        """reconstruction loss
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        raise NotImplementedError

    def beam_search_decode(self, z, K):
        """beam search decoding
        Args:
            z: (batch_size, nz)
            K: the beam size
        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def sample_decode(self, z):
        """sampling from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def greedy_decode(self, z):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def log_probability(self, x, z):
        """
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        raise NotImplementedError