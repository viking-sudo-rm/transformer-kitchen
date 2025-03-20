import torch

from .config import AttentionConfig


class Attention:
    def __init__(self, config: AttentionConfig):
        self.config = config

    def __call__(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute attention based on the specific config settings.
        
        Args:
            queries (batch_size, seq_len, d_key): The query tensor.
            keys (batch_size, seq_len, d_key): The key tensor.
            values (batch_size, seq_len, d_value): The value tensor.
        """
        _, seq_len, d_key = queries.size()

        scores = torch.einsum("bik,bjk->bij", queries, keys)
        scores *= 1 / (d_key ** 0.5)

        if self.config.masking == "causal":
            mask = (torch.arange(seq_len).unsqueeze(0) <= torch.arange(seq_len).unsqueeze(1))
            scores = torch.where(mask, scores, float("-inf") * torch.ones_like(scores))

        if self.config.temperature != 0.:
            scores = scores / self.config.temperature
        else:
            max_scores = scores.max(dim=-1, keepdim=True).values
            ones = torch.ones_like(scores)
            scores = torch.where(scores == max_scores, ones, float("-inf") * ones)

            if self.config.tie_breaking == "left":
                mask = (torch.cumsum(scores, dim=-1) == 1)
                scores = torch.where(mask, scores, float("-inf") * torch.ones_like(scores))
            elif self.config.tie_breaking == "right":
                activated = torch.isfinite(scores)
                n_activated = activated.int().sum(dim=-1, keepdim=True)
                activated *= (torch.cumsum(activated.int(), dim=-1) == n_activated)
                scores = torch.where(activated, scores, float("-inf") * torch.ones_like(scores))
        
        return torch.einsum("bij,bjk->bik", scores.softmax(dim=-1), values)
