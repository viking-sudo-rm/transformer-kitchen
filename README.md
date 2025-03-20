# transformer-kitchen

Lightweight library to implement, debug, and replicate transformer constructions from the transformer cookbook.

Example usage to test an averaging-hard-attention-transformer (AHAT) attention head:

```python
from transformer_kitchen import Attention, AHAT

queries = torch.tensor([[[0., 0.], [0., 0.], [0., 0.], [1., 0.5]]])
keys = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 0.]]])
values = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]])

attn = Attention(config=AHAT)
output = attn(queries, keys, values)
```

If you want to specify your own custom kind of attention, you can do it as follows:

```python
attn = Attention(AttentionConfig(temperature=1.23, masking="strict-causal"))
```