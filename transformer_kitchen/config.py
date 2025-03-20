from typing import NamedTuple, Optional


class AttentionConfig(NamedTuple):
    temperature: float = 1.
    tie_breaking: Optional[str] = None
    masking: Optional[str] = "causal"


SHAT = AttentionConfig()
AHAT = AttentionConfig(temperature=0.)
LUHAT = AttentionConfig(temperature=0., tie_breaking="left")
RUHAT = AttentionConfig(temperature=0., tie_breaking="right")


class Config(NamedTuple):
    attention: AttentionConfig = SHAT
    d_model: int = 512
    d_key: int = 64
    d_value: int = 64