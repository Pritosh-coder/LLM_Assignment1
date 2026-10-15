"""Regression gates for FIX6 Part C evaluation batching/padding."""
from src.attack_extract import (
    DISCOVERABLE_BATCH_SIZE,
    UNTARGETED_GENERATION_BATCH_SIZE,
    TARGETED_BATCH_SIZE,
    LIKELIHOOD_BATCH_SIZE,
    _configure_generation_tokenizer,
)


def test_generation_tokenizer_uses_left_padding_and_eos_pad_fallback():
    class FakeTokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "<eos>"
        padding_side = "right"

    tok = FakeTokenizer()
    _configure_generation_tokenizer(tok)
    assert tok.pad_token == "<eos>"
    assert tok.padding_side == "left"


def test_fix6_batches_are_large_enough_to_avoid_fix5_runtime_regression():
    assert DISCOVERABLE_BATCH_SIZE >= 32
    assert UNTARGETED_GENERATION_BATCH_SIZE >= 32
    assert LIKELIHOOD_BATCH_SIZE >= 16
    assert TARGETED_BATCH_SIZE >= 16
