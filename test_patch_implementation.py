"""Test script to verify patch-based implementation."""

import torch
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)


def test_standard_mode():
    """Test the standard cell-based tokenization."""
    print("=" * 60)
    print("Testing Standard Mode (cell-based tokenization)")
    print("=" * 60)

    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=900,  # 30x30
        vocab_size=12,
        num_puzzle_identifiers=100,
        puzzle_emb_ndim=512,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        use_patches=False,
    )

    model = TinyRecursiveReasoningModel_ACTV1_Inner(config)

    # Create dummy input
    batch = {
        "inputs": torch.randint(0, 12, (2, 900)),  # [batch_size, seq_len]
        "puzzle_identifiers": torch.randint(0, 100, (2,)),
    }

    # Forward pass
    carry = model.empty_carry(batch_size=2)
    new_carry, output, (q_halt, q_continue) = model(carry, batch)

    print(f"Input shape: {batch['inputs'].shape}")
    print(f"Expected: [2, 900]")
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [2, 900, 12]")
    print(f"\nQ-halt shape: {q_halt.shape}")
    print(f"Expected: [2]")

    assert output.shape == (2, 900, 12), (
        f"Expected output shape [2, 900, 12], got {output.shape}"
    )
    print("\n✓ Standard mode test passed!")


def test_patch_mode():
    """Test the patch-based tokenization."""
    print("\n" + "=" * 60)
    print("Testing Patch Mode (3x3 patches)")
    print("=" * 60)

    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=900,  # Still 30x30 in data
        vocab_size=12,
        num_puzzle_identifiers=100,
        puzzle_emb_ndim=512,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        use_patches=True,
        patch_size=3,
        grid_height=30,
        grid_width=30,
        upsample_method="pixel_shuffle",
        rope_2d=True,
        rope_2d_method="mixed",
    )

    model = TinyRecursiveReasoningModel_ACTV1_Inner(config)

    # Create dummy input (same format as standard mode)
    batch = {
        "inputs": torch.randint(0, 12, (2, 900)),  # [batch_size, seq_len]
        "puzzle_identifiers": torch.randint(0, 100, (2,)),
    }

    # Forward pass
    carry = model.empty_carry(batch_size=2)
    print(f"\nCarry z_H shape: {carry.z_H.shape}")
    print(f"Expected: [2, 116, 512]  (100 patches + 16 puzzle_emb)")

    new_carry, output, (q_halt, q_continue) = model(carry, batch)

    print(f"\nInput shape: {batch['inputs'].shape}")
    print(f"Expected: [2, 900]")
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [2, 900, 12]  (upsampled from 100 patches)")
    print(f"\nQ-halt shape: {q_halt.shape}")
    print(f"Expected: [2]")

    assert output.shape == (2, 900, 12), (
        f"Expected output shape [2, 900, 12], got {output.shape}"
    )
    assert carry.z_H.shape[1] == 116, (
        f"Expected carry seq_len 116 (100 patches + 16 puzzle_emb), got {carry.z_H.shape[1]}"
    )
    print("\n✓ Patch mode test passed!")


def test_patch_mode_linear_upsample():
    """Test patch mode with linear upsampling."""
    print("\n" + "=" * 60)
    print("Testing Patch Mode with Linear Upsampling")
    print("=" * 60)

    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=900,
        vocab_size=12,
        num_puzzle_identifiers=100,
        puzzle_emb_ndim=512,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4.0,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        use_patches=True,
        patch_size=3,
        grid_height=30,
        grid_width=30,
        upsample_method="linear",
        rope_2d=True,
        rope_2d_method="mixed",
    )

    model = TinyRecursiveReasoningModel_ACTV1_Inner(config)

    batch = {
        "inputs": torch.randint(0, 12, (2, 900)),
        "puzzle_identifiers": torch.randint(0, 100, (2,)),
    }

    carry = model.empty_carry(batch_size=2)
    new_carry, output, (q_halt, q_continue) = model(carry, batch)

    print(f"\nInput shape: {batch['inputs'].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [2, 900, 12]")

    assert output.shape == (2, 900, 12), (
        f"Expected output shape [2, 900, 12], got {output.shape}"
    )
    print("\n✓ Linear upsampling test passed!")


def test_2d_rope():
    """Test 2D RoPE implementation."""
    print("\n" + "=" * 60)
    print("Testing 2D RoPE")
    print("=" * 60)

    from models.layers import RotaryEmbedding2D

    # Test mixed method
    rope_2d_mixed = RotaryEmbedding2D(
        dim=64,  # head_dim
        grid_height=10,
        grid_width=10,
        base=10000.0,
        method="mixed",
    )

    cos, sin = rope_2d_mixed()
    print(f"Mixed 2D RoPE cos shape: {cos.shape}")
    print(f"Expected: [100, 64]  (10x10 grid, head_dim=64)")
    assert cos.shape == (100, 64), f"Expected cos shape [100, 64], got {cos.shape}"
    assert sin.shape == (100, 64), f"Expected sin shape [100, 64], got {sin.shape}"

    # Test axial method
    rope_2d_axial = RotaryEmbedding2D(
        dim=64, grid_height=10, grid_width=10, base=10000.0, method="axial"
    )

    cos, sin = rope_2d_axial()
    print(f"\nAxial 2D RoPE cos shape: {cos.shape}")
    print(f"Expected: [100, 64]")
    assert cos.shape == (100, 64), f"Expected cos shape [100, 64], got {cos.shape}"

    print("\n✓ 2D RoPE test passed!")


if __name__ == "__main__":
    test_standard_mode()
    test_patch_mode()
    test_patch_mode_linear_upsample()
    test_2d_rope()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
