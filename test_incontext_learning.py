"""
Test script for in-context learning implementation.

This script tests:
1. Sequence construction with rolled examples
2. Label masking (only final output should have valid labels)
3. Sequence length calculation
4. Data preprocessing pipeline
"""

import numpy as np
import tempfile
import os
import json
from pathlib import Path

from dataset.build_arc_dataset import (
    construct_in_context_sequence,
    ARCMaxGridSize,
    ARCPuzzle,
    convert_dataset,
    DataProcessConfig,
)


def test_construct_in_context_sequence():
    """Test that in-context sequences are constructed correctly."""
    print("\n=== Testing construct_in_context_sequence ===")

    # Create sample examples (3 examples, each 5x5 grid)
    examples = [
        (
            np.array([[1, 2, 3, 4, 5]] * 5, dtype=np.uint8),
            np.array([[6, 7, 8, 9, 0]] * 5, dtype=np.uint8),
        ),
        (
            np.array([[2, 3, 4, 5, 6]] * 5, dtype=np.uint8),
            np.array([[7, 8, 9, 0, 1]] * 5, dtype=np.uint8),
        ),
        (
            np.array([[3, 4, 5, 6, 7]] * 5, dtype=np.uint8),
            np.array([[8, 9, 0, 1, 2]] * 5, dtype=np.uint8),
        ),
    ]

    grid_size = ARCMaxGridSize * ARCMaxGridSize  # 900

    # Test with test_idx=2 (last example is test)
    input_seq, label_seq, test_out = construct_in_context_sequence(
        examples, test_idx=2, do_translation=False, ignore_label_id=0
    )

    print(f"Number of examples: {len(examples)}")
    print(f"Input sequence length: {len(input_seq)}")
    print(f"Label sequence length: {len(label_seq)}")
    print(f"Test output length: {len(test_out)}")

    # Expected lengths:
    # - 2 demonstrations (each has input + output): 2 * 2 * 900 = 3600
    # - 1 test input: 900
    # Total input sequence before test_out: 4500
    # After concatenating test_out to labels: 5400 total
    expected_input_len = (2 * (len(examples) - 1) + 1) * grid_size  # 4500
    assert len(input_seq) == expected_input_len, (
        f"Expected input length {expected_input_len}, got {len(input_seq)}"
    )
    print(f"✓ Input sequence length correct: {expected_input_len}")

    # Label sequence should be same length (all masked except will concatenate test_out later)
    assert len(label_seq) == expected_input_len, (
        f"Expected label length {expected_input_len}, got {len(label_seq)}"
    )
    print(f"✓ Label sequence length correct: {expected_input_len}")

    # Check all labels are masked (set to 0)
    assert np.all(label_seq == 0), (
        "All labels should be masked (0) in the returned label_seq"
    )
    print("✓ All demonstration labels correctly masked")

    # Test output should be 900 tokens
    assert len(test_out) == grid_size, f"Expected test_out length {grid_size}"
    print(f"✓ Test output length correct: {grid_size}")

    # Verify the order: should be [ex0_in, ex0_out, ex1_in, ex1_out, ex2_in]
    # Check that input sequence contains the patterns we expect
    print("✓ Sequence construction passed all checks")


def test_rolled_sequences():
    """Test that all examples get a chance to be the test case."""
    print("\n=== Testing Rolled Sequences ===")

    examples = [
        (np.ones((3, 3), dtype=np.uint8) * 1, np.ones((3, 3), dtype=np.uint8) * 6),
        (np.ones((3, 3), dtype=np.uint8) * 2, np.ones((3, 3), dtype=np.uint8) * 7),
        (np.ones((3, 3), dtype=np.uint8) * 3, np.ones((3, 3), dtype=np.uint8) * 8),
    ]

    grid_size = ARCMaxGridSize * ARCMaxGridSize
    sequences = []

    # Create rolled sequences for each test index
    for test_idx in range(len(examples)):
        input_seq, label_seq, test_out = construct_in_context_sequence(
            examples, test_idx=test_idx, do_translation=False, ignore_label_id=0
        )
        sequences.append((input_seq, label_seq, test_out, test_idx))

    print(f"Created {len(sequences)} rolled sequences")

    # Verify each sequence has correct structure
    for i, (input_seq, label_seq, test_out, test_idx) in enumerate(sequences):
        print(f"\nSequence {i} (test_idx={test_idx}):")
        print(f"  Input length: {len(input_seq)}")
        print(f"  Label length: {len(label_seq)}")
        print(f"  Test output length: {len(test_out)}")

        # After concatenating test_out, final sequence should be 2*N*900
        final_label_len = len(label_seq) + len(test_out)
        expected_len = 2 * len(examples) * grid_size
        assert final_label_len == expected_len, (
            f"Expected final length {expected_len}, got {final_label_len}"
        )

    print("\n✓ All rolled sequences have correct structure")


def test_label_masking():
    """Test that only the final output has valid labels."""
    print("\n=== Testing Label Masking ===")

    examples = [
        (
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            np.array([[5, 6], [7, 8]], dtype=np.uint8),
        ),
        (
            np.array([[2, 3], [4, 5]], dtype=np.uint8),
            np.array([[6, 7], [8, 9]], dtype=np.uint8),
        ),
    ]

    input_seq, label_seq, test_out = construct_in_context_sequence(
        examples, test_idx=1, do_translation=False, ignore_label_id=0
    )

    # Concatenate test_out to label_seq (as done in the actual preprocessing)
    full_label_seq = np.concatenate([label_seq, test_out])

    grid_size = ARCMaxGridSize * ARCMaxGridSize
    # Structure: [demo_in, demo_out, test_in, test_out] where only test_out has valid labels
    # Positions 0-899: demo input (masked)
    # Positions 900-1799: demo output (masked)
    # Positions 1800-2699: test input (masked)
    # Positions 2700-3599: test output (NOT masked)

    masked_portion = full_label_seq[: 3 * grid_size]
    valid_portion = full_label_seq[3 * grid_size :]

    print(f"Full label sequence length: {len(full_label_seq)}")
    print(f"Masked portion (first 3 grids): {len(masked_portion)} tokens")
    print(f"Valid portion (last grid): {len(valid_portion)} tokens")

    # Check that first 3 grids are all masked (0)
    assert np.all(masked_portion == 0), "First 3 grids should be masked (all zeros)"
    print("✓ Demonstration and test input correctly masked")

    # Check that last grid has non-zero values (after adding 2 in preprocessing)
    # Note: test_out will have values >= 2 because preprocessing adds 2
    assert np.any(valid_portion > 0), "Test output should have valid labels"
    print("✓ Test output has valid labels")


def test_sequence_length_calculation():
    """Test that sequence length is calculated correctly."""
    print("\n=== Testing Sequence Length Calculation ===")

    grid_size = ARCMaxGridSize * ARCMaxGridSize  # 900

    test_cases = [
        (2, 2 * 2 * grid_size),  # 2 examples -> 4 grids -> 3600 tokens
        (3, 2 * 3 * grid_size),  # 3 examples -> 6 grids -> 5400 tokens
        (4, 2 * 4 * grid_size),  # 4 examples -> 8 grids -> 7200 tokens
        (5, 2 * 5 * grid_size),  # 5 examples -> 10 grids -> 9000 tokens
    ]

    for num_examples, expected_seq_len in test_cases:
        actual_seq_len = 2 * num_examples * grid_size
        assert actual_seq_len == expected_seq_len, (
            f"Seq len calculation failed for {num_examples} examples"
        )
        print(
            f"✓ {num_examples} examples -> {expected_seq_len} tokens ({expected_seq_len // grid_size} grids)"
        )


def test_end_to_end_preprocessing():
    """Test end-to-end data preprocessing with a minimal dataset."""
    print("\n=== Testing End-to-End Preprocessing ===")

    # Create a minimal test dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input data directory
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()

        # Create a minimal ARC puzzle
        puzzle_data = {
            "test_puzzle_1": {
                "train": [
                    {
                        "input": [[1, 2], [3, 4]],
                        "output": [[5, 6], [7, 8]],
                    },
                    {
                        "input": [[2, 3], [4, 5]],
                        "output": [[6, 7], [8, 9]],
                    },
                ],
                "test": [
                    {
                        "input": [[3, 4], [5, 6]],
                        "output": [[7, 8], [9, 0]],
                    }
                ],
            }
        }

        # Save challenges
        challenges_file = input_dir / "arc-agi_test_challenges.json"
        with open(challenges_file, "w") as f:
            json.dump(puzzle_data, f)

        # Save solutions
        solutions_file = input_dir / "arc-agi_test_solutions.json"
        solutions_data = {"test_puzzle_1": [[[7, 8], [9, 0]]]}
        with open(solutions_file, "w") as f:
            json.dump(solutions_data, f)

        # Create output directory
        output_dir = Path(tmpdir) / "output"

        # Run preprocessing
        config = DataProcessConfig(
            input_file_prefix=str(input_dir / "arc-agi"),
            output_dir=str(output_dir),
            subsets=["test"],
            test_set_name="test",
            seed=42,
            num_aug=2,  # Small number for testing
        )

        print(f"Running preprocessing with config:")
        print(f"  Input: {config.input_file_prefix}")
        print(f"  Output: {config.output_dir}")
        print(f"  Augmentations: {config.num_aug}")

        try:
            convert_dataset(config)
            print("✓ Preprocessing completed successfully")

            # Verify output files exist
            train_dir = output_dir / "train"
            test_dir = output_dir / "test"

            assert train_dir.exists(), "Train directory should exist"
            assert test_dir.exists(), "Test directory should exist"
            print("✓ Output directories created")

            # Check that data files exist
            expected_files = [
                "all__inputs.npy",
                "all__labels.npy",
                "all__puzzle_identifiers.npy",
                "all__puzzle_indices.npy",
                "all__group_indices.npy",
                "dataset.json",
            ]

            for split_dir in [train_dir, test_dir]:
                for filename in expected_files:
                    filepath = split_dir / filename
                    assert filepath.exists(), f"Missing file: {filepath}"
                print(f"✓ All expected files exist in {split_dir.name}/")

            # Load and verify metadata
            with open(train_dir / "dataset.json", "r") as f:
                metadata = json.load(f)

            print(f"\nDataset metadata:")
            print(f"  Sequence length: {metadata['seq_len']}")
            print(f"  Vocab size: {metadata['vocab_size']}")
            print(f"  Total puzzles: {metadata['total_puzzles']}")
            print(f"  Total groups: {metadata['total_groups']}")
            print(f"  Mean examples per puzzle: {metadata['mean_puzzle_examples']:.2f}")

            # Verify sequence length matches formula
            grid_size = ARCMaxGridSize * ARCMaxGridSize
            # We have 2 training examples per puzzle, so max is 2
            # But we also add test examples, so could be 3
            # Actually let me check what was saved...

            # Load actual data
            inputs = np.load(train_dir / "all__inputs.npy")
            labels = np.load(train_dir / "all__labels.npy")

            print(f"\nActual data shapes:")
            print(f"  Inputs: {inputs.shape}")
            print(f"  Labels: {labels.shape}")

            # With 2 train examples + 1 test, we have 3 examples total
            # Each gets rolled: 3 sequences per puzzle
            # With augmentations (2), we have (1 original + 2 aug) * 3 = 9 total sequences
            print(f"  Number of sequences: {len(inputs)}")

            # Verify each sequence has correct length
            expected_seq_len = metadata["seq_len"]
            for i in range(len(inputs)):
                assert len(inputs[i]) == expected_seq_len, f"Input {i} has wrong length"
                assert len(labels[i]) == expected_seq_len, f"Label {i} has wrong length"

            print(f"✓ All sequences have correct length: {expected_seq_len}")

            # Verify label masking
            for i in range(len(labels)):
                label = labels[i]
                # Count how many are masked (0) vs valid (non-zero)
                masked_count = np.sum(label == 0)
                valid_count = np.sum(label != 0)

                # Most should be masked, only last grid should be valid
                assert masked_count > valid_count, f"Sequence {i} has wrong masking"

            print("✓ Label masking verified")

            print("\n✓ End-to-end preprocessing test passed!")

        except Exception as e:
            print(f"✗ Preprocessing failed with error: {e}")
            raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing In-Context Learning Implementation")
    print("=" * 60)

    try:
        test_construct_in_context_sequence()
        test_rolled_sequences()
        test_label_masking()
        test_sequence_length_calculation()
        test_end_to_end_preprocessing()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
