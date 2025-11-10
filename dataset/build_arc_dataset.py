from typing import List, Tuple, Dict
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import (
    PuzzleDatasetMetadata,
    dihedral_transform,
    inverse_dihedral_transform,
)


cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]
    test_set_name: str
    test_set_name2: str = "your_test_set"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1  # start > 1 to handle multiple datasets


ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"


@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]


def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(
    inp: np.ndarray, out: np.ndarray, do_translation: bool
):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(
            0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1
        )
        pad_c = np.random.randint(
            0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1
        )
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            (
                (pad_r, ARCMaxGridSize - pad_r - nrow),
                (pad_c, ARCMaxGridSize - pad_c - ncol),
            ),
            constant_values=0,
        )

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def construct_in_context_sequence(
    examples: list[tuple[np.ndarray, np.ndarray]],
    test_idx: int,
    do_translation: bool,
    ignore_label_id: int,
):
    """
    Construct in-context learning sequence where demonstrations come before test example.

    Args:
        examples: List of (input, output) pairs
        test_idx: Index of example to use as test (others become demonstrations)
        do_translation: Whether to apply translational augmentation
        ignore_label_id: Label ID to use for masking

    Returns:
        (input_sequence, label_sequence) where:
        - input_sequence contains: [demo1_in, demo1_out, demo2_in, demo2_out, ..., test_in]
        - label_sequence contains: [ignore, ignore, ignore, ignore, ..., test_out]
    """
    # Determine demonstration order - all examples except test_idx, in order
    demo_indices = list(range(len(examples)))
    demo_indices.pop(test_idx)  # Remove test example

    # Construct sequence
    input_parts = []
    label_parts = []

    # Add demonstration examples (input + output pairs)
    for idx in demo_indices:
        inp, out = examples[idx]
        inp_seq, out_seq = np_grid_to_seq_translational_augment(
            inp, out, do_translation
        )

        # Add to input sequence
        input_parts.append(inp_seq)
        input_parts.append(out_seq)

        # Add to label sequence (all masked as ignore)
        label_parts.append(np.full_like(inp_seq, ignore_label_id))
        label_parts.append(np.full_like(out_seq, ignore_label_id))

    # Add test example (input only, output is the target)
    test_inp, test_out = examples[test_idx]
    test_inp_seq, test_out_seq = np_grid_to_seq_translational_augment(
        test_inp, test_out, do_translation
    )

    input_parts.append(test_inp_seq)
    label_parts.append(
        np.full_like(test_inp_seq, ignore_label_id)
    )  # Test input is also masked

    # Concatenate all parts
    input_sequence = np.concatenate(input_parts)
    label_sequence = np.concatenate(label_parts)

    return input_sequence, label_sequence, test_out_seq


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")

    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate(
        [
            np.arange(0, 1, dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8)),
        ]
    )  # Permute colors, Excluding "0" (black)

    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    dest_mapping: Dict[str, Tuple[str, str]],
):
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        # Map to target split
        dest = dest_mapping[example_type]
        converted[dest].examples.extend(
            [
                (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
                for example in examples
            ]
        )

    group = [converted]

    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Check duplicate
            augmented = {
                dest: ARCPuzzle(
                    aug_name,
                    [
                        (_map_grid(input), _map_grid(label))
                        for (input, label) in puzzle.examples
                    ],
                )
                for dest, puzzle in converted.items()
            }
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        config.test_set_name2: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }

    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(
            f"{config.input_file_prefix}_{subset_name}_challenges.json", "r"
        ) as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)

                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print(f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)

        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(
                subset_name, test_examples_map["_default"]
            ):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle

            convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                {"train": train_examples_dest, "test": test_examples_dest},
            )
            total_puzzles += 1

    print(f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # Read dataset
    data, test_puzzles = load_puzzles_arcagi(config)

    # Calculate maximum number of examples per puzzle for sequence length
    max_examples_per_puzzle = 0
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    max_examples_per_puzzle = max(
                        max_examples_per_puzzle, len(puzzle.examples)
                    )
    print(f"Maximum examples per puzzle: {max_examples_per_puzzle}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0

        for subset_name, subset in split.items():  # "all" is the only subset
            # Construct subset
            results = {
                k: []
                for k in [
                    "inputs",
                    "labels",
                    "puzzle_identifiers",
                    "puzzle_indices",
                    "group_indices",
                ]
            }
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    # Create rolled sequences: each example becomes a test case
                    # For puzzle with examples [A, B, C], create:
                    # - [A_in, A_out, B_in, B_out, C_in] -> C_out
                    # - [C_in, C_out, A_in, A_out, B_in] -> B_out
                    # - [B_in, B_out, C_in, C_out, A_in] -> A_out

                    for test_idx in range(len(puzzle.examples)):
                        # Construct in-context sequence
                        input_seq, label_seq, test_out = construct_in_context_sequence(
                            puzzle.examples,
                            test_idx,
                            do_translation=enable_translational_augment,
                            ignore_label_id=0,  # Will be used for masking
                        )

                        # Concatenate test output to labels (this is what we predict)
                        label_seq = np.concatenate([label_seq, test_out])

                        results["inputs"].append(input_seq)
                        results["labels"].append(label_seq)
                        example_id += 1
                        total_examples += 1

                    # Add puzzle identifier (always 0 since we're not using puzzle IDs)
                    results["puzzle_identifiers"].append(0)
                    results["puzzle_indices"].append(example_id)
                    puzzle_id += 1
                    total_puzzles += 1

                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)

                np.save(
                    os.path.join(
                        config.output_dir, split_name, f"{subset_name}__{k}.npy"
                    ),
                    v,
                )

        # Metadata
        # Sequence length: 2*N * grid_size for N examples
        # where we have (N-1) demo pairs + 1 test pair (input + output), each grid is 900 tokens
        seq_len = 2 * max_examples_per_puzzle * ARCMaxGridSize * ARCMaxGridSize

        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,  # No longer using puzzle IDs
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys()),
        )

        # Save metadata as JSON.
        with open(
            os.path.join(config.output_dir, split_name, "dataset.json"), "w"
        ) as f:
            json.dump(metadata.model_dump(), f)

    # Save minimal IDs mapping (no longer using puzzle IDs)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
