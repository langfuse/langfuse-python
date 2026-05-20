import argparse
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
E2E_ROOT = REPO_ROOT / "tests" / "e2e"

# These weights keep the existing balance close to the observed runtime split,
# while new files automatically fall back to their local test count.
HISTORICAL_WEIGHTS = {
    "tests/e2e/test_batch_evaluation.py": 41,
    "tests/e2e/test_core_sdk.py": 53,
    "tests/e2e/test_datasets.py": 7,
    "tests/e2e/test_decorators.py": 32,
    "tests/e2e/test_experiments.py": 17,
    "tests/e2e/test_media.py": 1,
    "tests/e2e/test_prompt.py": 27,
}


def relative_test_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def discover_e2e_files() -> list[Path]:
    return sorted(E2E_ROOT.glob("test_*.py"))


def count_test_functions(path: Path) -> int:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def estimate_weight(path: Path) -> int:
    try:
        relative_path = relative_test_path(path)
    except ValueError:
        relative_path = None
    if relative_path is not None and relative_path in HISTORICAL_WEIGHTS:
        return HISTORICAL_WEIGHTS[relative_path]

    return max(count_test_functions(path), 1)


def assign_shards(
    paths: list[Path], shard_count: int
) -> tuple[list[list[str]], list[int]]:
    shard_loads = [0] * shard_count
    shards: list[list[str]] = [[] for _ in range(shard_count)]

    weighted_paths = sorted(
        ((estimate_weight(path), relative_test_path(path)) for path in paths),
        key=lambda item: (-item[0], item[1]),
    )

    for weight, relative_path in weighted_paths:
        shard_index = min(
            range(shard_count), key=lambda index: (shard_loads[index], index)
        )
        shards[shard_index].append(relative_path)
        shard_loads[shard_index] += weight

    return [sorted(shard) for shard in shards], shard_loads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the files for one e2e CI shard."
    )
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", default=2, type=int)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.shard_count < 1:
        raise SystemExit("--shard-count must be at least 1")

    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be within the configured shard count")

    shards, shard_loads = assign_shards(discover_e2e_files(), args.shard_count)
    selected_files = shards[args.shard_index]

    if args.json:
        print(
            json.dumps(
                {
                    "shard_count": args.shard_count,
                    "shard_index": args.shard_index,
                    "selected_files": selected_files,
                    "shard_loads": shard_loads,
                }
            )
        )
        return 0

    for path in selected_files:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
