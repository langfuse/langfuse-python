import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "select_e2e_shard.py"


def load_shard_script():
    spec = importlib.util.spec_from_file_location("select_e2e_shard", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load shard selector from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_shards_cover_all_files_once():
    shard_script = load_shard_script()

    all_files = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests" / "e2e").glob("test_*.py")
    )

    shards, shard_loads = shard_script.assign_shards(
        shard_script.discover_e2e_files(), shard_count=2
    )

    assert len(shards) == 2
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted([path for shard in shards for path in shard]) == all_files
    assert all(load > 0 for load in shard_loads)


def test_unknown_file_weight_falls_back_to_test_count(tmp_path: Path):
    shard_script = load_shard_script()

    test_file = tmp_path / "test_future_suite.py"
    test_file.write_text(
        "\n".join(
            [
                "def test_one():",
                "    pass",
                "",
                "async def test_two():",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    assert shard_script.count_test_functions(test_file) == 2
    assert shard_script.estimate_weight(test_file) == 2
