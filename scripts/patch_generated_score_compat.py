#!/usr/bin/env python3
"""Preserve the legacy score-create API after Fern moves it to ``scores``.

Fern owns ``langfuse/api`` and regenerates it from the main Langfuse repository.
This postprocessor is intentionally narrow and fail-closed: it only patches the
known generated shape where ``scores.create`` exists and
``legacy.score_v1.create`` no longer does.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

CLIENT_IMPORTS = """\
from ...commons.types.create_score_value import CreateScoreValue
from ...commons.types.score_data_type import ScoreDataType
from ...scores.client import (
    AsyncScoresClient as CanonicalAsyncScoresClient,
    ScoresClient as CanonicalScoresClient,
    OMIT,
)
from ...scores.types.create_score_response import CreateScoreResponse
from ...scores.types.create_score_source import CreateScoreSource
"""

SYNC_CREATE_METHOD = '''\
    def create(
        self,
        *,
        name: str,
        value: CreateScoreValue,
        id: typing.Optional[str] = OMIT,
        trace_id: typing.Optional[str] = OMIT,
        session_id: typing.Optional[str] = OMIT,
        observation_id: typing.Optional[str] = OMIT,
        dataset_run_id: typing.Optional[str] = OMIT,
        comment: typing.Optional[str] = OMIT,
        metadata: typing.Optional[typing.Dict[str, typing.Any]] = OMIT,
        environment: typing.Optional[str] = OMIT,
        queue_id: typing.Optional[str] = OMIT,
        data_type: typing.Optional[ScoreDataType] = OMIT,
        config_id: typing.Optional[str] = OMIT,
        source: typing.Optional[CreateScoreSource] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateScoreResponse:
        """**Deprecated compatibility alias.** Use ``client.scores.create``."""
        return CanonicalScoresClient(
            client_wrapper=self._raw_client._client_wrapper
        ).create(
            name=name,
            value=value,
            id=id,
            trace_id=trace_id,
            session_id=session_id,
            observation_id=observation_id,
            dataset_run_id=dataset_run_id,
            comment=comment,
            metadata=metadata,
            environment=environment,
            queue_id=queue_id,
            data_type=data_type,
            config_id=config_id,
            source=source,
            request_options=request_options,
        )

'''

ASYNC_CREATE_METHOD = '''\
    async def create(
        self,
        *,
        name: str,
        value: CreateScoreValue,
        id: typing.Optional[str] = OMIT,
        trace_id: typing.Optional[str] = OMIT,
        session_id: typing.Optional[str] = OMIT,
        observation_id: typing.Optional[str] = OMIT,
        dataset_run_id: typing.Optional[str] = OMIT,
        comment: typing.Optional[str] = OMIT,
        metadata: typing.Optional[typing.Dict[str, typing.Any]] = OMIT,
        environment: typing.Optional[str] = OMIT,
        queue_id: typing.Optional[str] = OMIT,
        data_type: typing.Optional[ScoreDataType] = OMIT,
        config_id: typing.Optional[str] = OMIT,
        source: typing.Optional[CreateScoreSource] = OMIT,
        request_options: typing.Optional[RequestOptions] = None,
    ) -> CreateScoreResponse:
        """**Deprecated compatibility alias.** Use ``client.scores.create``."""
        return await CanonicalAsyncScoresClient(
            client_wrapper=self._raw_client._client_wrapper
        ).create(
            name=name,
            value=value,
            id=id,
            trace_id=trace_id,
            session_id=session_id,
            observation_id=observation_id,
            dataset_run_id=dataset_run_id,
            comment=comment,
            metadata=metadata,
            environment=environment,
            queue_id=queue_id,
            data_type=data_type,
            config_id=config_id,
            source=source,
            request_options=request_options,
        )

'''

TYPE_NAMES = (
    "CreateScoreRequest",
    "CreateScoreResponse",
    "CreateScoreSource",
)
LEGACY_ALIAS_MARKER = (
    "**Deprecated compatibility alias.** Use ``client.scores.create``."
)


def _replace_once(contents: str, marker: str, replacement: str, *, path: Path) -> str:
    count = contents.count(marker)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one compatibility marker in {path}: {marker!r}; "
            f"found {count}"
        )
    return contents.replace(marker, replacement, 1)


def _append_type_exports(path: Path, imports_by_type: dict[str, str]) -> None:
    exports_marker = "# Score-create compatibility aliases (LFE-10397)."
    contents = path.read_text()
    if exports_marker in contents:
        return

    imports = "\n".join(
        f"from {module_name} import {type_name}"
        for type_name, module_name in imports_by_type.items()
    )
    names = ", ".join(repr(type_name) for type_name in TYPE_NAMES)
    defines_all = any(
        (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        )
        for node in ast.parse(contents).body
    )
    exports = (
        f"__all__ = [*__all__, {names}]" if defines_all else f"__all__ = [{names}]"
    )
    path.write_text(f"{contents.rstrip()}\n\n{exports_marker}\n{imports}\n{exports}\n")


def _create_type_aliases(types_dir: Path) -> None:
    types_dir.mkdir(parents=True, exist_ok=True)
    for type_name in TYPE_NAMES:
        module_name = "".join(
            f"_{character.lower()}" if character.isupper() else character
            for character in type_name
        ).lstrip("_")
        alias_path = types_dir / f"{module_name}.py"
        alias_path.write_text(
            "# This compatibility alias is applied after Fern generation.\n\n"
            f"from ....scores.types.{module_name} import {type_name}\n\n"
            f'__all__ = ["{type_name}"]\n'
        )


def patch_generated_api(api_root: Path) -> bool:
    canonical_client_path = api_root / "scores" / "client.py"
    legacy_client_path = api_root / "legacy" / "score_v1" / "client.py"

    canonical_client = canonical_client_path.read_text()
    legacy_client = legacy_client_path.read_text()

    canonical_has_create = (
        "\n    def create(" in canonical_client
        and "\n    async def create(" in canonical_client
    )
    legacy_has_create = (
        "\n    def create(" in legacy_client
        and "\n    async def create(" in legacy_client
    )

    if legacy_has_create:
        if LEGACY_ALIAS_MARKER in legacy_client and canonical_has_create:
            return False
        if canonical_has_create:
            raise RuntimeError(
                "Both canonical and legacy generated score clients define create; "
                "refusing to introduce a third compatibility surface"
            )
        return False

    if not canonical_has_create:
        raise RuntimeError(
            "Neither generated score client defines both sync and async create methods"
        )

    legacy_client = _replace_once(
        legacy_client,
        "from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper\n",
        "from ...core.client_wrapper import AsyncClientWrapper, SyncClientWrapper\n"
        f"{CLIENT_IMPORTS}",
        path=legacy_client_path,
    )
    legacy_client = _replace_once(
        legacy_client,
        "\n    def delete(",
        f"\n{SYNC_CREATE_METHOD}    def delete(",
        path=legacy_client_path,
    )
    legacy_client = _replace_once(
        legacy_client,
        "\n    async def delete(",
        f"\n{ASYNC_CREATE_METHOD}    async def delete(",
        path=legacy_client_path,
    )
    legacy_client_path.write_text(legacy_client)

    legacy_package_path = api_root / "legacy" / "score_v1" / "__init__.py"
    legacy_parent_package_path = api_root / "legacy" / "__init__.py"
    legacy_types_path = api_root / "legacy" / "score_v1" / "types" / "__init__.py"
    _create_type_aliases(legacy_types_path.parent)

    if not legacy_types_path.exists():
        legacy_types_path.write_text(
            "# This package is generated by Fern and extended for compatibility.\n\n"
            "__all__: list[str] = []\n"
        )

    _append_type_exports(
        legacy_types_path,
        {
            "CreateScoreRequest": ".create_score_request",
            "CreateScoreResponse": ".create_score_response",
            "CreateScoreSource": ".create_score_source",
        },
    )
    _append_type_exports(
        legacy_package_path, {type_name: ".types" for type_name in TYPE_NAMES}
    )
    _append_type_exports(
        legacy_parent_package_path,
        {type_name: ".score_v1" for type_name in TYPE_NAMES},
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-root",
        type=Path,
        default=Path("langfuse/api"),
        help="Path to the freshly generated Fern Python package",
    )
    args = parser.parse_args()
    changed = patch_generated_api(args.api_root)
    print(
        "Patched legacy score create compatibility alias."
        if changed
        else "No patch needed."
    )


if __name__ == "__main__":
    main()
