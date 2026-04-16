"""Unit tests that require ``temporalio`` to be installed.

These are skipped when ``temporalio`` is missing so that the rest of the
suite stays lightweight.
"""

from __future__ import annotations

import pytest

temporalio = pytest.importorskip("temporalio")


@pytest.mark.unit
def test_plugin_is_simpleplugin_subclass():
    import temporalio.plugin

    from langfuse.temporal import PLUGIN_NAME, LangfusePlugin

    plugin = LangfusePlugin()
    assert isinstance(plugin, temporalio.plugin.SimplePlugin)
    assert plugin.name() == PLUGIN_NAME


@pytest.mark.unit
def test_plugin_registers_two_interceptors():
    """Temporal OTel interceptor + Langfuse enrichment interceptor."""
    from temporalio.converter import default as default_converter

    from langfuse.temporal import LangfusePlugin

    plugin = LangfusePlugin()
    # Use a freshly-constructed ClientConfig-like dict; SimplePlugin
    # mutates ``interceptors`` in-place via list extension, so we pass a
    # mutable list and inspect it after the call.
    config = {
        "service_client": None,
        "namespace": "default",
        "data_converter": default_converter(),
        "interceptors": [],
        "default_workflow_query_reject_condition": None,
        "plugins": [],
        "header_codec_behavior": None,
        "api_key": None,
    }
    new_config = plugin.configure_client(config)
    assert len(new_config["interceptors"]) == 2


@pytest.mark.unit
def test_constructor_accepts_config_object():
    from langfuse.temporal import LangfusePlugin, LangfusePluginConfig

    cfg = LangfusePluginConfig(environment="prod", release="v1")
    plugin = LangfusePlugin(config=cfg)
    assert plugin.config.environment == "prod"
    assert plugin.config.release == "v1"


@pytest.mark.unit
def test_constructor_accepts_kwargs_as_overrides():
    from langfuse.temporal import LangfusePlugin

    plugin = LangfusePlugin(environment="staging")
    assert plugin.config.environment == "staging"


@pytest.mark.unit
def test_constructor_rejects_unknown_kwargs():
    from langfuse.temporal import LangfusePlugin

    with pytest.raises(TypeError):
        LangfusePlugin(not_a_real_option=True)
