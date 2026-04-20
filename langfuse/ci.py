"""CI/CD helpers for running Langfuse experiments in GitHub Actions.

Designed to be used in conjunction with the ``langfuse/experiment-action``
GitHub Action (https://github.com/langfuse/experiment-action). The action
constructs a :class:`RunnerContext` pre-populated with dataset, run name, and
GitHub-sourced metadata, then calls the user's ``experiment(context)``
function.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from langfuse.batch_evaluation import CompositeEvaluatorFunction
from langfuse.experiment import (
    EvaluatorFunction,
    ExperimentData,
    ExperimentResult,
    RunEvaluatorFunction,
    TaskFunction,
)

if TYPE_CHECKING:
    from langfuse._client.client import Langfuse


class RunnerContext:
    """Wraps :meth:`Langfuse.run_experiment` with CI-injected defaults.

    Intended for use with the ``langfuse/experiment-action`` GitHub Action
    (https://github.com/langfuse/experiment-action). The action builds a
    ``RunnerContext`` before invoking the user's ``experiment(context)``
    function. Defaults set here (dataset, name, run name, metadata tags) are
    applied when the user omits them on the :meth:`run_experiment` call;
    users can override any default by passing the corresponding argument
    explicitly.
    """

    def __init__(
        self,
        *,
        client: "Langfuse",
        data: Optional[ExperimentData] = None,
        dataset_version: Optional[datetime] = None,
        name: Optional[str] = None,
        run_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """Build a ``RunnerContext`` populated with defaults for ``run_experiment``.

        Typically called by the ``langfuse/experiment-action`` GitHub Action,
        not by end users directly. Every field except ``client`` is optional:
        fields left as ``None`` simply mean the corresponding argument must be
        supplied on the :meth:`run_experiment` call.

        Args:
            client: Initialized Langfuse SDK client used to execute the
                experiment. The action creates this from the
                ``langfuse_public_key`` / ``langfuse_secret_key`` /
                ``langfuse_base_url`` inputs.
            data: Default dataset items to run the experiment on. Accepts
                either ``List[LocalExperimentItem]`` or ``List[DatasetItem]``.
                Injected by the action when ``dataset_name`` is configured.
                If ``None``, the user must pass ``data=`` to
                :meth:`run_experiment`.
            dataset_version: Optional pinned dataset version. Injected by the
                action when ``dataset_version`` is configured.
            name: Default human-readable experiment name (e.g. the action's
                ``experiment_name`` input). If ``None``, the user must pass
                ``name=`` to :meth:`run_experiment`.
            run_name: Default exact run name. The action typically derives
                this from the commit SHA / PR number so that reruns produce
                distinct runs in Langfuse.
            metadata: Default metadata attached to every experiment trace and
                the dataset run. The action injects GitHub-sourced tags (SHA,
                PR link, workflow run link, branch, GH user, etc.). Merged
                with any ``metadata`` passed to :meth:`run_experiment`, with
                user-supplied keys winning on collision.
        """
        self.client = client
        self.data = data
        self.dataset_version = dataset_version
        self.name = name
        self.run_name = run_name
        self.metadata = metadata

    def run_experiment(
        self,
        *,
        name: Optional[str] = None,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        data: Optional[ExperimentData] = None,
        task: TaskFunction,
        evaluators: List[EvaluatorFunction] = [],
        composite_evaluator: Optional[CompositeEvaluatorFunction] = None,
        run_evaluators: List[RunEvaluatorFunction] = [],
        max_concurrency: int = 50,
        metadata: Optional[Dict[str, str]] = None,
        _dataset_version: Optional[datetime] = None,
    ) -> ExperimentResult:
        resolved_name = name if name is not None else self.name
        if resolved_name is None:
            raise ValueError(
                "`name` must be provided either on the RunnerContext or the run_experiment call"
            )

        resolved_data = data if data is not None else self.data
        if resolved_data is None:
            raise ValueError(
                "`data` must be provided either on the RunnerContext or the run_experiment call"
            )

        resolved_run_name = run_name if run_name is not None else self.run_name
        resolved_dataset_version = (
            _dataset_version if _dataset_version is not None else self.dataset_version
        )

        merged_metadata: Optional[Dict[str, str]]
        if self.metadata is None and metadata is None:
            merged_metadata = None
        else:
            merged_metadata = {**(self.metadata or {}), **(metadata or {})}

        return self.client.run_experiment(
            name=resolved_name,
            run_name=resolved_run_name,
            description=description,
            data=resolved_data,
            task=task,
            evaluators=evaluators,
            composite_evaluator=composite_evaluator,
            run_evaluators=run_evaluators,
            max_concurrency=max_concurrency,
            metadata=merged_metadata,
            _dataset_version=resolved_dataset_version,
        )


class RegressionError(Exception):
    """Raised by a user's ``experiment`` function to signal a CI gate failure.

    The GitHub action catches this exception and, when ``should_fail_on_error``
    is enabled, fails the workflow run and renders a callout in the PR comment
    using ``metric``/``value``/``threshold`` if supplied, otherwise ``str(exc)``.
    """

    def __init__(
        self,
        *,
        result: ExperimentResult,
        metric: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        message: Optional[str] = None,
    ):
        self.result = result
        self.metric = metric
        self.value = value
        self.threshold = threshold
        if message is not None:
            formatted = message
        elif metric is not None:
            formatted = f"Regression on `{metric}`: {value} (threshold {threshold})"
        else:
            formatted = "Experiment regression detected"
        super().__init__(formatted)
