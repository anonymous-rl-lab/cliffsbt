"""Experimental domain-calibrated warning workflow.

The module is intentionally not re-exported from :mod:`sbt_monitor`.  It ships no
pretrained coefficients, no universal event threshold and no safety policy.
Install the optional dependency with ``pip install sbt-monitor[warning]``.
"""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from ..schema import LeakageError, OptionalDependencyError, WarningEpisode, WarningEvaluation

try:  # optional dependency boundary
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover - environment dependent
    raise OptionalDependencyError(
        "experimental warning calibration requires scikit-learn; "
        "install with: pip install sbt-monitor[warning]"
    ) from exc


def _package_version() -> str:
    try:
        return version("sbt-monitor")
    except PackageNotFoundError:
        return "0.1.0"



@dataclass(frozen=True)
class _ScoreModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    selected_c: float
    calibration_hash: str
    calibration_episode_ids: tuple[str, ...]
    calibration_identity_set_ids: tuple[str, ...]

    def scores(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("states shape does not match fitted feature schema")
        z = (values - self.mean) / self.scale
        logits = z @ self.coefficients + self.intercept
        logits = np.clip(logits, -700, 700)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class FrozenReadout:
    """Serialized domain-calibrated score model and alarm threshold."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float
    horizon: int
    false_alarm_budget: float
    selected_c: float
    event_rule_name: str
    model_scope_fingerprint: str
    package_version: str
    calibration_hash: str
    calibration_episode_ids: tuple[str, ...]
    calibration_identity_set_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    def score(self, state: Sequence[float] | np.ndarray) -> float:
        arr = np.asarray(state, dtype=float)
        if arr.ndim != 1 or arr.size != len(self.feature_names):
            raise ValueError("state must be one-dimensional and match feature_names")
        return float(self.score_many(arr[None, :])[0])

    def score_many(self, states: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        model = _ScoreModel(
            self.feature_names,
            np.asarray(self.mean, dtype=float),
            np.asarray(self.scale, dtype=float),
            np.asarray(self.coefficients, dtype=float),
            float(self.intercept),
            float(self.selected_c),
            self.calibration_hash,
            self.calibration_episode_ids,
            self.calibration_identity_set_ids,
        )
        return model.scores(np.asarray(states, dtype=float))

    def alarm(self, state: Sequence[float] | np.ndarray) -> bool:
        return bool(self.score(state) >= self.threshold)

    def evaluate(
        self,
        episodes: Sequence[WarningEpisode],
        *,
        allow_overlap: bool = False,
    ) -> WarningEvaluation:
        _validate_episode_schema(episodes, self.feature_names)
        if not allow_overlap:
            episode_overlap = set(self.calibration_episode_ids) & {ep.episode_id for ep in episodes}
            identity_overlap = set(self.calibration_identity_set_ids) & {
                ep.identity_set_id for ep in episodes if ep.identity_set_id is not None
            }
            if episode_overlap or identity_overlap:
                raise LeakageError(
                    "evaluation overlaps calibration; "
                    f"episode_overlap={sorted(episode_overlap)!r}, "
                    f"identity_set_overlap={sorted(identity_overlap)!r}"
                )
        return _evaluate_scores(
            episodes,
            {ep.episode_id: self.score_many(ep.states) for ep in episodes},
            threshold=self.threshold,
            horizon=self.horizon,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scientific_object": "domain_calibrated_outcome_blind_warning_readout",
            "feature_names": list(self.feature_names),
            "mean": np.asarray(self.mean).tolist(),
            "scale": np.asarray(self.scale).tolist(),
            "coefficients": np.asarray(self.coefficients).tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "horizon": self.horizon,
            "false_alarm_budget": self.false_alarm_budget,
            "selected_c": self.selected_c,
            "event_rule_name": self.event_rule_name,
            "model_scope_fingerprint": self.model_scope_fingerprint,
            "package_version": self.package_version,
            "calibration_hash": self.calibration_hash,
            "calibration_episode_ids": list(self.calibration_episode_ids),
            "calibration_identity_set_ids": list(self.calibration_identity_set_ids),
            "metadata": dict(self.metadata),
            "scope_warnings": [
                "not a universal alarm",
                "requires domain-labelled calibration",
                "operational boundary and event labels are user supplied",
                "monitoring output only; no automatic intervention authorization",
            ],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FrozenReadout":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported warning-readout schema version")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            coefficients=np.asarray(payload["coefficients"], dtype=float),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
            horizon=int(payload["horizon"]),
            false_alarm_budget=float(payload["false_alarm_budget"]),
            selected_c=float(payload["selected_c"]),
            event_rule_name=str(payload["event_rule_name"]),
            model_scope_fingerprint=str(payload["model_scope_fingerprint"]),
            package_version=str(payload.get("package_version", "unknown")),
            calibration_hash=str(payload["calibration_hash"]),
            calibration_episode_ids=tuple(payload["calibration_episode_ids"]),
            calibration_identity_set_ids=tuple(payload["calibration_identity_set_ids"]),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class CalibrationResult:
    readout: FrozenReadout
    calibration_evaluation: WarningEvaluation
    selected_c: float


class WarningCalibrator:
    """Fit a user's own outcome-blind warning readout on labelled calibration episodes."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        horizon: int,
        false_alarm_budget: float,
        event_rule_name: str,
        model_scope_fingerprint: str,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        cv: str = "leave-one-group-out",
        random_state: int = 0,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not (0 <= false_alarm_budget <= 1):
            raise ValueError("false_alarm_budget must be in [0, 1]")
        if not event_rule_name.strip():
            raise ValueError("event_rule_name cannot be empty")
        if not model_scope_fingerprint.strip():
            raise ValueError("model_scope_fingerprint cannot be empty")
        if not c_grid or any(c <= 0 for c in c_grid):
            raise ValueError("c_grid must contain positive values")
        if cv != "leave-one-group-out":
            raise ValueError("v0.1 supports cv='leave-one-group-out' only")
        self.feature_names = tuple(feature_names)
        self.horizon = int(horizon)
        self.false_alarm_budget = float(false_alarm_budget)
        self.event_rule_name = event_rule_name.strip()
        self.model_scope_fingerprint = model_scope_fingerprint.strip()
        self.c_grid = tuple(float(c) for c in c_grid)
        self.cv = cv
        self.random_state = int(random_state)

    def fit(self, episodes: Sequence[WarningEpisode]) -> CalibrationResult:
        _validate_episode_schema(episodes, self.feature_names)
        if not any(not ep.is_control for ep in episodes) or not any(ep.is_control for ep in episodes):
            raise ValueError("calibration requires at least one event and one control episode")
        model = self._fit_score_model(episodes)
        score_map = {ep.episode_id: model.scores(ep.states) for ep in episodes}
        threshold = _select_threshold(
            episodes,
            score_map,
            horizon=self.horizon,
            false_alarm_budget=self.false_alarm_budget,
        )
        readout = FrozenReadout(
            feature_names=model.feature_names,
            mean=model.mean,
            scale=model.scale,
            coefficients=model.coefficients,
            intercept=model.intercept,
            threshold=threshold,
            horizon=self.horizon,
            false_alarm_budget=self.false_alarm_budget,
            selected_c=model.selected_c,
            event_rule_name=self.event_rule_name,
            model_scope_fingerprint=self.model_scope_fingerprint,
            package_version=_package_version(),
            calibration_hash=model.calibration_hash,
            calibration_episode_ids=model.calibration_episode_ids,
            calibration_identity_set_ids=model.calibration_identity_set_ids,
            metadata={
                "status": "experimental-v0.1",
                "cv": self.cv,
                "threshold_selection": "maximize timely episodes subject to calibration control false-alarm budget",
                "confirmation_labels_used_for_fit": False,
                "event_rule_name": self.event_rule_name,
                "model_scope_fingerprint": self.model_scope_fingerprint,
            },
        )
        evaluation = readout.evaluate(episodes, allow_overlap=True)
        return CalibrationResult(readout, evaluation, model.selected_c)

    def _fit_score_model(self, episodes: Sequence[WarningEpisode]) -> _ScoreModel:
        X, y, row_groups = _window_training_rows(episodes, self.horizon)
        selected_c = _select_c(X, y, row_groups, self.c_grid, self.random_state)
        scaler = StandardScaler().fit(X)
        z = scaler.transform(X)
        model = LogisticRegression(
            C=selected_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=self.random_state,
        ).fit(z, y)
        scale = np.asarray(scaler.scale_, dtype=float)
        scale[scale == 0] = 1.0
        return _ScoreModel(
            feature_names=self.feature_names,
            mean=np.asarray(scaler.mean_, dtype=float),
            scale=scale,
            coefficients=np.asarray(model.coef_[0], dtype=float),
            intercept=float(model.intercept_[0]),
            selected_c=selected_c,
            calibration_hash=_episodes_hash(episodes),
            calibration_episode_ids=tuple(ep.episode_id for ep in episodes),
            calibration_identity_set_ids=tuple(
                sorted({ep.identity_set_id for ep in episodes if ep.identity_set_id is not None})
            ),
        )


def nested_state_ablation(
    *,
    calibration_episodes: Sequence[WarningEpisode],
    evaluation_episodes: Sequence[WarningEpisode],
    feature_sets: Mapping[str, Sequence[str]],
    horizon: int,
    false_alarm_budget: float,
    event_rule_name: str,
    model_scope_fingerprint: str,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> dict[str, dict[str, Any]]:
    """Independently calibrate declared nested feature sets under one budget."""

    if not feature_sets:
        raise ValueError("feature_sets cannot be empty")
    full_names = calibration_episodes[0].feature_names
    if any(ep.feature_names != full_names for ep in tuple(calibration_episodes) + tuple(evaluation_episodes)):
        raise ValueError("all episodes must share the same full feature schema")
    index = {name: i for i, name in enumerate(full_names)}
    results: dict[str, dict[str, Any]] = {}
    for label, names_seq in feature_sets.items():
        names = tuple(names_seq)
        missing = [name for name in names if name not in index]
        if missing:
            raise KeyError(f"feature set {label!r} contains unknown features {missing!r}")
        columns = [index[name] for name in names]
        cal_subset = [_subset_episode(ep, names, columns) for ep in calibration_episodes]
        eval_subset = [_subset_episode(ep, names, columns) for ep in evaluation_episodes]
        result = WarningCalibrator(
            feature_names=names,
            horizon=horizon,
            false_alarm_budget=false_alarm_budget,
            event_rule_name=event_rule_name,
            model_scope_fingerprint=model_scope_fingerprint,
            c_grid=c_grid,
        ).fit(cal_subset)
        evaluation = result.readout.evaluate(eval_subset)
        results[label] = {
            "feature_names": list(names),
            "selected_c": result.selected_c,
            "threshold": result.readout.threshold,
            "calibration": result.calibration_evaluation.to_dict(),
            "evaluation": evaluation.to_dict(),
        }
    return results


def false_alarm_budget_curve(
    *,
    calibration_episodes: Sequence[WarningEpisode],
    evaluation_episodes: Sequence[WarningEpisode],
    feature_names: Sequence[str],
    horizon: int,
    budgets: Sequence[float],
    event_rule_name: str,
    model_scope_fingerprint: str,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> list[dict[str, Any]]:
    """Keep a fitted score model fixed and vary only the calibration alarm budget."""

    if not calibration_episodes or not evaluation_episodes:
        raise ValueError("calibration_episodes and evaluation_episodes are required")
    full_names = calibration_episodes[0].feature_names
    if any(ep.feature_names != full_names for ep in tuple(calibration_episodes) + tuple(evaluation_episodes)):
        raise ValueError("all episodes must share the same full feature schema")
    requested = tuple(feature_names)
    index = {name: i for i, name in enumerate(full_names)}
    missing = [name for name in requested if name not in index]
    if missing:
        raise KeyError(f"unknown features {missing!r}")
    columns = [index[name] for name in requested]
    calibration_subset = [_subset_episode(ep, requested, columns) for ep in calibration_episodes]
    evaluation_subset = [_subset_episode(ep, requested, columns) for ep in evaluation_episodes]
    calibrator = WarningCalibrator(
        feature_names=requested,
        horizon=horizon,
        false_alarm_budget=0.0,
        event_rule_name=event_rule_name,
        model_scope_fingerprint=model_scope_fingerprint,
        c_grid=c_grid,
    )
    model = calibrator._fit_score_model(calibration_subset)
    cal_scores = {ep.episode_id: model.scores(ep.states) for ep in calibration_subset}
    eval_scores = {ep.episode_id: model.scores(ep.states) for ep in evaluation_subset}
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        if not (0 <= budget <= 1):
            raise ValueError("budgets must lie in [0, 1]")
        threshold = _select_threshold(
            calibration_subset,
            cal_scores,
            horizon=horizon,
            false_alarm_budget=float(budget),
        )
        cal_eval = _evaluate_scores(calibration_subset, cal_scores, threshold=threshold, horizon=horizon)
        eval_eval = _evaluate_scores(evaluation_subset, eval_scores, threshold=threshold, horizon=horizon)
        rows.append(
            {
                "false_alarm_budget": float(budget),
                "threshold": float(threshold),
                "selected_c": model.selected_c,
                "calibration": cal_eval.to_dict(),
                "evaluation": eval_eval.to_dict(),
            }
        )
    return rows


def _validate_episode_schema(episodes: Sequence[WarningEpisode], feature_names: Sequence[str]) -> None:
    if not episodes:
        raise ValueError("at least one episode is required")
    names = tuple(feature_names)
    ids: set[str] = set()
    for episode in episodes:
        if episode.feature_names != names:
            raise ValueError(
                f"episode {episode.episode_id!r} has feature schema {episode.feature_names!r}, expected {names!r}"
            )
        if episode.episode_id in ids:
            raise ValueError(f"duplicate episode_id {episode.episode_id!r}")
        ids.add(episode.episode_id)


def _window_training_rows(
    episodes: Sequence[WarningEpisode], horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    groups: list[Hashable] = []
    for episode in episodes:
        stop = episode.states.shape[0] if episode.event_time is None else episode.event_time
        for t in range(stop):
            rows.append(episode.states[t])
            lead = None if episode.event_time is None else episode.event_time - t
            targets.append(int(lead is not None and 1 <= lead <= horizon))
            groups.append(episode.group if episode.group is not None else episode.episode_id)
    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=int)
    group_arr = np.asarray(groups, dtype=object)
    if np.unique(y).size < 2:
        raise ValueError("calibration window targets contain only one class")
    return X, y, group_arr


def _select_c(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    c_grid: Sequence[float],
    random_state: int,
) -> float:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return min(c_grid, key=lambda c: abs(np.log10(c)))
    splitter = LeaveOneGroupOut()
    best_c = c_grid[0]
    best_loss = float("inf")
    for c in c_grid:
        fold_losses: list[float] = []
        for train, test in splitter.split(X, y, groups):
            if np.unique(y[train]).size < 2:
                continue
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=5000,
                random_state=random_state,
            ).fit(scaler.transform(X[train]), y[train])
            probabilities = model.predict_proba(scaler.transform(X[test]))[:, 1]
            fold_losses.append(float(log_loss(y[test], probabilities, labels=[0, 1])))
        if fold_losses:
            score = float(np.mean(fold_losses))
            if score < best_loss - 1e-12 or (abs(score - best_loss) <= 1e-12 and c < best_c):
                best_loss = score
                best_c = c
    return float(best_c)


def _select_threshold(
    episodes: Sequence[WarningEpisode],
    score_map: Mapping[str, np.ndarray],
    *,
    horizon: int,
    false_alarm_budget: float,
) -> float:
    all_scores = np.concatenate([np.asarray(score_map[ep.episode_id], dtype=float) for ep in episodes])
    candidates = np.unique(np.concatenate(([np.nextafter(float(all_scores.max()), np.inf)], all_scores)))
    best_threshold = float(candidates.max())
    best_key: tuple[float, ...] | None = None
    for threshold in candidates:
        evaluation = _evaluate_scores(episodes, score_map, threshold=float(threshold), horizon=horizon)
        if evaluation.false_alarm_rate > false_alarm_budget + 1e-12:
            continue
        median_lead = -1.0 if evaluation.median_lead is None else float(evaluation.median_lead)
        key = (
            float(evaluation.timely_events),
            float(-evaluation.false_controls),
            float(-evaluation.premature_events),
            median_lead,
            float(threshold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _evaluate_scores(
    episodes: Sequence[WarningEpisode],
    score_map: Mapping[str, np.ndarray],
    *,
    threshold: float,
    horizon: int,
) -> WarningEvaluation:
    event_leads: list[int] = []
    timely = 0
    false = 0
    premature = 0
    late_or_missed = 0
    first_alarm: dict[str, int | None] = {}
    stored_scores: dict[str, tuple[float, ...]] = {}
    n_events = 0
    n_controls = 0
    for episode in episodes:
        scores = np.asarray(score_map[episode.episode_id], dtype=float)
        if scores.shape != (episode.states.shape[0],):
            raise ValueError(f"score length mismatch for episode {episode.episode_id!r}")
        alarm_indices = np.flatnonzero(scores >= threshold)
        alarm = None if alarm_indices.size == 0 else int(alarm_indices[0])
        first_alarm[episode.episode_id] = alarm
        stored_scores[episode.episode_id] = tuple(float(x) for x in scores)
        if episode.event_time is None:
            n_controls += 1
            false += int(alarm is not None)
            continue
        n_events += 1
        if alarm is None:
            late_or_missed += 1
            continue
        lead = int(episode.event_time - alarm)
        if 1 <= lead <= horizon:
            timely += 1
            event_leads.append(lead)
        elif lead > horizon:
            premature += 1
        else:
            late_or_missed += 1
    return WarningEvaluation(
        n_events=n_events,
        n_controls=n_controls,
        timely_events=timely,
        false_controls=false,
        timely_rate=float(timely / n_events) if n_events else float("nan"),
        false_alarm_rate=float(false / n_controls) if n_controls else float("nan"),
        median_lead=float(np.median(event_leads)) if event_leads else None,
        premature_events=premature,
        late_or_missed_events=late_or_missed,
        first_alarm_by_episode=first_alarm,
        score_by_episode=stored_scores,
    )


def _episodes_hash(episodes: Sequence[WarningEpisode]) -> str:
    hasher = hashlib.sha256()
    for episode in sorted(episodes, key=lambda ep: ep.episode_id):
        hasher.update(episode.episode_id.encode("utf-8"))
        hasher.update(str(episode.event_time).encode("ascii"))
        hasher.update(repr(episode.group).encode("utf-8"))
        hasher.update((episode.identity_set_id or "").encode("utf-8"))
        hasher.update("\0".join(episode.feature_names).encode("utf-8"))
        hasher.update(np.ascontiguousarray(episode.states).tobytes())
    return hasher.hexdigest()


def _subset_episode(
    episode: WarningEpisode, names: tuple[str, ...], columns: Sequence[int]
) -> WarningEpisode:
    return WarningEpisode(
        episode_id=episode.episode_id,
        states=episode.states[:, columns],
        feature_names=names,
        event_time=episode.event_time,
        group=episode.group,
        identity_set_id=episode.identity_set_id,
        metadata=episode.metadata,
    )
