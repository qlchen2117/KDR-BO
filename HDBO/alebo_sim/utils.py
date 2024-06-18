import re
import numpy as np
from scipy.stats import uniform
import torch
from torch import Tensor
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Set, DefaultDict, Callable
from collections import defaultdict
T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K")
X = TypeVar("X")
Y = TypeVar("Y")

Tensoray = Union[torch.Tensor, np.ndarray]
AX_ROOT_LOGGER_NAME = "ax"
DEFAULT_LOG_LEVEL: int = logging.INFO

def checked_cast(typ: Type[T], val: V, exception: Optional[Exception] = None) -> T:
    """
    Cast a value to a type (with a runtime safety check).

    Returns the value unchanged and checks its type at runtime. This signals to the
    typechecker that the value has the designated type.

    Like `typing.cast`_ ``check_cast`` performs no runtime conversion on its argument,
    but, unlike ``typing.cast``, ``checked_cast`` will throw an error if the value is
    not of the expected type. The type passed as an argument should be a python class.

    Args:
        typ: the type to cast to
        val: the value that we are casting
        exception: override exception to raise if  typecheck fails
    Returns:
        the ``val`` argument, unchanged

    .. _typing.cast: https://docs.python.org/3/library/typing.html#typing.cast
    """
    if not isinstance(val, typ):
        raise exception if exception is not None else ValueError(
            f"Value was not of type {typ}:\n{val}"
        )
    return val


def _to_inequality_constraints(
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None
) -> Optional[List[Tuple[Tensor, Tensor, float]]]:
    if linear_constraints is not None:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indices = torch.atleast_1d(A[i, :].nonzero(as_tuple=False).squeeze())
            coefficients = torch.atleast_1d(-A[i, indices])
            rhs = -b[i, 0].item()
            inequality_constraints.append((indices, coefficients, rhs))
    else:
        inequality_constraints = None
    return inequality_constraints


def _filter_X_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    r"""Filter input points to those appearing in objective or constraints.

    Args:
        Xs: The input tensors of a model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: All points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    # Get points observed for all objective and constraint outcomes
    X_obs = get_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
    )
    # Filter to those that satisfy constraints.
    X_obs = filter_constraints_and_fixed_features(
        X=X_obs,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    if len(X_obs) > 0:
        return torch.as_tensor(X_obs)  # please the linter

def _get_X_pending_and_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    r"""Get pending and observed points.

    If all points would otherwise be filtered, remove `linear_constraints`
    and `fixed_features` from filter and retry.

    Args:
        Xs: The input tensors of a model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        pending_observations:  A list of m (k_i x d) feature tensors X
            for m outcomes and k_i pending observations for outcome i.
            (Only used if n > 1).
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: Pending points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
        Tensor: Observed points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    if pending_observations is None:
        X_pending = None
    else:
        X_pending = _filter_X_observed(
            Xs=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
    filtered_X_observed = _filter_X_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    if filtered_X_observed is not None and len(filtered_X_observed) > 0:
        return X_pending, filtered_X_observed
    else:
        unfiltered_X_observed = _filter_X_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            outcome_constraints=outcome_constraints,
        )
        return X_pending, unfiltered_X_observed


def get_observed(
    Xs: Union[List[torch.Tensor], List[np.ndarray]],
    objective_weights: Tensoray,
    outcome_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
) -> Tensoray:
    """Filter points to those that are observed for objective outcomes and outcomes
    that show up in outcome_constraints (if there are any).

    Args:
        Xs: A list of m (k_i x d) feature matrices X. Number of rows k_i
            can vary from i=1,...,m.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.

    Returns:
        Points observed for all objective outcomes and outcome constraints.
    """
    objective_weights_np = as_array(objective_weights)
    used_outcomes: Set[int] = set(np.where(objective_weights_np != 0)[0])
    if len(used_outcomes) == 0:
        raise ValueError("At least one objective weight must be non-zero")
    if outcome_constraints is not None:
        used_outcomes = used_outcomes.union(
            np.where(as_array(outcome_constraints)[0] != 0)[1]
        )
    outcome_list = list(used_outcomes)
    X_obs_set = {tuple(float(x_i) for x_i in x) for x in Xs[outcome_list[0]]}
    for _, idx in enumerate(outcome_list, start=1):
        X_obs_set = X_obs_set.intersection(
            {tuple(float(x_i) for x_i in x) for x in Xs[idx]}
        )
    if isinstance(Xs[0], np.ndarray):
        # pyre-fixme[6]: For 2nd param expected `Union[None, Dict[str, Tuple[typing.A...
        return np.array(list(X_obs_set), dtype=Xs[0].dtype)  # (n x d)
    if isinstance(Xs[0], torch.Tensor):
        # pyre-fixme[7]: Expected `Union[np.ndarray, torch.Tensor]` but got implicit
        #  return value of `None`.
        # pyre-fixme[6]: For 3rd param expected `Optional[_C.dtype]` but got
        #  `Union[np.dtype, _C.dtype]`.
        return torch.tensor(list(X_obs_set), device=Xs[0].device, dtype=Xs[0].dtype)


def filter_constraints_and_fixed_features(
    X: Tensoray,
    bounds: List[Tuple[float, float]],
    linear_constraints: Optional[Tuple[Tensoray, Tensoray]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tensoray:
    """Filter points to those that satisfy bounds, linear_constraints, and
    fixed_features.

    Args:
        X: An tensor or array of points.
        bounds: A list of (lower, upper) tuples for each feature.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value in the best point.

    Returns:
        Feasible points.
    """
    if len(X) == 0:  # if there are no points, nothing to filter
        return X
    X_np = X
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    feas = np.ones(X_np.shape[0], dtype=bool)  # (n)
    for i, b in enumerate(bounds):
        feas &= (X_np[:, i] >= b[0]) & (X_np[:, i] <= b[1])
    if linear_constraints is not None:
        A, b = as_array(linear_constraints)  # (m x d) and (m x 1)
        feas &= (A @ X_np.transpose() <= b).all(axis=0)
    if fixed_features is not None:
        for idx, val in fixed_features.items():
            feas &= X_np[:, idx] == val
    X_feas = X_np[feas, :]
    if isinstance(X, torch.Tensor):
        return torch.from_numpy(X_feas).to(device=X.device, dtype=X.dtype)
    else:
        return X_feas

def as_array(
    x: Union[Tensoray, Tuple[Tensoray, ...]]
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Convert every item in a tuple of tensors/arrays into an array.

    Args:
        x: A tensor, array, or a tuple of potentially mixed tensors and arrays.

    Returns:
        x, with everything converted to array.
    """
    if isinstance(x, tuple):
        return tuple(as_array(x_i) for x_i in x)  # pyre-ignore
    elif isinstance(x, np.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    else:
        raise ValueError("Input to as_array must be numpy array or torch tensor")


class AxOutputNameFilter(logging.Filter):
    """This is a filter which sets the record's output_name, if
    not configured
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "output_name"):
            # pyre-ignore[16]: Record supports arbitrary attributes
            record.output_name = record.name
        return True


def get_logger(
    name: str, level: int = DEFAULT_LOG_LEVEL, force_name: bool = False
) -> logging.Logger:
    """Get an Axlogger.

    To set a human-readable "output_name" that appears in logger outputs,
    add `{"output_name": "[MY_OUTPUT_NAME]"}` to the logger's contextual
    information. By default, we use the logger's `name`

    NOTE: To change the log level on particular outputs (e.g. STDERR logs),
    set the proper log level on the relevant handler, instead of the logger
    e.g. logger.handers[0].setLevel(INFO)

    Args:
        name: The name of the logger.
        level: The level at which to actually log.  Logs
            below this level of importance will be discarded
        force_name: If set to false and the module specified
            is not ultimately a descendent of the `ax` module
            specified by `name`, "ax." will be prepended to `name`

    Returns:
        The logging.Logger object.
    """
    # because handlers are attached to the "ax" module
    if not force_name and not re.search(
        r"^{ax_root}(\.|$)".format(ax_root=AX_ROOT_LOGGER_NAME), name
    ):
        name = f"{AX_ROOT_LOGGER_NAME}.{name}"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addFilter(AxOutputNameFilter())
    return logger


def add_fixed_features(
    tunable_points: np.ndarray,
    d: int,
    fixed_features: Optional[Dict[int, float]],
    tunable_feature_indices: np.ndarray,
) -> np.ndarray:
    """Add fixed features to points in tunable space.

    Args:
        tunable_points: Points in tunable space.
        d: Dimension of parameter space.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        tunable_feature_indices: Parameter indices (in d) which are tunable.

    Returns:
        points: Points in the full d-dimensional space, defined by bounds.
    """
    n = np.shape(tunable_points)[0]
    points = np.zeros((n, d))
    points[:, tunable_feature_indices] = tunable_points
    if fixed_features:
        fixed_feature_indices = np.array(list(fixed_features.keys()))
        fixed_values = np.tile(list(fixed_features.values()), (n, 1))
        points[:, fixed_feature_indices] = fixed_values
    return points


DEFAULT_MAX_RS_DRAWS = 10000
TParamCounter = DefaultDict[int, int]

def rejection_sample(
    gen_unconstrained: Callable[
        [int, int, np.ndarray, Optional[Dict[int, float]]], np.ndarray
    ],
    n: int,
    d: int,
    tunable_feature_indices: np.ndarray,
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    deduplicate: bool = False,
    max_draws: Optional[int] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    existing_points: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Rejection sample in parameter space.

    Models must implement a `gen_unconstrained` method in order to support
    rejection sampling via this utility.
    """
    # We need to perform the round trip transformation on our generated point
    # in order to deduplicate in the original search space.
    # The transformation is applied above.
    if deduplicate and rounding_func is None:
        raise ValueError("Rounding function must be provided for deduplication.")

    failed_constraint_dict: TParamCounter = defaultdict(lambda: 0)
    # Rejection sample with parameter constraints.
    points = np.zeros((n, d))

    attempted_draws = 0
    successful_draws = 0
    if max_draws is None:
        max_draws = DEFAULT_MAX_RS_DRAWS

    while successful_draws < n and attempted_draws <= max_draws:
        # _gen_unconstrained returns points including fixed features.
        # pyre-ignore: Anonymous function w/ named args.
        point = gen_unconstrained(
            n=1,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )[0]

        # Note: this implementation may not be performant, if the feasible volume
        # is small, since applying the rounding_func is relatively expensive.
        # If sampling in spaces with low feasible volume is slow, this function
        # could be applied after checking the linear constraints.
        if rounding_func is not None:
            point = rounding_func(point)

        # Check parameter constraints, always in raw transformed space.
        if linear_constraints is not None:
            all_constraints_satisfied, violators = check_param_constraints(
                linear_constraints=linear_constraints, point=point
            )
            for violator in violators:
                failed_constraint_dict[violator] += 1
        else:
            all_constraints_satisfied = True
            violators = np.array([])

        # Deduplicate: don't add the same point twice.
        duplicate = False
        if deduplicate:
            if existing_points is not None:
                prev_points = np.vstack([points[:successful_draws, :], existing_points])
            else:
                prev_points = points[:successful_draws, :]
            duplicate = check_duplicate(point=point, points=prev_points)

        # Add point if valid.
        if all_constraints_satisfied and not duplicate:
            points[successful_draws] = point
            successful_draws += 1
        attempted_draws += 1

    if successful_draws < n:
        # Only possible if attempted_draws >= max_draws.
        raise Exception(
            f"Rejection sampling error (specified maximum draws ({max_draws}) exhausted"
            f", without finding sufficiently many ({n}) candidates). This likely means "
            "that there are no new points left in the search space."
        )
    else:
        return (points, attempted_draws)


def check_param_constraints(
    linear_constraints: Tuple[np.ndarray, np.ndarray], point: np.ndarray
) -> Tuple[bool, np.ndarray]:
    """Check if a point satisfies parameter constraints.

    Args:
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b.
        point: A candidate point in d-dimensional space, as a (1 x d) matrix.

    Returns:
        2-element tuple containing

        - Flag that is True if all constraints are satisfied by the point.
        - Indices of constraints which are violated by the point.
    """
    constraints_satisfied = (
        linear_constraints[0] @ np.expand_dims(point, axis=1) <= linear_constraints[1]
    )
    if np.all(constraints_satisfied):
        return True, np.array([])
    else:
        return (False, np.where(constraints_satisfied == False)[0])  # noqa: E712


def check_duplicate(point: np.ndarray, points: np.ndarray) -> bool:
    """Check if a point exists in another array.

    Args:
        point: Newly generated point to check.
        points: Points previously generated.

    Returns:
        True if the point is contained in points, else False
    """
    for p in points:
        if np.array_equal(p, point):
            return True
    return False
