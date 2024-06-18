import re
from collections import OrderedDict
from logging import Logger
from math import inf
from numbers import Number
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

import gpytorch
import numpy as np
import torch

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.optim.optimize import optimize_acqf
from botorch.optim.utils import (
    _handle_numerical_errors,
    get_parameters_and_bounds,
    TorchAttr,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.datasets import SupervisedDataset
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from scipy.optimize import approx_fprime
from torch import Tensor
from torch.nn.parameter import Parameter

from .utils import checked_cast, _to_inequality_constraints, _get_X_pending_and_observed, get_logger


def module_to_array(
    module: torch.nn.Module,
) -> Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
    r"""Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for optimizing.

    NOTE: `module_to_array` was originally a BoTorch function and was later
    deprecated. It has been copied here because ALEBO depends on it, and because
    ALEBO itself is deprecated, it is not worth moving ALEBO to the new syntax.

    Args:
        module: A module with parameters. May specify parameter constraints in
            a `named_parameters_and_constraints` method.

    Returns:
        3-element tuple containing
        - The parameter values as a numpy array.
        - An ordered dictionary with the name and tensor attributes of each
        parameter.
        - A `2 x n_params` numpy array with lower and upper bounds if at least
        one constraint is finite, and None otherwise.

    Example:
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
    """
    param_dict, bounds_dict = get_parameters_and_bounds(
        module=module,
        name_filter=None,
        requires_grad=True,
    )

    # Record tensor metadata and read parameter values to the tape
    param_tape: List[Number] = []
    property_dict = OrderedDict()
    with torch.no_grad():
        for name, param in param_dict.items():
            property_dict[name] = TorchAttr(param.shape, param.dtype, param.device)
            param_tape.extend(param.view(-1).cpu().double().tolist())

    # Extract lower and upper bounds
    start = 0
    bounds_np = None
    params_np = np.asarray(param_tape)
    for name, param in param_dict.items():
        numel = param.numel()
        if name in bounds_dict:
            for row, bound in enumerate(bounds_dict[name]):
                if bound is None:
                    continue

                if isinstance(bound, Tensor):
                    if torch.eq(bound, (2 * row - 1) * inf).all():
                        continue
                    bound = bound.detach().cpu()

                elif bound == (2 * row - 1) * inf:
                    continue

                if bounds_np is None:
                    bounds_np = np.full((2, len(params_np)), ((-inf,), (inf,)))

                bounds_np[row, start : start + numel] = bound
        start += numel

    return params_np, property_dict, bounds_np


TModule = TypeVar("TModule", bound=torch.nn.Module)


def set_params_with_array(
    module: TModule, x: np.ndarray, property_dict: Dict[str, TorchAttr]
) -> TModule:
    r"""Set module parameters with values from numpy array.

    NOTE: `set_params_with_array` was originally a BoTorch function and was
    later deprecated. It has been copied here because ALEBO depends on it, and
    because ALEBO itself is deprecated, it is not worth moving ALEBO to the new
    syntax.

    Args:
        module: Module with parameters to be set
        x: Numpy array with parameter values
        property_dict: Dictionary of parameter names and torch attributes as
            returned by module_to_array.

    Returns:
        Module: module with parameters updated in-place.

    Example:
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
        >>> parameter_array += 0.1  # perturb parameters (for example only)
        >>> mll = set_params_with_array(mll, parameter_array,  property_dict)
    """
    param_dict = OrderedDict(module.named_parameters())
    start_idx = 0
    for p_name, attrs in property_dict.items():
        # Construct the new tensor
        if len(attrs.shape) == 0:  # deal with scalar tensors
            end_idx = start_idx + 1
            new_data = torch.tensor(
                x[start_idx], dtype=attrs.dtype, device=attrs.device
            )
        else:
            end_idx = start_idx + np.prod(attrs.shape)
            new_data = torch.tensor(
                x[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
            ).view(*attrs.shape)
        start_idx = end_idx
        # Update corresponding parameter in-place. Disable autograd to update.
        param_dict[p_name].requires_grad_(False)
        param_dict[p_name].copy_(new_data)
        param_dict[p_name].requires_grad_(True)
    return module


def _scipy_objective_and_grad(
    x: np.ndarray, mll: ExactMarginalLogLikelihood, property_dict: Dict[str, TorchAttr]
) -> Tuple[Union[float, np.ndarray], np.ndarray]:
    r"""Get objective and gradient in format that scipy expects.


    NOTE: `_scipy_objective_and_grad` was originally a BoTorch function and was later
    deprecated. It has been copied here because ALEBO depends on it, and because
    ALEBO itself is deprecated, it is not worth moving ALEBO to the new syntax.

    Args:
        x: The (flattened) input parameters.
        mll: The MarginalLogLikelihood module to evaluate.
        property_dict: The property dictionary required to "unflatten" the input
            parameter vector, as generated by `module_to_array`.

    Returns:
        2-element tuple containing

        - The objective value.
        - The gradient of the objective.
    """
    mll = set_params_with_array(mll, x, property_dict)
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    mll.zero_grad()
    try:  # catch linear algebra errors in gpytorch
        output = mll.model(*train_inputs)
        args = [output, train_targets] + list(mll.model.train_inputs)
        # pyre-fixme[16]: Undefined attribute. Item
        # `torch.distributions.distribution.Distribution` of
        # `typing.Union[linear_operator.operators._linear_operator.LinearOperator,
        # torch._tensor.Tensor, torch.distributions.distribution.Distribution]`
        # has no attribute `sum`.
        loss = -mll(*args).sum()
    except RuntimeError as e:
        return _handle_numerical_errors(error=e, x=x)
    loss.backward()

    i = 0
    param_dict = OrderedDict(mll.named_parameters())
    grad = np.zeros(sum([tattr.shape.numel() for tattr in property_dict.values()]))
    for p_name in property_dict:
        t = param_dict[p_name]
        size = t.numel()
        t_grad = t.grad
        if t.requires_grad and t_grad is not None:
            grad[i : i + size] = t_grad.detach().view(-1).cpu().double().clone().numpy()
        i += size

    mll.zero_grad()
    return loss.item(), grad

class MahKernel(Kernel):
    """The kernel for ALEBO.

    Suppose there exists an ARD RBF GP on an (unknown) linear embedding with
    projection matrix A. We make function evaluations in a different linear
    embedding with projection matrix B (known). This is the appropriate kernel
    for fitting those data.

    This kernel computes a Mahalanobis distance, and the (d x d) PD distance
    matrix Gamma is a parameter that must be fit. This is done by fitting its
    upper Cholesky decomposition, U.

    Args:
        B: (d x D) Projection matrix.
        batch_shape: Batch shape as usual for gpytorch kernels.
    """

    def __init__(self, B: Tensor, batch_shape: torch.Size) -> None:
        super().__init__(
            has_lengthscale=False, ard_num_dims=None, eps=0.0, batch_shape=batch_shape
        )
        warn(
            "ALEBOKernel is deprecated and should be removed in Ax 0.5.0.",
            DeprecationWarning,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.d, D = B.shape
        self.D = D
        if not self.d < D:
            raise ValueError(f"Expected B.shape[0] < B.shape[1], but got {B.shape=}.")
        self.B = B
        # Initialize U
        Arnd = torch.randn(D, D, dtype=B.dtype, device=B.device)
        Arnd = torch.linalg.qr(Arnd)[0]
        Rrnd = Arnd[: self.d, :]  # shape(d, D)

        Uvec = Rrnd.T.contiguous().view(-1).repeat(*batch_shape, 1)
        self.register_parameter(name="Uvec", parameter=torch.nn.Parameter(Uvec))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        """Compute kernel distance."""
        # Unpack Uvec into an upper triangular matrix U
        shapeU = self.Uvec.shape[:-1] + torch.Size([self.D, self.d])
        U_t = checked_cast(Tensor, self.Uvec).view(shapeU)
        # Compute kernel distance
        z1 = torch.matmul(x1, U_t)
        z2 = torch.matmul(x2, U_t)

        diff = self.covar_dist(
            z1,
            z2,
            square_dist=True,
            diag=diag,
            **params,
        )
        return postprocess_rbf(diff)


class MahGP(SingleTaskGP):
    """The GP for ALEBO.

    Uses the Mahalanobis kernel defined in ALEBOKernel, along with a
    ScaleKernel to add a kernel variance and a fitted constant mean.

    In non-batch mode, there is a single kernel that produces MVN predictions
    as usual for a GP.
    With b batches, each batch has its own set of kernel hyperparameters and
    each batch represents a sample from the hyperparameter posterior
    distribution. When making a prediction (with `__call__`), these samples are
    integrated over using moment matching. So, the predictions are an MVN as
    usual with the same shape as in non-batch mode.

    Args:
        B: (d x D) Projection matrix.
        train_X: (n x d) X training data.
        train_Y: (n x 1) Y training data.
        train_Yvar: (n x 1) Noise variances of each training Y.
    """

    def __init__(
        self, B: Tensor, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor
    ) -> None:
        warn(
            "ALEBOGP is deprecated and should be removed in Ax 0.5.0. SAASBO "
            "(Models.SAASBO from ax.modelbridge.registry) likely provides better "
            "performance.",
            DeprecationWarning,
        )
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.covar_module = ScaleKernel(
            base_kernel=MahKernel(B=B, batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )
        self.to(train_X)

    def __call__(self, x: Tensor) -> MultivariateNormal:
        """
        If model is non-batch, then just make a prediction. If model has
        multiple batches, then these are samples from the kernel hyperparameter
        posterior and we integrate over them with moment matching.

        The shape of the MVN that this outputs will be the same regardless of
        whether the model is batched or not.

        Args:
            x: Point to be predicted.

        Returns: MultivariateNormal distribution of prediction.
        """
        if len(self._aug_batch_shape) == 0:
            return super().__call__(x)
        # Else, approximately integrate over batches with moment matching.
        # Take X as (b) x q x d, and expand to (b) x ns x q x d
        if x.ndim > 3:
            raise ValueError("Don't know how to predict this shape")
        x = x.unsqueeze(-3).expand(
            x.shape[:-2]
            + torch.Size([self._aug_batch_shape[0]])  # pyre-ignore
            # pyre-fixme[58]: `+` is not supported for operand types `Tuple[int,
            #  ...]` and `Size`.
            + x.shape[-2:]
        )
        mvn_b = super().__call__(x)
        mu = mvn_b.mean.mean(dim=-2)
        C = (
            mvn_b.covariance_matrix.mean(dim=-3)
            + torch.matmul(mvn_b.mean.transpose(-2, -1), mvn_b.mean)
            / mvn_b.mean.shape[-2]
            - torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
        )  # Law of Total Covariance
        mvn = MultivariateNormal(mu, C)
        return mvn

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        assert output_indices is None
        assert not observation_noise
        mvn = self(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior


def get_fitted_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    restarts: int,
    nsamp: int,
    init_state_dict: Optional[Dict[str, Tensor]],
) -> MahGP:
    """Get a fitted ALEBO GP.

    We do random restart optimization to get a MAP model, then use the Laplace
    approximation to draw posterior samples of kernel hyperparameters, and
    finally construct a batch-mode model where each batch is one of those
    sampled sets of kernel hyperparameters.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        restarts: Number of restarts for MAP estimation.
        nsamp: Number of samples to draw from kernel hyperparameter posterior.
        init_state_dict: Optionally begin MAP estimation with this state dict.

    Returns: Batch-mode (nsamp batches) fitted ALEBO GP.
    """
    warn(
        "`get_fitted_model` from ax.models.torch.alebo.py is deprecated and "
        "should be removed in Ax 0.5.0.",
        DeprecationWarning,
    )
    # Get MAP estimate.
    mll, m = get_map_model(
        B=B,
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        restarts=restarts,
        init_state_dict=init_state_dict,
    )
    return m
    # Compute Laplace approximation of posterior
    Uvec_batch, mean_constant_batch, output_scale_batch = laplace_sample_U(
        mll=mll, nsamp=nsamp
    )
    # Construct batch model with samples
    m_b = get_batch_model(
        B=B,
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        Uvec_batch=Uvec_batch,
        mean_constant_batch=mean_constant_batch,
        output_scale_batch=output_scale_batch,
    )
    return m_b


def get_map_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    restarts: int,
    init_state_dict: Optional[Dict[str, Tensor]],
) -> ExactMarginalLogLikelihood:
    """Do random-restart optimization for MAP fitting of an ALEBO GP model.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        restarts: Number of restarts for MAP estimation.
        init_state_dict: Optionally begin MAP estimation with this state dict.

    Returns: non-batch ALEBO GP with MAP kernel hyperparameters.
    """
    warn(
        "`get_map_model` from ax.models.torch.alebo.py is deprecated and should "
        "be removed in Ax 0.5.0.",
        DeprecationWarning,
    )
    f_best = 1e8
    sd_best = {}
    # Fit with random restarts
    for _ in range(restarts):
        m = MahGP(B=B, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        if init_state_dict is not None:
            m.load_state_dict(init_state_dict)
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        mll.train()
        result = fit_gpytorch_mll_scipy(mll, method="tnc")
        # logger.debug(result)
        if result.fval < f_best:
            f_best = float(result.fval)
            sd_best = m.state_dict()
    # Set the final value
    m = MahGP(B=B, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
    m.load_state_dict(sd_best)
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    return mll, m


def laplace_sample_U(
    mll: ExactMarginalLogLikelihood, nsamp: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Draw posterior samples of kernel hyperparameters using Laplace
    approximation.

    Only the Mahalanobis distance matrix is sampled.

    The diagonal of the Hessian is estimated using finite differences of the
    autograd gradients. The Laplace approximation is then N(p_map, inv(-H)).
    We construct a set of nsamp kernel hyperparameters by drawing nsamp-1
    values from this distribution, and prepending as the first sample the MAP
    parameters.

    Args:
        mll: MLL object of MAP ALEBO GP.
        nsamp: Number of samples to return.

    Returns: Batch tensors of the kernel hyperparameters Uvec, mean constant,
        and output scale.
    """
    warn(
        "laplace_sample_U is deprecated and should be removed in Ax 0.5.0.",
        DeprecationWarning,
    )
    # Estimate diagonal of the Hessian
    mll.train()
    x0, property_dict, bounds = module_to_array(module=mll)
    x0 = x0.astype(np.float64)  # This is the MAP parameters
    H = np.zeros((len(x0), len(x0)))
    epsilon = 1e-4 + 1e-3 * np.abs(x0)
    for i, _ in enumerate(x0):
        # Compute gradient of df/dx_i wrt x_i
        # pyre-fixme[53]: Captured variable `property_dict` is not annotated.
        # pyre-fixme[53]: Captured variable `x0` is not annotated.
        # pyre-fixme[53]: Captured variable `i` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def f(x):
            x_all = x0.copy()
            x_all[i] = x[0]
            return -_scipy_objective_and_grad(x_all, mll, property_dict)[1][i]

        H[i, i] = approx_fprime(np.array([x0[i]]), f, epsilon=epsilon[i])  # pyre-ignore

    # Sample only Uvec; leave mean and output scale fixed.
    assert list(property_dict.keys()) == [
        "model.mean_module.raw_constant",
        "model.covar_module.raw_outputscale",
        "model.covar_module.base_kernel.Uvec",
    ]
    H = H[2:, 2:]
    H += np.diag(-1e-3 * np.ones(H.shape[0]))  # Add a nugget for inverse stability
    Sigma = np.linalg.inv(-H)
    samples = np.random.multivariate_normal(mean=x0[2:], cov=Sigma, size=(nsamp - 1))
    # Include the MAP estimate
    samples = np.vstack((x0[2:], samples))
    # Reshape
    attrs = property_dict["model.covar_module.base_kernel.Uvec"]
    Uvec_batch = torch.tensor(samples, dtype=attrs.dtype, device=attrs.device).reshape(
        nsamp, *attrs.shape
    )
    # Get the other properties into batch mode
    mean_constant_batch = mll.model.mean_module.constant.repeat(nsamp)
    output_scale_batch = mll.model.covar_module.raw_outputscale.repeat(nsamp)
    return Uvec_batch, mean_constant_batch, output_scale_batch


def get_batch_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    Uvec_batch: Tensor,
    mean_constant_batch: Tensor,
    output_scale_batch: Tensor,
) -> MahGP:
    """Construct a batch-mode ALEBO GP using batch tensors of hyperparameters.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        Uvec_batch: Batch tensor of Uvec hyperparameters.
        mean_constant_batch: Batch tensor of mean constant hyperparameter.
        output_scale_batch: Batch tensor of output scale hyperparameter.

    Returns: Batch-mode ALEBO GP.
    """
    warn(
        "`get_batch_model` from ax.models.torch.alebo.py is deprecated and "
        "should be removed in Ax 0.5.0.",
        DeprecationWarning,
    )
    b = Uvec_batch.size(0)
    m_b = MahGP(
        B=B,
        train_X=train_X.repeat(b, 1, 1),
        train_Y=train_Y.repeat(b, 1, 1),
        train_Yvar=train_Yvar.repeat(b, 1, 1),
    )
    m_b.train()
    # Set mean constant
    # pyre-fixme[16]: `Optional` has no attribute `raw_constant`.
    m_b.mean_module.raw_constant.requires_grad_(False)
    m_b.mean_module.raw_constant.copy_(mean_constant_batch)
    m_b.mean_module.raw_constant.requires_grad_(True)
    # Set output scale
    m_b.covar_module.raw_outputscale.requires_grad_(False)
    checked_cast(Parameter, m_b.covar_module.raw_outputscale).copy_(output_scale_batch)
    m_b.covar_module.raw_outputscale.requires_grad_(True)
    # Set Uvec
    m_b.covar_module.base_kernel.Uvec.requires_grad_(False)
    checked_cast(Parameter, m_b.covar_module.base_kernel.Uvec).copy_(Uvec_batch)
    m_b.covar_module.base_kernel.Uvec.requires_grad_(True)
    m_b.eval()
    return m_b
