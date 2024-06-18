import numpy as np
from .acq import kerAcq, acq
from scipy.optimize import direct, Bounds


def maximize_acq(model, dopt, acq_type='ei'):
    def fn_wrapper(x):
        return acq(model, x, acq_type).item()
    cur_min = np.inf

    if model.optimizer == 'direct':
        res = direct(fn_wrapper, model['bounds'], maxfun=dopt['maxfun'], maxiter=dopt['maxiter'])
        x_max_direct = res.x
        fmin_direct = fn_wrapper(x_max_direct)
        if fmin_direct < cur_min:
            x_min = x_max_direct
            cur_min = fmin_direct
    elif model.optimizer == 'CMAES':
        lb, ub = model.copts['LBounds'], model.copts['UBounds']
        mean = np.random.uniform(lb, ub)
        import cma
        x_cmaes, _ = cma.fmin2(
            fn_wrapper, mean, 1,
            options={'maxfevals': model.copts['maxfun'], 'bounds':[lb, ub],
                     'verbose': -9, 'tolfun': model.copts['TolFun']}
        )
        fmin_cmaes = fn_wrapper(x_cmaes)
        if fmin_cmaes < cur_min:
            x_min = x_cmaes
            cur_min = fmin_cmaes
    elif model.optimizer == 'fminsearch':
        raise NotImplementedError
        # use fmin_search to optimize the acquision even further.
        options['MaxFunEval'] = 800
        x_max_search, fmin_search, exit_flag, output = fminsearchbnd()

        if fmin_search < cur_min:
            x_min = x_max_search
            cur_min = fmin_search

    elif model.optimizer == 'use_gradient':
        raise NotImplementedError
        def funProj(x):
            max(min(x, model['bounds'][:, 1]), model['bounds'][:, 0])
            x_max_grad, fmin_grad = minConf_SPG()
            if fmin_grad < cur_min:
                x_min = x_max_grad
                cur_min = fmin_grad
    else:
        raise NotImplementedError
    return x_min


def maximize_kacq(model, dopt, acq_type='ei'):
    def fn_wrapper(x):
        return kerAcq(model, x, acq_type).item()
    cur_min = np.inf

    if model.optimizer == 'direct':
        res = direct(fn_wrapper, model['bounds'], maxfun=dopt['maxfun'], maxiter=dopt['maxiter'])
        x_max_direct = res.x
        fmin_direct = fn_wrapper(x_max_direct)
        if fmin_direct < cur_min:
            x_min = x_max_direct
            cur_min = fmin_direct
    elif model.optimizer == 'CMAES':
        lb, ub = model.copts['LBounds'], model.copts['UBounds']
        mean = np.random.uniform(lb, ub)
        import cma
        x_cmaes, _ = cma.fmin2(
            fn_wrapper, mean, 1,
            options={'maxfevals': model.copts['maxfun'], 'bounds':[lb, ub],
                     'verbose': -9, 'tolfun': model.copts['TolFun']}
        )
        fmin_cmaes = fn_wrapper(x_cmaes)
        if fmin_cmaes < cur_min:
            x_min = x_cmaes
            cur_min = fmin_cmaes
    elif model.optimizer == 'fminsearch':
        raise NotImplementedError
        # use fmin_search to optimize the acquision even further.
        options['MaxFunEval'] = 800
        x_max_search, fmin_search, exit_flag, output = fminsearchbnd()

        if fmin_search < cur_min:
            x_min = x_max_search
            cur_min = fmin_search

    elif model.optimizer == 'use_gradient':
        raise NotImplementedError
        def funProj(x):
            max(min(x, model['bounds'][:, 1]), model['bounds'][:, 0])
            x_max_grad, fmin_grad = minConf_SPG()
            if fmin_grad < cur_min:
                x_min = x_max_grad
                cur_min = fmin_grad
    else:
        raise NotImplementedError
    return x_min
