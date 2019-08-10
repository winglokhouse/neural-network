from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
#
from simple_forecast_framework import measure_rmse, train_test_split
# create a set of exponential smoothing configs to try


def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            models.append(cfg)
    return models


def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    first_i_for_each_cfg = True
    cfg_viable = True
    for i in range(len(test)):
        # fit model and make forecast for history
        # yhat = exp_smoothing_forecast(history, cfg)
        t, d, s, p, b, r = cfg
        history_np = np.array(history)
        # print('t = {} d = {} s = {} p = {}'.format(t, d, s, p))
        try:
            model = ExponentialSmoothing(
                history_np, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            # fit model
            model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
            # make one step forecast
            yhat = model_fit.predict(len(history_np), len(history_np))
            if first_i_for_each_cfg:
                # print('OK {} yhat[0] {}'.format(cfg, yhat))
                first_i_for_each_cfg = False
        except AttributeError:
            # print('AttributeError {}'.format(cfg))
            cfg_viable = False
            break
        except ValueError:
            # print('ValueError {}'.format(cfg))
            cfg_viable = False
            break
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    if cfg_viable:
        try:
            error = measure_rmse(test, predictions)
        except ValueError:
            error = None
    else:
        error = None
    return error

# score a model, return None on failure


def score_model(data, n_test, cfg, debug=False):
    key = str(cfg)
    error = walk_forward_validation(data, n_test, cfg)
    return (key, error)


def grid_search(data, cfg_list, n_test, parallel=False):
    scores = None
    if parallel:
        cpu = cpu_count()
        # cpu = 4
        print('cpu_count {}'.format(cpu))
        executor = Parallel(n_jobs=cpu, backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    print(scores)
    scores = [s for s in scores if s[1] is not None]
    # sort scores
    scores.sort(key=lambda tup: tup[1])
    return scores
