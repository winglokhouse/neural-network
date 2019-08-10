import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

# one step average forecast


def simple_forecast1(history, config):
    n, offset, avg_type = config
    # if persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
    # collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
        # skip bad config
        if n * offset > len(history):
            pass
            # raise Exception('Config beyond end of data: {} {}'.format(n, offset))
        else:
            # try and collect n values using offset
            for i in range(1, n + 1):
                ix = i * offset
                # print('negative position : {}'.format(-ix))
                values.append(history[-ix])
    if len(values) == 0:
        return 0
    else:
        # mean of last n values
        # print('{} subprocess values = {}'.format(avg_type, values))
        if avg_type == 'mean':
            return np.mean(values)
        return np.median(values)


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def measure_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def walk_forward_validation(data, n_test, cfg):
    prediction = list()
    # split dataset
    X_train, X_test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in X_train]
    # step over each time stpe in the test set
    for i in range(len(X_test)):
        # fit model and make forecast for history
        yhat = simple_forecast1(history, cfg)
        # fit model and make forecast for history
        prediction.append(yhat)
        # add actual observation to history for the next loop
        history.append(X_test[i])
    # estimate prediction error
    error = measure_rmse(X_test, prediction)
    return error


def score_model(data, n_test, cfg, debug=True):
    result = None
    # convert config to key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings('ignore')
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
            print(error)
    if result is not None:
        pass
        # print(' > Model {} rmse error {}'.format(key, result))
    return (key, result)

# create a set of simple configs to try


def simple_configs(max_length, offsets=[1]):
    configs = list()
    for i in range(1, max_length + 1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs


def grid_search(data, cfg_list, n_test, parallel=False):
    scores = None
    if parallel:
        cpu = cpu_count()
        cpu = 4
        print('cpu_count {}'.format(cpu))
        executor = Parallel(n_jobs=cpu, backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [s for s in scores if s is not None]
    # sort scores
    scores.sort(key=lambda tup: tup[1])
    return scores
