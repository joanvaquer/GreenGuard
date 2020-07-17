import gc
import logging
import warnings

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

from greenguard.demo import load_demo
from greenguard.pipeline import GreenGuardPipeline

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(level=logging.WARNING)
warnings.simplefilter("ignore")
gc.enable()


def score_template(template, metric, target_times, readings, tuning_iterations,
                   init_params=None, cost=False, test_size=0.25, cv_splits=3, random_state=0):

    scores = {}

    try:
        train, test = train_test_split(target_times, test_size=test_size,
                                       random_state=random_state)
        pipeline = GreenGuardPipeline(template, metric, cost=cost,
                                      cv_splits=cv_splits, init_params=init_params)

        # Computing the default test score
        pipeline.fit(train, readings)
        predictions = pipeline.predict(test, readings)

        scores['default_test'] = f1_score(test['target'], predictions)

        # Computing de default cross validation score
        gc.collect()
        session = pipeline.tune(train, readings)
        session.run(1)

        scores['default_cv'] = pipeline.cv_score

        # Computing the cross validation score with tuned hyperparameters
        session.run(tuning_iterations)
        pipeline.get_hyperparameters()

        scores['tuned_cv'] = pipeline.cv_score

        # Computing the test score with tuned hyperparameters
        pipeline.fit(train, readings)
        predictions = pipeline.predict(test, readings)

        scores['tuned_test'] = f1_score(test['target'], predictions)

        return scores

    except:
        return scores


# returns the init_params of lstm
def build_lstm_init_params(rule, window_size):
    window_size = int(pd.to_timedelta(window_size) / pd.to_timedelta(rule))
    return [{
        'pandas.DataFrame.resample#1': {
            'rule': rule,
        },
        'featuretools.dfs.json#1': {
            'window_size': window_size,
        }
    }]


# returns the init_params of the dfs
def build_dfs_init_params(rule, window_size):
    return [{
        'pandas.DataFrame.resample#1': {
            'rule': rule,
        },
        'mlprimitives.custom.timeseries_preprocessing.cutoff_window_sequences#1': {
            'training_window': window_size,
        }
    }]


# evaluates the score of a pipeline with diferents window_size and rule
def evaluate_template(template, window_rule_size, metric, tuning_iterations, cost=False,
                      test_size=0.25, cv_splits=3, random_state=0):
    scores_list = []

    INIT_PARAMS_BUILDERS = {
        'resample_600s_normalize_dfs_1d_xgb_classifier': build_dfs_init_params,
        'resample_600s_unstack_double_144_lstm_timeseries_classifier': build_lstm_init_params,
    }

    target_times, readings = load_demo()

    for x in window_rule_size:
        window_size = x[0]
        rule = x[1]

        init_params_builder = INIT_PARAMS_BUILDERS[template]
        init_params = init_params_builder(rule, window_size)

        scores = score_template(
            template=template,
            metric=f1_score,
            target_times=target_times,
            readings=readings,
            tuning_iterations=50,
            init_params=init_params,
            cost=False,
            test_size=0.25,
            cv_splits=3,
            random_state=0)

        scores['template'] = template
        scores['window_size'] = window_size
        scores['rule'] = rule
        scores_list.append(scores)

    return scores_list
