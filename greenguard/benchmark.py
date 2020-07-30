from itertools import product

import pandas as pd
from sklearn.model_selection import train_test_split

from greenguard.demo import load_demo
from greenguard.pipeline import GreenGuardPipeline


def build_init_params(template, window_size, rule):
    if 'lstm' in template:
        window_size = int(pd.to_timedelta(window_size) / pd.to_timedelta(rule))
        return [{
            'pandas.DataFrame.resample#1': {
                'rule': rule,
            },
            'featuretools.dfs.json#1': {
                'window_size': window_size,
            }
        }]
    elif 'dfs' in template:
        return [{
            'pandas.DataFrame.resample#1': {
                'rule': rule,
            },
            'mlprimitives.custom.timeseries_preprocessing.cutoff_window_sequences#1': {
                'training_window': window_size,
            }
        }]


def build_init_preprocessing(templates, template, preprocessing):
    if isinstance(preprocessing, dict):
        return preprocessing[template]
    elif isinstance(preprocessing, list):
        return preprocessing[templates.index(template)]
    else:
        return preprocessing


def evaluate_template(template, metric, target_times, readings, tuning_iterations, preprocessing=0,
                      init_params=None, cost=False, test_size=0.25, cv_splits=3, random_state=0):

    scores = {}

    try:

        train, test = train_test_split(target_times, test_size=test_size,
                                       random_state=random_state)

        pipeline = GreenGuardPipeline(template, metric, cost=cost, cv_splits=cv_splits,
                                      init_params=init_params, preprocessing=preprocessing)

        # Computing the default test score
        pipeline.fit(train, readings)
        predictions = pipeline.predict(test, readings)

        scores['default_test'] = metric(test['target'], predictions)

        # Computing de default cross validation score
        session = pipeline.tune(train, readings)
        session.run(1)

        scores['default_cv'] = pipeline.cv_score

        # Computing the cross validation score with tuned hyperparameters
        session.run(tuning_iterations)

        scores['tuned_cv'] = pipeline.cv_score

        # Computing the test score with tuned hyperparameters
        pipeline.fit(train, readings)
        predictions = pipeline.predict(test, readings)

        scores['tuned_test'] = metric(test['target'], predictions)

    except Exception:
        return scores
    else:
        return scores


def evaluate_templates(templates, window_size_rule, metric, tuning_iterations, readings=None,
                       target_times=None, preprocessing=0, cost=False, test_size=0.25,
                       cv_splits=3, random_state=0):

    if readings is None and target_times is None:
        target_times, readings = load_demo()

    scores_list = []

    for template, window_rule in product(templates, window_size_rule):

        window_size, rule = window_rule

        scores = dict()
        scores['template'] = template
        scores['window_size'] = window_size
        scores['rule'] = rule

        try:
            init_params = build_init_params(template, window_size, rule)
            init_preprocessing = build_init_preprocessing(templates, template, preprocessing)

            result = evaluate_template(
                template=template,
                metric=metric,
                target_times=target_times,
                readings=readings,
                tuning_iterations=tuning_iterations,
                preprocessing=init_preprocessing,
                init_params=init_params,
                cost=cost,
                test_size=test_size,
                cv_splits=cv_splits,
                random_state=random_state)

            scores.update(result)
            scores_list.append(scores)

        except Exception:
            print('Error')

    return pd.DataFrame.from_records(scores_list)


'''
A possible call to this method could be:

    templates =  [
        'normalize_dfs_xgb_classifier',
        'unstack_double_lstm_timeseries_classifier',
    ]

    window_size_rule = [
        ('30d','12h'),
        ('30d','1d')
    ]

    preprocessing = [0,1,2]
    preprocessing = 1
    preprocessing = {
        'normalize_dfs_xgb_classifier': 0,
        'unstack_double_lstm_timeseries_classifier': 1,
        'unstack_dfs_xgb_classifier': 2
    }

    scores_df = evaluate_templates(
                    templates=templates,
                    window_size_rule=window_size_rule,
                    metric=f1_score,
                    tuning_iterations=50,
                    preprocessing=preprocessing,
                    cost=False,
                    test_size=0.25,
                    cv_splits=3,
                    random_state=0
                )

where scores_df is the resulting DataFrame.
'''
