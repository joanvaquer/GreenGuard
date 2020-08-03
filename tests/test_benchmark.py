"""Tests for `greenguard.benchmark` module."""
from unittest import TestCase

from sklearn.metrics import f1_score

from greenguard import benchmark


class BenchmarkTest(TestCase):

    def test_predict(self):
        # setup
        templates = [
            'normalize_dfs_xgb_classifier'
        ]
        window_size_rule = [
            ('30d', '12h')
        ]
        # run
        scores_df = benchmark.evaluate_templates(
            templates=templates,
            window_size_rule=window_size_rule,
            metric=f1_score,
            tuning_iterations=5
        )

        # assert
        columns = (['template', 'window_size', 'resample_rule', 'default_test',
                    'default_cv', 'tuned_cv', 'tuned_test', 'status'])
        dtypes = ['object', 'object', 'object', 'float64', 'float64', 'float64', 'float64',
                  'object']
        assert(scores_df.columns == columns).all()
        assert(scores_df.tuned_test.notnull)
        assert(scores_df.dtypes == dtypes).all()
