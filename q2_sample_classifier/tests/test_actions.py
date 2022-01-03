# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os

import pandas as pd
import pandas.testing as pdt
import numpy as np
import biom

import qiime2
from qiime2.plugins import sample_classifier

from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase
from q2_sample_classifier.tests.test_estimators import SampleEstimatorTestBase
from q2_sample_classifier.classify import summarize


class NowLetsTestTheActions(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        md = pd.Series(['a', 'a', 'b', 'b', 'b'],
                       index=['a', 'b', 'c', 'd', 'e'], name='bugs')
        md.index.name = 'SampleID'
        self.md = qiime2.CategoricalMetadataColumn(md)

        tab = biom.Table(
            np.array([[3, 6, 7, 3, 6], [3, 4, 5, 6, 2], [8, 6, 4, 1, 0],
                      [8, 6, 4, 1, 0], [8, 6, 4, 1, 0]]),
            observation_ids=['v', 'w', 'x', 'y', 'z'],
            sample_ids=['a', 'b', 'c', 'd', 'e'])
        self.tab = qiime2.Artifact.import_data('FeatureTable[Frequency]', tab)

        md2 = pd.DataFrame({'trash': ['a', 'a', 'b', 'b', 'b', 'junk'],
                            'floats': [0.1, 0.1, 1.3, 1.8, 1000.1, 0.1],
                            'ints': [0, 1, 2, 2, 2, 0],
                            'nans': [1, 1, 2, 2, np.nan, np.nan],
                            'negatives': [-7, -3, -1.2, -4, -9, -1]},
                           index=['a', 'b', 'c', 'd', 'e', 'peanut'])
        md2.index.name = 'SampleID'
        self.md2 = qiime2.Metadata(md2)

    # let's make sure the function runs w/o errors and that the correct
    # transformers are in place (see issue 114)
    def test_action_split_table(self):
        res = sample_classifier.actions.split_table(
            self.tab, self.md, test_size=0.5)
        y_train = res.training_targets.view(pd.Series)
        y_test = res.test_targets.view(pd.Series)

        # test whether extracted target is correct
        self.assertEqual(y_train.name, 'bugs')

        # test if complete target column is covered
        y_all = y_train.append(y_test).sort_index()
        y_all.index.name = 'SampleID'
        pdt.assert_series_equal(y_all, self.md._series)

    def test_metatable(self):
        exp = biom.Table(
            np.array([[0.1, 0.1, 1.3, 1.8, 1000.1, 0.1],
                      [0, 1, 2, 2, 2, 0]]),
            observation_ids=['floats', 'ints'],
            sample_ids=['a', 'b', 'c', 'd', 'e', 'peanut'])
        res, = sample_classifier.actions.metatable(
            self.md2, missing_values='drop_features')
        report = res.view(biom.Table).descriptive_equality(exp)
        self.assertIn('Tables appear equal', report, report)

    def test_metatable_missing_error(self):
        with self.assertRaisesRegex(ValueError, "missing values"):
            sample_classifier.actions.metatable(
                self.md2, missing_values='error')

    def test_metatable_drop_samples(self):
        exp = biom.Table(
            np.array([[3, 6, 7, 3], [3, 4, 5, 6], [8, 6, 4, 1],
                      [8, 6, 4, 1], [8, 6, 4, 1],
                      [0.1, 0.1, 1.3, 1.8],
                      [0, 1, 2, 2], [1, 1, 2, 2]]),
            observation_ids=['v', 'w', 'x', 'y', 'z', 'floats', 'ints',
                             'nans'],
            sample_ids=['a', 'b', 'c', 'd'])
        res, = sample_classifier.actions.metatable(
            self.md2, self.tab, missing_values='drop_samples')
        report = res.view(biom.Table).descriptive_equality(exp)
        self.assertIn('Tables appear equal', report, report)

    def test_metatable_fill_na(self):
        exp = biom.Table(
            np.array([[3, 6, 7, 3, 6], [3, 4, 5, 6, 2], [8, 6, 4, 1, 0],
                      [8, 6, 4, 1, 0], [8, 6, 4, 1, 0],
                      [0.1, 0.1, 1.3, 1.8, 1000.1],
                      [0, 1, 2, 2, 2], [1., 1., 2., 2., 0.]]),
            observation_ids=['v', 'w', 'x', 'y', 'z', 'floats', 'ints',
                             'nans'],
            sample_ids=['a', 'b', 'c', 'd', 'e'])
        res, = sample_classifier.actions.metatable(
            self.md2, self.tab, missing_values='fill')
        report = res.view(biom.Table).descriptive_equality(exp)
        self.assertIn('Tables appear equal', report, report)

    def test_metatable_with_merge(self):
        exp = biom.Table(
            np.array([[3, 6, 7, 3, 6], [3, 4, 5, 6, 2], [8, 6, 4, 1, 0],
                      [8, 6, 4, 1, 0], [8, 6, 4, 1, 0],
                      [0.1, 0.1, 1.3, 1.8, 1000.1],
                      [0, 1, 2, 2, 2]]),
            observation_ids=['v', 'w', 'x', 'y', 'z', 'floats', 'ints'],
            sample_ids=['a', 'b', 'c', 'd', 'e'])
        res, = sample_classifier.actions.metatable(
            self.md2, self.tab, missing_values='drop_features')
        report = res.view(biom.Table).descriptive_equality(exp)
        self.assertIn('Tables appear equal', report, report)

    def test_metatable_with_merge_successful_inner_join(self):
        exp = biom.Table(
            np.array([[3, 6, 7, 3], [3, 4, 5, 6], [8, 6, 4, 1],
                      [8, 6, 4, 1], [8, 6, 4, 1], [0.1, 0.1, 1.3, 1.8],
                      [0, 1, 2, 2], [1., 1., 2., 2.]]),
            observation_ids=['v', 'w', 'x', 'y', 'z', 'floats', 'ints',
                             'nans'],
            sample_ids=['a', 'b', 'c', 'd'])
        res, = sample_classifier.actions.metatable(
            self.md2.filter_ids(['a', 'b', 'c', 'd']), self.tab,
            missing_values='error')
        report = res.view(biom.Table).descriptive_equality(exp)
        self.assertIn('Tables appear equal', report, report)

    def test_metatable_with_merge_error_inner_join(self):
        with self.assertRaisesRegex(ValueError, "Missing samples"):
            sample_classifier.actions.metatable(
                self.md2.filter_ids(['a', 'b', 'c', 'd']),
                self.tab, missing_samples='error',
                missing_values='drop_samples')

    def test_metatable_empty_metadata_after_drop_all_unique(self):
        with self.assertRaisesRegex(
                ValueError, "All metadata"):  # are belong to us
            sample_classifier.actions.metatable(
                self.md2.filter_ids(['b', 'c']), self.tab,
                missing_values='drop_samples', drop_all_unique=True)

    def test_metatable_no_samples_after_filtering(self):
        junk_md = pd.DataFrame(
            {'trash': ['a', 'a', 'b', 'b', 'b', 'junk'],
             'floats': [np.nan, np.nan, np.nan, 1.8, 1000.1, 0.1],
             'ints': [0, 1, 2, np.nan, 2, 0],
             'nans': [1, 1, 2, 2, np.nan, np.nan],
             'negatives': [-7, -4, -1.2, -4, -9, -1]},
            index=['a', 'b', 'c', 'd', 'e', 'peanut'])
        junk_md.index.name = 'SampleID'
        junk_md = qiime2.Metadata(junk_md)
        with self.assertRaisesRegex(ValueError, "All metadata samples"):
            sample_classifier.actions.metatable(
                junk_md, missing_values='drop_samples')


# make sure summarize visualizer works and that rfe_scores are stored properly
class TestSummarize(SampleEstimatorTestBase):

    def test_summary_with_rfecv(self):
        summarize(self.temp_dir.name, self.pipeline)

        self.assertTrue('rfe_plot.pdf' in os.listdir(self.temp_dir.name))
        self.assertTrue('rfe_plot.png' in os.listdir(self.temp_dir.name))
        self.assertTrue('rfe_scores.tsv' in os.listdir(self.temp_dir.name))

    def test_summary_without_rfecv(self):
        # nuke the rfe_scores to test the other branch of _summarize_estimator
        del self.pipeline.rfe_scores
        summarize(self.temp_dir.name, self.pipeline)

        self.assertFalse('rfe_plot.pdf' in os.listdir(self.temp_dir.name))
        self.assertFalse('rfe_plot.png' in os.listdir(self.temp_dir.name))
        self.assertFalse('rfe_scores.tsv' in os.listdir(self.temp_dir.name))
