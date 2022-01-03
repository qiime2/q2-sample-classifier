# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import pandas.testing as pdt
from os import mkdir, listdir
from os.path import join
import biom

import qiime2
from qiime2.plugins import sample_classifier

from q2_sample_classifier.visuals import (
    _linear_regress, _calculate_baseline_accuracy,
    _add_sample_size_to_xtick_labels)
from q2_sample_classifier.classify import (
    scatterplot, confusion_matrix)
from q2_sample_classifier.utilities import (
    _match_series_or_die, _predict_and_plot)
from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase


class TestVisuals(SampleClassifierTestPluginBase):

    md = pd.DataFrame([(1, 'a', 0.11), (1, 'a', 0.12), (1, 'a', 0.13),
                       (2, 'a', 0.19), (2, 'a', 0.18), (2, 'a', 0.21),
                       (1, 'b', 0.14), (1, 'b', 0.13), (1, 'b', 0.14),
                       (2, 'b', 0.26), (2, 'b', 0.27), (2, 'b', 0.29)],
                      columns=['Time', 'Group', 'Value'])

    def test_linear_regress(self):
        res = _linear_regress(self.md['Value'], self. md['Time'])
        self.assertAlmostEqual(res.iloc[0]['Mean squared error'], 1.9413916666)
        self.assertAlmostEqual(res.iloc[0]['r-value'], 0.86414956372460128)
        self.assertAlmostEqual(res.iloc[0]['r-squared'], 0.74675446848541871)
        self.assertAlmostEqual(res.iloc[0]['P-value'], 0.00028880275858705694)

    def test_calculate_baseline_accuracy(self):
        accuracy = 0.9
        y_test = pd.Series(['a', 'a', 'a', 'b', 'b', 'b'], name="class")
        classifier_accuracy = _calculate_baseline_accuracy(y_test, accuracy)
        expected_results = (6, 3, 0.5, 1.8)
        for i in zip(classifier_accuracy, expected_results):
            self.assertEqual(i[0], i[1])


class TestHeatmap(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        md_vaw = self.get_data_path('vaw.txt')
        md_vaw = qiime2.Metadata.load(md_vaw)
        self.md_vaw = md_vaw.get_column('Column')
        table_vaw = self.get_data_path('vaw.qza')
        self.table_vaw = qiime2.Artifact.load(table_vaw)
        imp = pd.read_csv(
            self.get_data_path('vaw_importance.tsv'), sep='\t',
            header=0, index_col=0)
        self.imp = qiime2.Artifact.import_data('FeatureData[Importance]', imp)

    def test_heatmap_default_feature_count_zero(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_vaw, self.imp, self.md_vaw, group_samples=True,
            feature_count=0)
        self.assertEqual(table.view(biom.Table).shape, (5, 2))

    def test_heatmap_importance_threshold(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_vaw, self.imp, self.md_vaw,
            importance_threshold=0.062, group_samples=False, feature_count=0)
        self.assertEqual(table.view(biom.Table).shape, (3, 6))

    def test_heatmap_feature_count(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_vaw, self.imp, self.md_vaw, group_samples=True,
            feature_count=2)
        self.assertEqual(table.view(biom.Table).shape, (2, 2))

    def test_heatmap_must_group_or_die(self):
        with self.assertRaisesRegex(ValueError, "metadata are not optional"):
            heatmap, table, = sample_classifier.actions.heatmap(
                self.table_vaw, self.imp, sample_metadata=None,
                group_samples=True)


# This class really just checks that these visualizers run without error. Yay.
# Also test some internal nuts/bolts but there's not much else we can do.
class TestPlottingVisualizers(SampleClassifierTestPluginBase):
    def setUp(self):
        super().setUp()
        self.tmpd = join(self.temp_dir.name, 'viz')
        mkdir(self.tmpd)

        self.a = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='site',
                           index=['a1', 'a2', 'b1', 'b2', 'c1', 'c2'])
        self.a.index.name = 'SampleID'
        self.bogus = pd.Series(['a', 'a', 'b', 'b', 'c', 'c'], name='site',
                               index=['a1', 'e3', 'f5', 'b2', 'z1', 'c2'])
        self.bogus.index.name = 'SampleID'
        self.c = pd.Series(
            [0, 1, 2, 3], index=['a', 'b', 'c', 'd'], name='peanuts')
        self.c.index.name = 'SampleID'

    def test_confusion_matrix(self):
        b = qiime2.CategoricalMetadataColumn(self.a)
        confusion_matrix(self.tmpd, self.a, b)

    def test_confusion_matrix_class_overlap_error(self):
        b = pd.Series([1, 2, 3, 4, 5, 6], name='site',
                      index=['a1', 'a2', 'b1', 'b2', 'c1', 'c2'])
        b.index.name = 'id'
        b = qiime2.NumericMetadataColumn(b)
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            confusion_matrix(self.tmpd, self.a, b)

    def test_confusion_matrix_vmin_too_high(self):
        b = qiime2.CategoricalMetadataColumn(self.a)
        with self.assertRaisesRegex(ValueError, r'vmin must be less than.*\s\s'
                                    r'0\.5.*greater.*0\.0'):
            confusion_matrix(self.tmpd, self.a, b, vmin=.5, vmax=None)

    def test_confusion_matrix_vmax_too_low(self):
        b = qiime2.CategoricalMetadataColumn(self.a)
        with self.assertRaisesRegex(ValueError, r'vmax must be greater than.*'
                                    r'\s\s0\.5.*less.*1\.0'):
            confusion_matrix(self.tmpd, self.a, b, vmin=None, vmax=.5)

    def test_confusion_matrix_vmin_too_high_and_vmax_too_low(self):
        b = qiime2.CategoricalMetadataColumn(self.a)
        with self.assertRaisesRegex(ValueError, r'vmin must be less than.*\s'
                                    r'\s0\.5.*greater.*0\.0\s.*vmax must be '
                                    r'greater than.*\s\s0\.5.*less.*1\.0'):
            confusion_matrix(self.tmpd, self.a, b, vmin=.5, vmax=.5)

    def test_confusion_matrix_dtype_coercion(self):
        predictions = pd.Series([1, 1, 1, 2, 2, 2],
                                index=pd.Index(['a', 'b', 'c', 'd', 'e', 'f'],
                                name='sample_id'), name='features')

        # NOTE: the targets are numbers but represented as str
        truth = qiime2.CategoricalMetadataColumn(pd.Series(
            ['1', '2', '1', '2', '1', '2'],
            index=pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], name='sample-id'),
            name='target'))

        confusion_matrix(self.tmpd, predictions, truth)

        self.assertTrue('index.html' in listdir(self.tmpd))

    # test confusion matrix plotting independently to see how it handles
    # partially overlapping classes when true labels are superset
    def test_predict_and_plot_true_labels_are_superset(self):
        b = pd.Series(['a', 'a', 'b', 'b', 'b', 'b'], name='site',
                      index=['a1', 'a2', 'b1', 'b2', 'c1', 'c2'])
        exp = pd.DataFrame(
            [[1., 0., 0., ''],
             [0., 1., 0., ''],
             [0., 1., 0., ''],
             ['', '', '', 0.666666666],
             ['', '', '', 0.3333333333],
             ['', '', '', 2.]],
            columns=['a', 'b', 'c', 'Overall Accuracy'],
            index=['a', 'b', 'c', 'Overall Accuracy', 'Baseline Accuracy',
                   'Accuracy Ratio'])
        predictions, confusion = _predict_and_plot(self.tmpd, self.a, b)
        pdt.assert_frame_equal(exp, predictions)

    # test confusion matrix plotting independently to see how it handles
    # partially overlapping classes when true labels are superset
    def test_predict_and_plot_true_labels_are_subset(self):
        b = pd.Series(['a', 'a', 'b', 'b', 'c', 'd'], name='site',
                      index=['a1', 'a2', 'b1', 'b2', 'c1', 'c2'])
        exp = pd.DataFrame(
            [[1., 0., 0., 0., ''],
             [0., 1., 0., 0., ''],
             [0., 0., 0.5, 0.5, ''],
             [0., 0., 0., 0., ''],
             ['', '', '', '', 0.8333333333],
             ['', '', '', '', 0.3333333333],
             ['', '', '', '', 2.5]],
            columns=['a', 'b', 'c', 'd', 'Overall Accuracy'],
            index=['a', 'b', 'c', 'd', 'Overall Accuracy', 'Baseline Accuracy',
                   'Accuracy Ratio'])
        predictions, confusion = _predict_and_plot(self.tmpd, self.a, b)
        pdt.assert_frame_equal(exp, predictions)

    # test confusion matrix plotting independently to see how it handles
    # partially overlapping classes when true labels are mutually exclusive
    def test_predict_and_plot_true_labels_are_mutually_exclusive(self):
        b = pd.Series(['a', 'a', 'e', 'e', 'd', 'd'], name='site',
                      index=['a1', 'a2', 'b1', 'b2', 'c1', 'c2'])
        exp = pd.DataFrame(
            [[1., 0., 0., 0., 0., ''],
             [0., 0., 0., 0., 1., ''],
             [0., 0., 0., 1., 0., ''],
             [0., 0., 0., 0., 0., ''],
             [0., 0., 0., 0., 0., ''],
             ['', '', '', '', '', 0.3333333333],
             ['', '', '', '', '', 0.3333333333],
             ['', '', '', '', '', 1.]],
            columns=['a', 'b', 'c', 'd', 'e', 'Overall Accuracy'],
            index=['a', 'b', 'c', 'd', 'e', 'Overall Accuracy',
                   'Baseline Accuracy', 'Accuracy Ratio'])
        predictions, confusion = _predict_and_plot(self.tmpd, self.a, b)
        pdt.assert_frame_equal(exp, predictions)

    def test_scatterplot(self):
        b = qiime2.NumericMetadataColumn(self.c)
        scatterplot(self.tmpd, self.c, b)

    def test_add_sample_size_to_xtick_labels(self):
        labels = _add_sample_size_to_xtick_labels(self.a, ['a', 'b', 'c'])
        exp = ['a (n=2)', 'b (n=2)', 'c (n=2)']
        self.assertListEqual(labels, exp)

    # now test performance when extra classes are present
    def test_add_sample_size_to_xtick_labels_extra_classes(self):
        labels = _add_sample_size_to_xtick_labels(
            self.a, [0, 'a', 'b', 'bb', 'c'])
        exp = ['0 (n=0)', 'a (n=2)', 'b (n=2)', 'bb (n=0)', 'c (n=2)']
        self.assertListEqual(labels, exp)

    def test_match_series_or_die(self):
        exp = pd.Series(['a', 'b', 'c'], name='site', index=['a1', 'b2', 'c2'])
        exp.index.name = 'SampleID'
        a, b = _match_series_or_die(self.a, self.bogus, 'ignore')
        pdt.assert_series_equal(exp, a)
        pdt.assert_series_equal(exp, b)

    def test_match_series_or_die_missing_samples(self):
        with self.assertRaisesRegex(ValueError, "Missing samples"):
            a, b = _match_series_or_die(self.a, self.bogus, 'error')
