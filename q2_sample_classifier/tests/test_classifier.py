# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from warnings import filterwarnings
import pandas as pd
import numpy as np
import skbio
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pandas.testing as pdt
import biom

import qiime2
from qiime2.plugins import sample_classifier

from q2_sample_classifier.visuals import (
    _custom_palettes, _roc_palette, _roc_per_class, _roc_micro_average,
    _roc_macro_average, _binarize_labels, _generate_roc_plots)
from q2_sample_classifier.utilities import _extract_rfe_scores
from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=Warning)


class TestRFEExtractor(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        self.X = np.random.rand(50, 20)
        self.y = np.random.randint(0, 2, 50)

        self.exp1 = pd.Series([
            0.4999999999999999, 0.52, 0.52, 0.5399999999999999,
            0.44000000000000006, 0.52, 0.4600000000000001,
            0.5599999999999998, 0.52, 0.52, 0.5, 0.5399999999999999, 0.54,
            0.5599999999999999, 0.47999999999999987, 0.6199999999999999,
            0.5399999999999999, 0.5, 0.4999999999999999, 0.45999999999999996],
            index=pd.Index(range(1, 21)), name='Accuracy')
        self.exp2 = pd.Series([
            0.5000000000000001, 0.52, 0.48, 0.5599999999999998, 0.5,
            0.5799999999999998, 0.54, 0.4600000000000001, 0.6,
            0.45999999999999996, 0.45999999999999996],
            index=pd.Index([1] + [i for i in range(2, 21, 2)]),
            name='Accuracy')
        self.exp3 = pd.Series({1: 0.4600000000000001, 20: 0.45999999999999996},
                              name='Accuracy')

    def extract_rfe_scores_template(self, steps, expected):
        selector = RFECV(RandomForestClassifier(
            random_state=123, n_estimators=2), step=steps, cv=10)
        selector = selector.fit(self.X, self.y.ravel())
        pdt.assert_series_equal(
            _extract_rfe_scores(selector), expected)

    def test_extract_rfe_scores_step_int_one(self):
        self.extract_rfe_scores_template(1, self.exp1)

    def test_extract_rfe_scores_step_float_one(self):
        self.extract_rfe_scores_template(0.05, self.exp1)

    def test_extract_rfe_scores_step_int_two(self):
        self.extract_rfe_scores_template(2, self.exp2)

    def test_extract_rfe_scores_step_float_two(self):
        self.extract_rfe_scores_template(0.1, self.exp2)

    def test_extract_rfe_scores_step_full_range(self):
        self.extract_rfe_scores_template(20, self.exp3)

    def test_extract_rfe_scores_step_out_of_range(self):
        # should be equal to full_range
        self.extract_rfe_scores_template(21, self.exp3)


# test classifier pipelines succeed on binary data
class TestBinaryClassification(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        self.md = qiime2.CategoricalMetadataColumn(pd.Series(
            ['a', 'a', 'a', 'b', 'b', 'b'],
            index=pd.Index([c for c in 'abcdef'], name='id'), name='foo'))

        tab = biom.Table(np.array(
            [[13, 26, 37, 3, 6, 1], [33, 24, 23, 5, 6, 2],
             [38, 26, 33, 4, 1, 0], [3, 2, 1, 22, 25, 31],
             [2, 1, 3, 44, 46, 42]]),
            observation_ids=[c for c in 'vwxyz'],
            sample_ids=[c for c in 'abcdef'])
        self.tab = qiime2.Artifact.import_data('FeatureTable[Frequency]', tab)

        dist = skbio.DistanceMatrix.from_iterable(
            iterable=[1, 16, 2, 1, 16, 17],
            metric=lambda x, y: abs(y-x), keys=[c for c in 'abcdef']
        )
        self.dist = qiime2.Artifact.import_data('DistanceMatrix', dist)

    # we will make sure predictions are correct, but no need to validate
    # other outputs, which are tested elsewhere.
    def test_classify_samples_binary(self):
        res = sample_classifier.actions.classify_samples(
            table=self.tab, metadata=self.md,
            test_size=0.3, cv=1, n_estimators=2, n_jobs=1, random_state=123,
            parameter_tuning=False, optimize_feature_selection=False)
        exp = pd.Series(['a', 'b'], name='prediction',
                        index=pd.Index(['c', 'f'], name='id'))
        pdt.assert_series_equal(exp, res[2].view(pd.Series))

    def test_classify_samples_ncv_binary(self):
        res = sample_classifier.actions.classify_samples_ncv(
            table=self.tab, metadata=self.md, cv=3, n_estimators=2, n_jobs=1,
            random_state=123, parameter_tuning=False)
        exp = pd.Series([c for c in 'ababab'], name='prediction',
                        index=pd.Index([i for i in 'aebdcf'], name='id'))
        pdt.assert_series_equal(exp, res[0].view(pd.Series))

    def test_classify_samples_dist_binary(self):
        res = sample_classifier.actions.classify_samples_from_dist(
            distance_matrix=self.dist, metadata=self.md, k=2, cv=3,
            n_jobs=1, random_state=123)
        exp = pd.Series([c for c in 'abaaaa'], name='0',
                        index=pd.Index([i for i in 'abcdef'], name='id'))
        pdt.assert_series_equal(
            exp.sort_index(), res[0].view(pd.Series).sort_index()
        )


class TestROC(SampleClassifierTestPluginBase):
    def setUp(self):
        super().setUp()
        self.md = np.array(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
             [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
             [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
             [0, 0, 1], [0, 0, 1]])

        np.random.seed(0)
        probs = np.random.rand(20, 3)
        # probabilities should sum to 1 for each sample
        self.probs = np.apply_along_axis(
            lambda x: x / x.sum(), axis=1, arr=probs)

        self.exp_fpr = {0: [0., 0.07692308, 0.46153846, 0.46153846, 0.76923077,
                            0.76923077, 0.84615385, 0.84615385, 1., 1.],
                        1: [0., 0., 0.15384615, 0.15384615, 0.61538462,
                            0.61538462, 0.69230769, 0.69230769, 1., 1.],
                        2: [0., 0.07142857, 0.07142857, 0.14285714, 0.14285714,
                            0.78571429, 0.78571429, 0.92857143, 0.92857143,
                            1.]}
        self.exp_tdr = {0: [0., 0., 0., 0.57142857, 0.57142857, 0.71428571,
                            0.71428571, 0.85714286, 0.85714286, 1.],
                        1: [0., 0.14285714, 0.14285714, 0.28571429, 0.28571429,
                            0.57142857, 0.57142857, 0.85714286, 0.85714286,
                            1.],
                        2: [0., 0., 0.16666667, 0.16666667, 0.5, 0.5,
                            0.66666667, 0.66666667, 1., 1.]}
        self.exp_roc_auc = {0: 0.3626373626373626, 1: 0.4615384615384615,
                            2: 0.49999999999999994}

    # this test confirms that all palettes load properly.
    def test_roc_palette(self):
        [_roc_palette(p, 3) for p in _custom_palettes().keys()]

    def test_roc_per_class(self):
        fpr, tdr, roc_auc = _roc_per_class(self.md, self.probs, [0, 1, 2])
        for d, e in zip([fpr, tdr, roc_auc],
                        [self.exp_fpr, self.exp_tdr, self.exp_roc_auc]):
            for c in [0, 1, 2]:
                np.testing.assert_array_almost_equal(d[c], e[c])

    def test_roc_micro_average(self):
        fpr, tdr, roc_auc = _roc_micro_average(
            self.md, self.probs, self.exp_fpr, self.exp_tdr, self.exp_roc_auc)
        np.testing.assert_array_almost_equal(fpr['micro'], np.array(
            [0., 0.025, 0.025, 0.075, 0.075, 0.1, 0.1, 0.225, 0.225, 0.275,
             0.275, 0.475, 0.475, 0.575, 0.575, 0.6, 0.6, 0.65, 0.65, 0.675,
             0.675, 0.725, 0.725, 0.75, 0.75, 0.825, 0.825, 0.925, 0.925, 1.,
             1.]))
        np.testing.assert_array_almost_equal(tdr['micro'], np.array(
            [0., 0., 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25,
             0.35, 0.35, 0.4, 0.4, 0.45, 0.45, 0.5, 0.5, 0.55, 0.55, 0.6, 0.6,
             0.75, 0.75, 0.8, 0.8, 0.95, 0.95, 1.]))
        self.assertAlmostEqual(roc_auc['micro'], 0.41374999999999995)

    def test_roc_macro_average(self):
        fpr, tdr, roc_auc = _roc_macro_average(
            self.exp_fpr, self.exp_tdr, self.exp_roc_auc, [0, 1, 2])
        np.testing.assert_array_almost_equal(fpr['macro'], np.array(
            [0., 0.07142857, 0.07692308, 0.14285714, 0.15384615, 0.46153846,
             0.61538462, 0.69230769, 0.76923077, 0.78571429, 0.84615385,
             0.92857143, 1.]))
        np.testing.assert_array_almost_equal(tdr['macro'], np.array(
            [0.04761905, 0.1031746, 0.1031746, 0.21428571, 0.26190476,
             0.45238095, 0.54761905, 0.64285714, 0.69047619, 0.74603175,
             0.7936508, 0.90476191, 1.]))
        self.assertAlmostEqual(roc_auc['macro'], 0.49930228548098726)

    # Proves that the ROC nuts + bolts work if predictions does not have all
    # the classes present in probabilities. This will occur if there are many
    # classes or few samples and the data are not stratified:
    # https://github.com/qiime2/q2-sample-classifier/issues/171
    def test_binarize_and_roc_on_missing_classes(self):
        # seven samples with only 4 classes (adeh) of 8 possible classes
        # (abcdefgh) represented
        md = pd.Series([i for i in 'hedhadd'])
        # array of 7 samples X 8 classes
        # the values do not matter, only the labels
        probs = pd.DataFrame(np.random.rand(7, 8),
                             columns=[i for i in 'abcdefgh'])
        _generate_roc_plots(md, probs, 'GreenBlue')


class TestBinarize(SampleClassifierTestPluginBase):
    def setUp(self):
        super().setUp()

    def test_binarize_labels_binary(self):
        md = pd.Series([c for c in 'aabbaa'])
        labels = _binarize_labels(md, ['a', 'b'])
        exp = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]])
        np.testing.assert_array_equal(exp, labels)

    def test_binarize_labels_multiclass(self):
        md = pd.Series([c for c in 'abcabc'])
        labels = _binarize_labels(md, ['a', 'b', 'c'])
        exp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(exp, labels)
