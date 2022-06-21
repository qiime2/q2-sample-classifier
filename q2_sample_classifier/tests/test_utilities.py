# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import pandas as pd
import biom
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas.testing as pdt

import qiime2

from q2_sample_classifier.utilities import (
    _load_data, _calculate_feature_importances, _extract_important_features,
    _disable_feature_selection, _mean_feature_importance,
    _null_feature_importance, _extract_features)
from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase


class UtilitiesTests(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()

        exp_rf = pd.DataFrame(
            {'importance': [0.1, 0.2, 0.3]}, index=['a', 'b', 'c'])
        exp_rf.index.name = 'feature'
        self.exp_rf = exp_rf

        exp_svm = pd.DataFrame(
            {'importance0': [0.1, 0.2, 0.3], 'importance1': [0.4, 0.5, 0.6]},
            index=['a', 'b', 'c'])
        exp_svm.index.name = 'feature'
        self.exp_svm = exp_svm

        exp_lsvm = pd.DataFrame(
            {'importance0': [-0.048794, -0.048794, -0.048794]},
            index=['a', 'b', 'c'])
        exp_lsvm.index.name = 'feature'
        self.exp_lsvm = exp_lsvm

        self.features = biom.Table(np.array([[1]*5]*3), ['a', 'b', 'c'],
                                   list(map(str, range(5))))

        self.targets = pd.Series(['a', 'a', 'b', 'b', 'a'], name='bullseye')

    def test_extract_important_features_1d_array(self):
        importances = _extract_important_features(
            self.features.ids('observation'),
            np.ndarray((3,), buffer=np.array([0.1, 0.2, 0.3])))
        self.assertEqual(sorted(self.exp_rf), sorted(importances))

    def test_extract_important_features_2d_array(self):
        importances = _extract_important_features(
            self.features.ids('observation'),
            np.ndarray(
                (2, 3), buffer=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])))
        self.assertEqual(sorted(self.exp_svm), sorted(importances))

    # test feature importance calculation with main classifier types
    def test_calculate_feature_importances_ensemble(self):
        estimator = Pipeline(
            [('dv', DictVectorizer()),
             ('est', RandomForestClassifier(n_estimators=10))])
        estimator.fit(_extract_features(self.features),
                      self.targets.values.ravel())
        fi = _calculate_feature_importances(estimator)
        self.assertEqual(sorted(self.exp_rf), sorted(fi))

    def test_calculate_feature_importances_svm(self):
        estimator = Pipeline(
            [('dv', DictVectorizer()), ('est', LinearSVC())])
        estimator.fit(_extract_features(self.features),
                      self.targets.values.ravel())
        fi = _calculate_feature_importances(estimator)
        self.assertEqual(sorted(self.exp_lsvm), sorted(fi))

    # confirm that feature selection incompatibility warnings work
    def test_disable_feature_selection_unsupported(self):
        with self.assertWarnsRegex(UserWarning, "does not support recursive"):
            _disable_feature_selection('KNeighborsClassifier', False)

    def test_mean_feature_importance_1d_arrays(self):
        exp = pd.DataFrame([10., 9., 8., 7.], columns=["importance0"],
                           index=[3, 2, 1, 0])
        imps = [pd.DataFrame([1, 2, 3, 4], columns=["importance0"]),
                pd.DataFrame([5, 6, 7, 8], columns=["importance0"]),
                pd.DataFrame([9, 10, 11, 12], columns=["importance0"]),
                pd.DataFrame([13, 14, 15, 16], columns=["importance0"])]
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    def test_mean_feature_importance_different_column_names(self):
        exp = pd.DataFrame([[6., 5., 4., 3.], [14., 13., 12., 11.]],
                           index=["importance0", "importance1"],
                           columns=[3, 2, 1, 0]).T
        imps = [pd.DataFrame([1, 2, 3, 4], columns=["importance0"]),
                pd.DataFrame([5, 6, 7, 8], columns=["importance0"]),
                pd.DataFrame([9, 10, 11, 12], columns=["importance1"]),
                pd.DataFrame([13, 14, 15, 16], columns=["importance1"])]
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    def test_mean_feature_importance_2d_arrays(self):
        exp = pd.DataFrame([[3.5] * 4, [9.5] * 4],
                           index=["importance0", "importance1"],
                           columns=[0, 1, 2, 3]).T
        imps = [pd.DataFrame([[6, 5, 4, 3], [14, 13, 12, 11]],
                             index=["importance0", "importance1"],
                             columns=[0, 1, 2, 3]).T,
                pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                             index=["importance0", "importance1"],
                             columns=[0, 1, 2, 3]).T]
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    # and this should not occur now, but theoretically should just concat and
    # sort but not collapse if all column names are unique
    def test_mean_feature_importance_do_not_collapse(self):
        imps = [pd.DataFrame([4.0, 3.0, 2.0, 1.0], columns=["importance0"]),
                pd.DataFrame([16.0, 15.0, 14.0, 13.0],
                columns=["importance1"])]
        exp = pd.concat(imps, axis=1)
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    def test_null_feature_importance(self):
        exp = pd.DataFrame(
            [1, 1, 1], index=['o1', 'o2', 'o3'], columns=['importance'])
        exp.index.name = 'feature'
        tab = biom.Table(np.array([[1., 2., 3.], [3., 2., 1.], [7., 6., 9.]]),
                         ['o1', 'o2', 'o3'], ['s1', 's2', 's3'])
        tab = _extract_features(tab)
        pdt.assert_frame_equal(_null_feature_importance(tab), exp)

    def test_load_data(self):
        # phony feature table
        id_map = {'0': 'peanut', '1': 'bugs', '2': 'qiime2', '3': 'matt',
                  '4': 'pandas'}
        a = self.features.update_ids(id_map, axis='sample')
        # phony metadata, convert to qiime2.Metadata
        b = self.targets
        b.index = ['pandas', 'peanut', 'qiime1', 'flapjacks', 'bugs']
        b.index.name = '#SampleID'
        b = qiime2.Metadata(b.to_frame())
        # test that merge of tables is inner merge
        intersection = set(('peanut', 'bugs', 'pandas'))
        feature_data, targets = _load_data(a, b, missing_samples='ignore')
        exp = [{'c': 1.0, 'a': 1.0, 'b': 1.0}, {'c': 1.0, 'a': 1.0, 'b': 1.0},
               {'c': 1.0, 'a': 1.0, 'b': 1.0}]
        np.testing.assert_array_equal(feature_data, exp)
        self.assertEqual(set(targets.index), intersection)
