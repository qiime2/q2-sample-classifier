# ----------------------------------------------------------------------------
# Copyright (c) 2017-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
from os import mkdir
from os.path import join
from warnings import filterwarnings
import tempfile
import shutil
import json
import tarfile

import qiime2
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from q2_sample_classifier.visuals import (
    _linear_regress, _calculate_baseline_accuracy, _custom_palettes,
    _plot_heatmap_from_confusion_matrix, _add_sample_size_to_xtick_labels)
from q2_sample_classifier.classify import (
    regress_samples_ncv, classify_samples_ncv, fit_classifier, fit_regressor,
    detect_outliers, split_table, predict_classification,
    predict_regression, scatterplot, confusion_matrix, summarize)
from q2_sample_classifier.utilities import (
    _set_parameters_and_estimator, _load_data,
    _calculate_feature_importances, _extract_important_features,
    _train_adaboost_base_estimator, _disable_feature_selection,
    _mean_feature_importance, _null_feature_importance, _extract_features,
    _match_series_or_die, _extract_rfe_scores, _predict_and_plot)
from q2_sample_classifier import (
    BooleanSeriesFormat, BooleanSeriesDirectoryFormat, BooleanSeries,
    PredictionsFormat, PredictionsDirectoryFormat, ClassifierPredictions,
    RegressorPredictions, ImportanceFormat, ImportanceDirectoryFormat,
    Importance, SampleEstimatorDirFmt, PickleFormat, SampleEstimator,
    Classifier, Regressor, ProbabilitiesFormat, ProbabilitiesDirectoryFormat,
    Probabilities)
from q2_sample_classifier._format import JSONFormat
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
from q2_types.feature_table import (FeatureTable, PercentileNormalized)
import pkg_resources
from qiime2.plugin.testing import TestPluginBase
from qiime2.plugin import ValidationError
from qiime2.plugins import sample_classifier, feature_table
import sklearn
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
import joblib
import pandas.util.testing as pdt
import biom
import skbio


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=Warning)
filterwarnings("ignore", category=ConvergenceWarning)


class SampleClassifierTestPluginBase(TestPluginBase):
    package = 'q2_sample_classifier.tests'

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='q2-sample-classifier-test-temp-')

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_data_path(self, filename):
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)


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
        exp = pd.DataFrame([10, 9, 8, 7], columns=["importance0"],
                           index=[3, 2, 1, 0])
        imps = [pd.DataFrame([1, 2, 3, 4], columns=["importance0"]),
                pd.DataFrame([5, 6, 7, 8], columns=["importance0"]),
                pd.DataFrame([9, 10, 11, 12], columns=["importance0"]),
                pd.DataFrame([13, 14, 15, 16], columns=["importance0"])]
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    def test_mean_feature_importance_different_column_names(self):
        exp = pd.DataFrame([[6, 5, 4, 3], [14, 13, 12, 11]],
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
                           columns=[3, 2, 1, 0]).T
        imps = [pd.DataFrame([[6, 5, 4, 3], [14, 13, 12, 11]],
                             index=["importance0", "importance1"],
                             columns=[3, 2, 1, 0]).T,
                pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                             index=["importance0", "importance1"],
                             columns=[3, 2, 1, 0]).T]
        pdt.assert_frame_equal(_mean_feature_importance(imps), exp)

    # and this should not occur now, but theoretically should just concat and
    # sort but not collapse if all column names are unique
    def test_mean_feature_importance_do_not_collapse(self):
        imps = [pd.DataFrame([4, 3, 2, 1], columns=["importance0"]),
                pd.DataFrame([16, 15, 14, 13], columns=["importance1"])]
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


class TestRFEExtractor(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        self.X = np.random.rand(50, 20)
        self.y = np.random.randint(0, 2, 50)

        self.exp1 = pd.Series([
            0.52, 0.61, 0.475, 0.49833333, 0.515, 0.51166667, 0.43166667,
            0.50666667, 0.61666667, 0.50333333, 0.58166667, 0.495, 0.51166667,
            0.465, 0.57833333, 0.57833333, 0.70166667, 0.45333333, 0.60666667,
            0.44166667], index=pd.Index(range(1, 21)), name='Accuracy')
        self.exp2 = pd.Series([
            0.39166666666666666, 0.47833333333333333, 0.5766666666666667,
            0.6066666666666667, 0.5366666666666667, 0.4, 0.5316666666666666,
            0.4, 0.57, 0.4533333333333333, 0.4416666666666666],
            index=pd.Index([1] + [i for i in range(2, 21, 2)]),
            name='Accuracy')
        self.exp3 = pd.Series({1: 0.38666667, 20: 0.44166667}, name='Accuracy')

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


class VisualsTests(SampleClassifierTestPluginBase):

    def test_linear_regress(self):
        res = _linear_regress(md['Value'], md['Time'])
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


class TestSemanticTypes(SampleClassifierTestPluginBase):

    def test_boolean_series_format_validate_positive(self):
        filepath = self.get_data_path('outliers.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        # These should both just succeed
        format.validate('min')
        format.validate('max')

    def test_boolean_series_format_validate_negative_col_count(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'BooleanSeriesFormat'):
            format.validate()

    def test_boolean_series_format_validate_negative_cell_values(self):
        filepath = self.get_data_path('predictions.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'BooleanSeriesFormat'):
            format.validate()

    def test_boolean_series_format_validate_negative_empty(self):
        filepath = self.get_data_path('empty_file.txt')
        format = BooleanSeriesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'one data record'):
            format.validate()

    def test_boolean_series_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('outliers.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = BooleanSeriesDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_boolean_series_semantic_type_registration(self):
        self.assertRegisteredSemanticType(BooleanSeries)

    def test_sample_data_boolean_series_to_boolean_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[BooleanSeries], BooleanSeriesDirectoryFormat)

    def test_pd_series_to_boolean_format(self):
        transformer = self.get_transformer(pd.Series, BooleanSeriesFormat)
        exp_index = pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], dtype=object)
        exp = pd.Series([True, False, True, False, True, False],
                        name='outlier', index=exp_index)
        obs = transformer(exp)
        obs = pd.read_csv(str(obs), sep='\t', header=0, index_col=0,
                          squeeze=True)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_pd_series(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, pd.Series, 'outliers.tsv')
        exp_index = pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], dtype=object)
        exp = pd.Series(['True', 'False', 'True', 'False', 'True', 'False'],
                        name='outlier', index=exp_index)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_metadata(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, qiime2.Metadata, 'outliers.tsv')

        exp_index = pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], name='id')
        exp = pd.DataFrame([['True'], ['False'], ['True'],
                            ['False'], ['True'], ['False']],
                           columns=['outlier'], index=exp_index, dtype='str')
        exp = qiime2.Metadata(exp)
        self.assertEqual(obs, exp)

    # test predictions format
    def test_Predictions_format_validate_positive_numeric_predictions(self):
        filepath = self.get_data_path('predictions.tsv')
        format = PredictionsFormat(filepath, mode='r')
        format.validate(level='min')
        format.validate()

    def test_Predictions_format_validate_positive_nonnumeric_predictions(self):
        filepath = self.get_data_path('categorical_predictions.tsv')
        format = PredictionsFormat(filepath, mode='r')
        format.validate(level='min')
        format.validate()

    def test_Predictions_format_validate_negative(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = PredictionsFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'PredictionsFormat'):
            format.validate()

    def test_Predictions_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('predictions.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = PredictionsDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_RegressorPredictions_semantic_type_registration(self):
        self.assertRegisteredSemanticType(RegressorPredictions)

    def test_ClassifierPredictions_semantic_type_registration(self):
        self.assertRegisteredSemanticType(ClassifierPredictions)

    def test_RegressorPredictions_to_Predictions_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[RegressorPredictions], PredictionsDirectoryFormat)

    def test_ClassifierPredictions_to_Predictions_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[ClassifierPredictions], PredictionsDirectoryFormat)

    def test_pd_series_to_Predictions_format(self):
        transformer = self.get_transformer(pd.Series, PredictionsFormat)
        exp = pd.Series([1, 2, 3, 4],
                        name='prediction', index=['a', 'b', 'c', 'd'])
        obs = transformer(exp)
        obs = pd.read_csv(str(obs), sep='\t', header=0, index_col=0,
                          squeeze=True)
        pdt.assert_series_equal(obs, exp)

    def test_pd_series_to_Predictions_format_allow_nans(self):
        transformer = self.get_transformer(pd.Series, PredictionsFormat)
        exp = pd.Series([1, np.nan, 3, np.nan],
                        name='prediction', index=['a', 'b', 'c', 'd'])
        obs = transformer(exp)
        obs = pd.read_csv(str(obs), sep='\t', header=0, index_col=0,
                          squeeze=True)
        pdt.assert_series_equal(obs, exp)

    def test_Predictions_format_to_pd_series(self):
        _, obs = self.transform_format(
            PredictionsFormat, pd.Series, 'predictions.tsv')
        exp_index = pd.Index(['10249.C001.10SS', '10249.C002.05SS',
                              '10249.C004.01SS', '10249.C004.11SS'],
                             name='id', dtype=object)
        exp = pd.Series([4.5, 2.5, 0.5, 4.5], name='prediction',
                        index=exp_index)
        pdt.assert_series_equal(obs[:4], exp)

    def test_Predictions_format_to_metadata(self):
        _, obs = self.transform_format(
            PredictionsFormat, qiime2.Metadata, 'predictions.tsv')
        exp_index = pd.Index(['10249.C001.10SS', '10249.C002.05SS',
                              '10249.C004.01SS', '10249.C004.11SS'],
                             name='id')
        exp = pd.DataFrame([4.5, 2.5, 0.5, 4.5], columns=['prediction'],
                           index=exp_index)
        pdt.assert_frame_equal(obs.to_dataframe()[:4], exp)

    # test Importance format
    def test_Importance_format_validate_positive(self):
        filepath = self.get_data_path('importance.tsv')
        format = ImportanceFormat(filepath, mode='r')
        format.validate(level='min')
        format.validate()

    def test_Importance_format_validate_negative_nonnumeric(self):
        filepath = self.get_data_path('chardonnay.map.txt')
        format = ImportanceFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'numeric values'):
            format.validate()

    def test_Importance_format_validate_negative_empty(self):
        filepath = self.get_data_path('empty_file.txt')
        format = ImportanceFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'one data record'):
            format.validate()

    def test_Importance_format_validate_negative(self):
        filepath = self.get_data_path('garbage.txt')
        format = ImportanceFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'two or more fields'):
            format.validate()

    def test_Importance_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('importance.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = ImportanceDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_Importance_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Importance)

    def test_sample_data_Importance_to_Importance_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            FeatureData[Importance], ImportanceDirectoryFormat)

    def test_pd_dataframe_to_Importance_format(self):
        transformer = self.get_transformer(pd.DataFrame, ImportanceFormat)
        exp = pd.DataFrame([1, 2, 3, 4],
                           columns=['importance'], index=['a', 'b', 'c', 'd'])
        obs = transformer(exp)
        obs = pd.read_csv(str(obs), sep='\t', header=0, index_col=0)
        pdt.assert_frame_equal(exp, obs)

    def test_Importance_format_to_pd_dataframe(self):
        _, obs = self.transform_format(
            ImportanceFormat, pd.DataFrame, 'importance.tsv')
        exp_index = pd.Index(['74ec9fe6ffab4ecff6d5def74298a825',
                              'c82032c40c98975f71892e4be561c87a',
                              '79280cea51a6fe8a3432b2f266dd34db',
                              'f7686a74ca2d3729eb66305e8a26309b'],
                             name='id')
        exp = pd.DataFrame([0.44469828320835586, 0.07760118417569697,
                            0.06570251750505914, 0.061718558716901406],
                           columns=['importance'],
                           index=exp_index)
        pdt.assert_frame_equal(exp, obs[:4])

    def test_Importance_format_to_metadata(self):
        _, obs = self.transform_format(
            ImportanceFormat, qiime2.Metadata, 'importance.tsv')
        exp_index = pd.Index(['74ec9fe6ffab4ecff6d5def74298a825',
                              'c82032c40c98975f71892e4be561c87a',
                              '79280cea51a6fe8a3432b2f266dd34db',
                              'f7686a74ca2d3729eb66305e8a26309b'],
                             name='id')
        exp = pd.DataFrame([0.44469828320835586, 0.07760118417569697,
                            0.06570251750505914, 0.061718558716901406],
                           columns=['importance'],
                           index=exp_index)
        pdt.assert_frame_equal(obs.to_dataframe()[:4], exp)

    # test Probabilities format
    def test_Probabilities_format_validate_positive(self):
        filepath = self.get_data_path('class_probabilities.tsv')
        format = ProbabilitiesFormat(filepath, mode='r')
        format.validate(level='min')
        format.validate()

    def test_Probabilities_format_validate_negative_nonnumeric(self):
        filepath = self.get_data_path('chardonnay.map.txt')
        format = ProbabilitiesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'numeric values'):
            format.validate()

    def test_Probabilities_format_validate_negative_empty(self):
        filepath = self.get_data_path('empty_file.txt')
        format = ProbabilitiesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'one data record'):
            format.validate()

    def test_Probabilities_format_validate_negative(self):
        filepath = self.get_data_path('garbage.txt')
        format = ProbabilitiesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'two or more fields'):
            format.validate()

    def test_Probabilities_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('class_probabilities.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = ProbabilitiesDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_Probabilities_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Probabilities)

    def test_sample_data_Probabilities_to_Probs_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[Probabilities], ProbabilitiesDirectoryFormat)

    def test_pd_dataframe_to_Probabilities_format(self):
        transformer = self.get_transformer(pd.DataFrame, ProbabilitiesFormat)
        exp = pd.DataFrame([[0.1, 0.77], [0.8, 0.4], [0.7, 0.1], [0.44, 0.73]],
                           columns=['classA', 'classB'],
                           index=['a', 'b', 'c', 'd'])
        obs = transformer(exp)
        obs = pd.DataFrame.from_csv(str(obs), sep='\t', header=0)
        pdt.assert_frame_equal(exp, obs)

    def test_Probabilities_format_to_pd_dataframe(self):
        _, obs = self.transform_format(
            ProbabilitiesFormat, pd.DataFrame, 'class_probabilities.tsv')
        exp_index = pd.Index(['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                             name='id')
        exp = pd.DataFrame([[0.4446, 0.9828, 0.3208],
                            [0.0776, 0.0118, 0.4175],
                            [0.0657, 0.0251, 0.7505],
                            [0.0617, 0.1855, 0.8716],
                            [0.0281, 0.8616, 0.0291],
                            [0.0261, 0.0253, 0.9075],
                            [0.0252, 0.7385, 0.4068]],
                           columns=['classA', 'classB', 'classC'],
                           index=exp_index)
        pdt.assert_frame_equal(exp, obs)

    def test_Probabilities_format_to_metadata(self):
        _, obs = self.transform_format(
            ProbabilitiesFormat, qiime2.Metadata, 'class_probabilities.tsv')
        exp_index = pd.Index(['s1', 's2', 's3', 's4', 's5', 's6', 's7'],
                             name='id')
        exp = pd.DataFrame([[0.4446, 0.9828, 0.3208],
                            [0.0776, 0.0118, 0.4175],
                            [0.0657, 0.0251, 0.7505],
                            [0.0617, 0.1855, 0.8716],
                            [0.0281, 0.8616, 0.0291],
                            [0.0261, 0.0253, 0.9075],
                            [0.0252, 0.7385, 0.4068]],
                           columns=['classA', 'classB', 'classC'],
                           index=exp_index)
        pdt.assert_frame_equal(obs.to_dataframe(), exp)

    # test utility formats
    def test_pickle_format_validate_negative(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = PickleFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'pickled file'):
            format.validate()

    def test_json_format_validate_negative(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = JSONFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'Expecting value'):
            format.validate()

    # this just checks that palette names are valid input
    def test_custom_palettes(self):
        confused = np.array([[1, 0], [0, 1]])
        for palette in _custom_palettes().keys():
            _plot_heatmap_from_confusion_matrix(confused, palette)


class EstimatorsTests(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()

        def _load_biom(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(biom.Table)
            return table

        def _load_md(md_fp):
            md_fp = self.get_data_path(md_fp)
            md = pd.read_csv(md_fp, sep='\t', header=0, index_col=0)
            md = qiime2.Metadata(md)
            return md

        def _load_nmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.read_csv(md_fp, sep='\t', header=0, index_col=0)
            md = qiime2.NumericMetadataColumn(md[column])
            return md

        def _load_cmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.read_csv(md_fp, sep='\t', header=0, index_col=0)
            md = qiime2.CategoricalMetadataColumn(md[column])
            return md

        self.table_chard_fp = _load_biom('chardonnay.table.qza')
        self.md_chard_fp = _load_md('chardonnay.map.txt')
        self.mdc_chard_fp = _load_cmc('chardonnay.map.txt', 'Region')
        self.table_ecam_fp = _load_biom('ecam-table-maturity.qza')
        self.md_ecam_fp = _load_md('ecam_map_maturity.txt')
        self.mdc_ecam_fp = _load_nmc('ecam_map_maturity.txt', 'month')
        self.exp_imp = pd.read_csv(
            self.get_data_path('importance.tsv'), sep='\t', header=0,
            index_col=0)
        self.exp_pred = pd.read_csv(
            self.get_data_path('predictions.tsv'), sep='\t', header=0,
            index_col=0, squeeze=True)
        index = pd.Index(['A', 'B', 'C', 'D'], name='id')
        self.table_percnorm = qiime2.Artifact.import_data(
            FeatureTable[PercentileNormalized], pd.DataFrame(
                [[20.0, 20.0, 50.0, 10.0], [10.0, 10.0, 70.0, 10.0],
                 [90.0, 8.0, 1.0, 1.0], [30.0, 15.0, 20.0, 35.0]],
                index=index,
                columns=['feat1', 'feat2', 'feat3', 'feat4'])).view(biom.Table)
        self.mdc_percnorm = qiime2.CategoricalMetadataColumn(
            pd.Series(['X', 'X', 'Y', 'Y'], index=index, name='name'))

    # test feature extraction
    def test_extract_features(self):
        table = self.table_ecam_fp
        dicts = _extract_features(table)
        dv = DictVectorizer()
        dv.fit(dicts)
        features = table.ids('observation')
        self.assertEqual(set(dv.get_feature_names()), set(features))
        self.assertEqual(len(dicts), len(table.ids()))
        for dict_row, (table_row, _, _) in zip(dicts, table.iter()):
            for feature, count in zip(features, table_row):
                if count == 0:
                    self.assertTrue(feature not in dict_row)
                else:
                    self.assertEqual(dict_row[feature], count)

    def test_classify_samples_from_dist(self):
        # -- setup -- #
        # 1,2 are a group, 3,4 are a group
        sample_ids = ('f1', 'f2', 's1', 's2')
        distance_matrix = skbio.DistanceMatrix([
            [0, 1, 4, 4],
            [1, 0, 4, 4],
            [4, 4, 0, 1],
            [4, 4, 1, 0],
            ], ids=sample_ids)

        dm = qiime2.Artifact.import_data('DistanceMatrix', distance_matrix)
        categories = pd.Series(('skinny', 'skinny', 'fat', 'fat'),
                               index=sample_ids[::-1], name='body_mass')
        categories.index.name = 'SampleID'
        metadata = qiime2.CategoricalMetadataColumn(categories)

        # -- test -- #
        res = sample_classifier.actions.classify_samples_from_dist(
            distance_matrix=dm, metadata=metadata, k=1)
        pred = res[0].view(pd.Series).sort_values()
        expected = pd.Series(('fat', 'skinny', 'fat', 'skinny'),
                             index=['f1', 's1', 'f2', 's2'])
        not_expected = pd.Series(('fat', 'fat', 'fat', 'skinny'),
                                 index=sample_ids)

        # order matters for pd.Series.equals()
        self.assertTrue(expected.sort_index().equals(pred.sort_index()))
        self.assertFalse(not_expected.sort_index().equals(pred.sort_index()))

    def test_classify_samples_from_dist_with_group_of_single_item(self):
        # -- setup -- #
        # 1 is a group, 2,3,4 are a group
        sample_ids = ('f1', 's1', 's2', 's3')
        distance_matrix = skbio.DistanceMatrix([
            [0, 2, 3, 3],
            [2, 0, 1, 1],
            [3, 1, 0, 1],
            [3, 1, 1, 0],
            ], ids=sample_ids)

        dm = qiime2.Artifact.import_data('DistanceMatrix', distance_matrix)
        categories = pd.Series(('fat', 'skinny', 'skinny', 'skinny'),
                               index=sample_ids, name='body_mass')
        categories.index.name = 'SampleID'
        metadata = qiime2.CategoricalMetadataColumn(categories)

        # -- test -- #
        res = sample_classifier.actions.classify_samples_from_dist(
            distance_matrix=dm, metadata=metadata, k=1)
        pred = res[0].view(pd.Series)
        expected = pd.Series(('skinny', 'skinny', 'skinny', 'skinny'),
                             index=sample_ids)

        self.assertTrue(expected.sort_index().equals(pred.sort_index()))

    def test_2nn(self):
        # -- setup -- #
        # 2 nearest neighbors of each sample are
        # f1: s1, s2 (classified as skinny)
        # s1: f1, s2 (closer to f1 so fat)
        # s2: f1, (s1 or s3) (closer to f1 so fat)
        # s3: s1, s2 (skinny)
        sample_ids = ('f1', 's1', 's2', 's3')
        distance_matrix = skbio.DistanceMatrix([
            [0, 2, 1, 5],
            [2, 0, 3, 4],
            [1, 3, 0, 3],
            [5, 4, 3, 0],
            ], ids=sample_ids)

        dm = qiime2.Artifact.import_data('DistanceMatrix', distance_matrix)
        categories = pd.Series(('fat', 'skinny', 'skinny', 'skinny'),
                               index=sample_ids, name='body_mass')
        categories.index.name = 'SampleID'
        metadata = qiime2.CategoricalMetadataColumn(categories)

        # -- test -- #
        res = sample_classifier.actions.classify_samples_from_dist(
            distance_matrix=dm, metadata=metadata, k=2)
        pred = res[0].view(pd.Series)
        expected = pd.Series(('skinny', 'fat', 'fat', 'skinny'),
                             index=sample_ids)
        self.assertTrue(expected.sort_index().equals(pred.sort_index()))

    # test that each classifier works and delivers an expected accuracy result
    # when a random seed is set.
    def test_classifiers(self):
        for classifier in ['RandomForestClassifier', 'ExtraTreesClassifier',
                           'GradientBoostingClassifier', 'AdaBoostClassifier',
                           'LinearSVC', 'SVC', 'KNeighborsClassifier']:
            table_fp = self.get_data_path('chardonnay.table.qza')
            table = qiime2.Artifact.load(table_fp)
            res = sample_classifier.actions.classify_samples(
                table=table, metadata=self.mdc_chard_fp,
                test_size=0.5, cv=1, n_estimators=10, n_jobs=1,
                estimator=classifier, random_state=123,
                parameter_tuning=False, optimize_feature_selection=False,
                missing_samples='ignore')
            pred = res[2].view(pd.Series)
            pred, truth = _match_series_or_die(
                pred, self.mdc_chard_fp.to_series(), 'ignore')
            accuracy = accuracy_score(truth, pred)
            self.assertAlmostEqual(
                accuracy, seeded_results[classifier], places=4,
                msg='Accuracy of %s classifier was %f, but expected %f' % (
                    classifier, accuracy, seeded_results[classifier]))

    # test that the plugin methods/visualizers work
    def test_regress_samples_ncv(self):
        y_pred, importances = regress_samples_ncv(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, stratify=True, parameter_tuning=True,
            missing_samples='ignore')

    def test_classify_samples_ncv(self):
        y_pred, importances, probabilities = classify_samples_ncv(
            self.table_chard_fp, self.mdc_chard_fp, random_state=123,
            n_estimators=2, n_jobs=1, missing_samples='ignore')

    # test reproducibility of classifier results, probabilities
    def test_classify_samples_ncv_accuracy(self):
        dat = biom.Table(np.array(
            [[4446, 9828, 3208, 776, 118, 4175, 657, 251, 7505, 617],
             [1855, 8716, 3257, 1251, 3205, 2557, 4251, 7405, 1417, 1215],
             [6616, 281, 8616, 291, 261, 253, 9075, 252, 7385, 4068]]),
            observation_ids=['o1', 'o2', 'o3'],
            sample_ids=['s1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10'])
        md = qiime2.CategoricalMetadataColumn(pd.Series(
            ['red', 'red', 'red', 'red', 'red',
             'blue', 'blue', 'blue', 'blue', 'blue'],
            index=pd.Index(['s1', 's2', 's3', 's4', 's5',
                            's6', 's7', 's8', 's9', 's10'],
                           name='sample-id'), name='color'))
        y_pred, importances, probabilities = classify_samples_ncv(
            dat, md, random_state=123, n_estimators=2, n_jobs=1,
            missing_samples='ignore')
        exp_pred = pd.Series(
            ['blue', 'blue', 'blue', 'red', 'blue',
             'blue', 'blue', 'red', 'red', 'blue'],
            index=pd.Index(['s1', 's7', 's5', 's9', 's3', 's10', 's4', 's6',
                            's2', 's8'], dtype='object', name='SampleID'),
            name='prediction')
        exp_importances = pd.DataFrame(
            [0.5551111111111111, 0.2671111111111111, 0.1777777777777778],
            index=pd.Index(['o3', 'o1', 'o2']), columns=['importance'])
        exp_probabilities = pd.DataFrame(
            [[1., 0.], [1., 0.], [1., 0.], [0., 1.], [0.5, 0.5], [0.5, 0.5],
             [0.5, 0.5], [0., 1.], [0., 1.], [0.5, 0.5]],
            index=pd.Index(['s1', 's7', 's5', 's9', 's3', 's10', 's4', 's6',
                            's2', 's8'], name='SampleID'),
            columns=['blue', 'red'])
        # order does not matter for predictions or probabilities, so align
        pdt.assert_series_equal(*y_pred.align(exp_pred, axis=0))
        # order matters for importances, so do not align
        pdt.assert_frame_equal(importances, exp_importances)
        pdt.assert_frame_equal(*probabilities.align(exp_probabilities, axis=0))

    # test ncv a second time with KNeighborsRegressor (no feature importance)
    def test_regress_samples_ncv_knn(self):
        y_pred, importances = regress_samples_ncv(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, stratify=False, parameter_tuning=False,
            estimator='KNeighborsRegressor', missing_samples='ignore')

    # test that ncv gives expected results
    def test_regress_samples_ncv_accuracy(self):
        y_pred, importances = regress_samples_ncv(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, missing_samples='ignore')
        pdt.assert_series_equal(y_pred, self.exp_pred)
        pdt.assert_frame_equal(importances, self.exp_imp)

    # test that fit_* methods output consistent importance scores
    def test_fit_regressor(self):
        pipeline, importances = fit_regressor(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, missing_samples='ignore')
        exp_imp = pd.read_csv(
            self.get_data_path('importance_cv.tsv'), sep='\t', header=0,
            index_col=0)
        pdt.assert_frame_equal(importances, exp_imp)

    # just make sure this method runs. Uses the same internal function as
    # fit_regressor, so importance score consistency is covered by the above
    # test.
    def test_fit_classifier(self):
        pipeline, importances = fit_classifier(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, optimize_feature_selection=True,
            parameter_tuning=True, missing_samples='ignore')

    # test that each regressor works and delivers an expected accuracy result
    # when a random seed is set.
    def test_regressors(self):
        for regressor in ['RandomForestRegressor', 'ExtraTreesRegressor',
                          'GradientBoostingRegressor', 'AdaBoostRegressor',
                          'Lasso', 'Ridge', 'ElasticNet',
                          'KNeighborsRegressor', 'LinearSVR', 'SVR']:
            table_fp = self.get_data_path('ecam-table-maturity.qza')
            table = qiime2.Artifact.load(table_fp)
            res = sample_classifier.actions.regress_samples(
                table=table, metadata=self.mdc_ecam_fp,
                test_size=0.5, cv=1, n_estimators=10, n_jobs=1,
                estimator=regressor, random_state=123,
                parameter_tuning=False, optimize_feature_selection=False,
                missing_samples='ignore', stratify=True)
            pred = res[2].view(pd.Series)
            pred, truth = _match_series_or_die(
                pred, self.mdc_ecam_fp.to_series(), 'ignore')
            accuracy = mean_squared_error(truth, pred)
            self.assertAlmostEqual(
                accuracy, seeded_results[regressor], places=4,
                msg='Accuracy of %s regressor was %f, but expected %f' % (
                    regressor, accuracy, seeded_results[regressor]))

    # test adaboost base estimator trainer
    def test_train_adaboost_base_estimator(self):
        abe = _train_adaboost_base_estimator(
            self.table_chard_fp, self.mdc_chard_fp, 'Region',
            n_estimators=10, n_jobs=1, cv=3, random_state=None,
            parameter_tuning=True, classification=True,
            missing_samples='ignore')
        self.assertEqual(type(abe.named_steps.est), AdaBoostClassifier)

    # test some invalid inputs/edge cases
    def test_invalids(self):
        estimator, pad, pt = _set_parameters_and_estimator(
            'RandomForestClassifier', self.table_chard_fp, self.md_chard_fp,
            'Region', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=True,
            missing_samples='ignore')
        regressor, pad, pt = _set_parameters_and_estimator(
            'RandomForestRegressor', self.table_chard_fp, self.md_chard_fp,
            'Region', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=True,
            missing_samples='ignore')

    def test_split_table_no_rounding_error(self):
        X_train, X_test = split_table(
            self.table_chard_fp, self.mdc_chard_fp, test_size=0.5,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()) + len(X_test.ids()), 21)

    def test_split_table_no_split(self):
        X_train, X_test = split_table(
            self.table_chard_fp, self.mdc_chard_fp, test_size=0.0,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()), 21)

    def test_split_table_invalid_test_size(self):
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            X_train, X_test = split_table(
                self.table_chard_fp, self.mdc_chard_fp, test_size=1.0,
                random_state=123, stratify=True, missing_samples='ignore')

    def test_split_table_percnorm(self):
        X_train, X_test = split_table(
            self.table_percnorm, self.mdc_percnorm, test_size=0.5,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()) + len(X_test.ids()), 4)

    # test experimental functions
    def test_detect_outliers(self):
        detect_outliers(self.table_chard_fp, self.md_chard_fp,
                        random_state=123, n_jobs=1, contamination=0.05)

    def test_detect_outliers_with_subsets(self):
        detect_outliers(self.table_chard_fp, self.md_chard_fp,
                        random_state=123, n_jobs=1, contamination=0.05,
                        subset_column='Vineyard', subset_value=1)

    def test_detect_outliers_raise_error_on_missing_subset_data(self):
        with self.assertRaisesRegex(ValueError, "must both be provided"):
            detect_outliers(self.table_chard_fp, self.md_chard_fp,
                            random_state=123, n_jobs=1, contamination=0.05,
                            subset_column='Vineyard', subset_value=None)
        with self.assertRaisesRegex(ValueError, "must both be provided"):
            detect_outliers(self.table_chard_fp, self.md_chard_fp,
                            random_state=123, n_jobs=1, contamination=0.05,
                            subset_column=None, subset_value=1)

    # just test that this works by making sure a classifier trained on samples
    # x, y, and z predicts the correct metadata values for those same samples.
    def test_predict_classifications(self):
        for classifier in ['RandomForestClassifier', 'ExtraTreesClassifier',
                           'GradientBoostingClassifier', 'AdaBoostClassifier',
                           'LinearSVC', 'SVC', 'KNeighborsClassifier']:
            estimator, importances = fit_classifier(
                self.table_chard_fp, self.mdc_chard_fp, random_state=123,
                n_estimators=2, estimator=classifier, n_jobs=1,
                missing_samples='ignore')
            pred, prob = predict_classification(self.table_chard_fp, estimator)
            exp = self.mdc_chard_fp.to_series().reindex(pred.index).dropna()
            # reindex both pred and exp because not all samples present in pred
            # are present in the metadata! (hence missing_samples='ignore')
            sample_ids = pred.index.intersection(exp.index)
            pred = pred.loc[sample_ids]
            exp = exp.loc[sample_ids]
            # test that expected number of correct results is achieved (these
            # are mostly quite high as we would expect (total n=21))
            correct_results = np.sum(pred == exp)
            self.assertEqual(
                correct_results, seeded_predict_results[classifier],
                msg='Accuracy of %s classifier was %f, but expected %f' % (
                    classifier, correct_results,
                    seeded_predict_results[classifier]))

    def test_predict_regressions(self):
        for regressor in ['RandomForestRegressor', 'ExtraTreesRegressor',
                          'GradientBoostingRegressor', 'AdaBoostRegressor',
                          'Lasso', 'Ridge', 'ElasticNet',
                          'KNeighborsRegressor', 'SVR', 'LinearSVR']:
            estimator, importances = fit_regressor(
                self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
                n_estimators=2, estimator=regressor, n_jobs=1,
                missing_samples='ignore')
            pred = predict_regression(self.table_ecam_fp, estimator)
            exp = self.mdc_ecam_fp.to_series()
            # reindex both pred and exp because not all samples present in pred
            # are present in the metadata! (hence missing_samples='ignore')
            sample_ids = pred.index.intersection(exp.index)
            pred = pred.loc[sample_ids]
            exp = exp.loc[sample_ids]
            # test that expected MSE is achieved (these are mostly quite high
            # as we would expect)
            mse = mean_squared_error(exp, pred)
            self.assertAlmostEqual(
                mse, seeded_predict_results[regressor],
                msg='Accuracy of %s regressor was %f, but expected %f' % (
                    regressor, mse, seeded_predict_results[regressor]))

    # make sure predict still works when features are given in a different
    # order from training set.
    def test_predict_feature_order_aint_no_thing(self):
        table = self.table_ecam_fp
        estimator, importances = fit_regressor(
            table, self.mdc_ecam_fp, random_state=123, n_estimators=2,
            n_jobs=1, missing_samples='ignore')

        # randomly shuffle and reorder features in biom table.
        feature_ids = table.ids(axis='observation')
        # look ma no seed! we should get the same result no matter the order.
        np.random.shuffle(feature_ids)
        shuffled_table = table.sort_order(feature_ids, axis='observation')

        # now predict values on shuffled data
        pred = predict_regression(shuffled_table, estimator)
        exp = self.mdc_ecam_fp.to_series()
        # reindex both pred and exp because not all samples present in pred
        # are present in the metadata! (hence missing_samples='ignore')
        sample_ids = pred.index.intersection(exp.index)
        pred = pred.loc[sample_ids]
        exp = exp.loc[sample_ids]
        # test that expected MSE is achieved (these are mostly quite high
        # as we would expect)
        mse = mean_squared_error(exp, pred)
        self.assertAlmostEqual(
            mse, seeded_predict_results['RandomForestRegressor'])


class TestHeatmap(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()
        md_ecam = self.get_data_path('ecam_map_maturity.txt')
        md_ecam = qiime2.Metadata.load(md_ecam)
        self.md_ecam = md_ecam.get_column('delivery')
        table_ecam = self.get_data_path('ecam-table-maturity.qza')
        table_ecam = qiime2.Artifact.load(table_ecam)
        self.table_ecam, = feature_table.actions.filter_samples(
            table_ecam, metadata=md_ecam)
        imp = pd.read_csv(
            self.get_data_path('importance.tsv'), sep='\t', header=0,
            index_col=0)
        self.imp = qiime2.Artifact.import_data('FeatureData[Importance]', imp)

    def test_heatmap_default_feature_count_zero(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_ecam, self.imp, self.md_ecam, group_samples=True,
            feature_count=0)
        self.assertEqual(table.view(biom.Table).shape, (1056, 2))

    def test_heatmap_importance_threshold(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_ecam, self.imp, self.md_ecam,
            importance_threshold=0.017, group_samples=False, feature_count=0)
        self.assertEqual(table.view(biom.Table).shape, (10, 121))

    def test_heatmap_feature_count(self):
        heatmap, table, = sample_classifier.actions.heatmap(
            self.table_ecam, self.imp, self.md_ecam, group_samples=True,
            feature_count=20)
        self.assertEqual(table.view(biom.Table).shape, (20, 2))

    def test_heatmap_must_group_or_die(self):
        with self.assertRaisesRegex(ValueError, "metadata are not optional"):
            heatmap, table, = sample_classifier.actions.heatmap(
                self.table_ecam, self.imp, metadata=None, group_samples=True)


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

    # let's make sure the correct transformers are in place! See issue 114
    # if this runs without error, that's good enough for me. We already
    # validate the function above.
    def test_action_split_table(self):
        sample_classifier.actions.split_table(self.tab, self.md, test_size=0.5)

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

    def test_metatable_empty_metadata_after_filtering(self):
        with self.assertRaisesRegex(
                ValueError, "All metadata"):  # are belong to us
            sample_classifier.actions.metatable(
                self.md2.filter_ids(['b', 'c']), self.tab,
                missing_values='drop_samples')

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


class SampleEstimatorTestBase(SampleClassifierTestPluginBase):
    package = 'q2_sample_classifier.tests'

    def setUp(self):
        super().setUp()

        def _load_biom(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(biom.Table)
            return table

        def _load_cmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.read_csv(md_fp, sep='\t', header=0, index_col=0)
            md = qiime2.CategoricalMetadataColumn(md[column])
            return md

        table_chard_fp = _load_biom('chardonnay.table.qza')
        mdc_chard_fp = _load_cmc('chardonnay.map.txt', 'Region')

        pipeline, importances = fit_classifier(
            table_chard_fp, mdc_chard_fp, random_state=123,
            n_estimators=2, n_jobs=1, optimize_feature_selection=True,
            parameter_tuning=True, missing_samples='ignore')
        transformer = self.get_transformer(
            Pipeline, SampleEstimatorDirFmt)
        self._sklp = transformer(pipeline)
        sklearn_pipeline = self._sklp.sklearn_pipeline.view(PickleFormat)
        self.sklearn_pipeline = str(sklearn_pipeline)
        self.pipeline = pipeline

    def _custom_setup(self, version):
        with open(os.path.join(self.temp_dir.name,
                               'sklearn_version.json'), 'w') as fh:
            fh.write(json.dumps({'sklearn-version': version}))
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        return SampleEstimatorDirFmt(
            self.temp_dir.name, mode='r')


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


class TestTypes(SampleClassifierTestPluginBase):
    def test_sample_estimator_semantic_type_registration(self):
        self.assertRegisteredSemanticType(SampleEstimator)

    def test_classifier_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Classifier)

    def test_regressor_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Regressor)

    def test_sample_classifier_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleEstimator[Classifier], SampleEstimatorDirFmt)

    def test_sample_regressor_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleEstimator[Regressor], SampleEstimatorDirFmt)


class TestFormats(SampleEstimatorTestBase):
    def test_sample_classifier_dir_fmt(self):
        format = self._custom_setup(sklearn.__version__)

        # Should not error
        format.validate()


class TestTransformers(SampleEstimatorTestBase):
    def test_old_sklearn_version(self):
        transformer = self.get_transformer(
            SampleEstimatorDirFmt, Pipeline)
        input = self._custom_setup('a very old version')
        with self.assertRaises(ValueError):
            transformer(input)

    def test_taxo_class_dir_fmt_to_taxo_class_result(self):
        input = self._custom_setup(sklearn.__version__)

        transformer = self.get_transformer(
            SampleEstimatorDirFmt, Pipeline)
        obs = transformer(input)

        self.assertTrue(obs)

    def test_taxo_class_result_to_taxo_class_dir_fmt(self):
        def read_pipeline(pipeline_filepath):
            with tarfile.open(pipeline_filepath) as tar:
                dirname = tempfile.mkdtemp()
                tar.extractall(dirname)
                pipeline = joblib.load(os.path.join(dirname,
                                       'sklearn_pipeline.pkl'))
                for fn in tar.getnames():
                    os.unlink(os.path.join(dirname, fn))
                os.rmdir(dirname)
            return pipeline

        exp = read_pipeline(self.sklearn_pipeline)
        transformer = self.get_transformer(
            Pipeline, SampleEstimatorDirFmt)
        obs = transformer(exp)
        sklearn_pipeline = obs.sklearn_pipeline.view(PickleFormat)
        obs_pipeline = read_pipeline(str(sklearn_pipeline))
        obs = obs_pipeline
        self.assertTrue(obs)


# make sure summarize visualizer works and that rfe_scores are stored properly
class TestSummarize(SampleEstimatorTestBase):

    def test_summary_with_rfecv(self):
        summarize(self.temp_dir.name, self.pipeline)

    def test_summary_without_rfecv(self):
        del self.pipeline.rfe_scores
        summarize(self.temp_dir.name, self.pipeline)


md = pd.DataFrame([(1, 'a', 0.11), (1, 'a', 0.12), (1, 'a', 0.13),
                   (2, 'a', 0.19), (2, 'a', 0.18), (2, 'a', 0.21),
                   (1, 'b', 0.14), (1, 'b', 0.13), (1, 'b', 0.14),
                   (2, 'b', 0.26), (2, 'b', 0.27), (2, 'b', 0.29)],
                  columns=['Time', 'Group', 'Value'])

tab1 = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], columns=['Junk'])

seeded_results = {
    'RandomForestClassifier': 0.63636363636363635,
    'ExtraTreesClassifier': 0.454545454545,
    'GradientBoostingClassifier': 0.272727272727,
    'AdaBoostClassifier': 0.272727272727,
    'LinearSVC': 0.727272727273,
    'SVC': 0.36363636363636365,
    'KNeighborsClassifier': 0.363636363636,
    'RandomForestRegressor': 23.226508,
    'ExtraTreesRegressor': 19.725397,
    'GradientBoostingRegressor': 34.157100,
    'AdaBoostRegressor': 30.920635,
    'Lasso': 722.827623,
    'Ridge': 123.625210,
    'ElasticNet': 618.532273,
    'KNeighborsRegressor': 44.7847619048,
    'LinearSVR': 511.816385601,
    'SVR': 51.325146}

seeded_predict_results = {
    'RandomForestClassifier': 18,
    'ExtraTreesClassifier': 21,
    'GradientBoostingClassifier': 21,
    'AdaBoostClassifier': 21,
    'LinearSVC': 21,
    'SVC': 12,
    'KNeighborsClassifier': 14,
    'RandomForestRegressor': 7.4246031746,
    'ExtraTreesRegressor': 0.,
    'GradientBoostingRegressor': 50.1955883469,
    'AdaBoostRegressor': 9.7857142857142865,
    'Lasso': 0.173138653701,
    'Ridge': 7.57617215386,
    'ElasticNet': 0.0614243397637,
    'KNeighborsRegressor': 26.8625396825,
    'SVR': 37.86704865859832,
    'LinearSVR': 0.0099912565770459132}
