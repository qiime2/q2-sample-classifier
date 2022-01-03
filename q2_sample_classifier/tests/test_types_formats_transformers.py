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
import shutil
import tempfile
import tarfile
import joblib
import sklearn
from sklearn.pipeline import Pipeline


import qiime2
from q2_types.feature_data import FeatureData
from qiime2.plugin import ValidationError
from q2_types.sample_data import SampleData

from q2_sample_classifier import (
    BooleanSeriesFormat, BooleanSeriesDirectoryFormat, BooleanSeries,
    PredictionsFormat, PredictionsDirectoryFormat, ClassifierPredictions,
    RegressorPredictions, ImportanceFormat, ImportanceDirectoryFormat,
    Importance, PickleFormat, ProbabilitiesFormat,
    ProbabilitiesDirectoryFormat, Probabilities, Classifier, Regressor,
    SampleEstimator, SampleEstimatorDirFmt,
    TrueTargetsDirectoryFormat, TrueTargets)
from q2_sample_classifier.visuals import (
    _custom_palettes, _plot_heatmap_from_confusion_matrix,)
from q2_sample_classifier._format import JSONFormat
from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase
from q2_sample_classifier.tests.test_estimators import SampleEstimatorTestBase


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
        obs = pd.read_csv(str(obs), sep='\t', header=0, index_col=0,
                          parse_dates=True)
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

    # test TrueTarget
    def test_TrueTargets_semantic_type_registration(self):
        self.assertRegisteredSemanticType(TrueTargets)

    # test TrueTargetDirectoryFormats
    def test_TrueTargets_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('true_targets.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = TrueTargetsDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_TrueTarget_to_TrueTargets_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[TrueTargets], TrueTargetsDirectoryFormat)


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
