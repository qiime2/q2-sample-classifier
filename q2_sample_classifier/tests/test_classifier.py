# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
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
    _two_way_anova, _pairwise_stats, _linear_regress,
    _calculate_baseline_accuracy, _custom_palettes,
    _plot_heatmap_from_confusion_matrix)
from q2_sample_classifier.classify import (
    classify_samples, regress_samples, regress_samples_ncv,
    classify_samples_ncv, fit_classifier, fit_regressor, maturity_index,
    detect_outliers)
from q2_sample_classifier.utilities import (
    split_optimize_classify, _set_parameters_and_estimator, _load_data,
    _calculate_feature_importances, _extract_important_features,
    _train_adaboost_base_estimator, _disable_feature_selection,
    _mean_feature_importance, _null_feature_importance, _extract_features)
from q2_sample_classifier import (
    BooleanSeriesFormat, BooleanSeriesDirectoryFormat, BooleanSeries,
    PredictionsFormat, PredictionsDirectoryFormat, Predictions,
    ImportanceFormat, ImportanceDirectoryFormat, Importance,
    SampleEstimatorDirFmt, PickleFormat, SampleEstimator)
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData
import pkg_resources
from qiime2.plugin.testing import TestPluginBase
from qiime2.plugin import ValidationError
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pandas.util.testing as pdt
import biom


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
            [('dv', DictVectorizer()), ('est', RandomForestClassifier())])
        estimator.fit(_extract_features(self.features), self.targets)
        fi = _calculate_feature_importances(estimator)
        self.assertEqual(sorted(self.exp_rf), sorted(fi))

    def test_calculate_feature_importances_svm(self):
        estimator = Pipeline(
            [('dv', DictVectorizer()), ('est', LinearSVC())])
        estimator.fit(_extract_features(self.features), self.targets)
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


class VisualsTests(SampleClassifierTestPluginBase):

    def test_two_way_anova(self):
        aov, mod_sum = _two_way_anova(tab1, md, 'Value', 'Time', 'Group')
        self.assertAlmostEqual(aov['PR(>F)']['Group'], 0.00013294988301061492)
        self.assertAlmostEqual(aov['PR(>F)']['Time'], 4.1672315658105502e-07)
        self.assertAlmostEqual(aov['PR(>F)']['Time:Group'], 0.0020603144625217)

    def test_pairwise_tests(self):
        res = _pairwise_stats(tab1, md, 'Value', 'Time', 'Group')
        self.assertAlmostEqual(
            res['q-value'][(1, 'a')][(1, 'b')], 0.066766544811987918)
        self.assertAlmostEqual(
            res['q-value'][(1, 'a')][(2, 'b')], 0.00039505928148818022)

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
        format.validate()

    def test_boolean_series_format_validate_negative(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'BooleanSeriesFormat'):
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
        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series([True, False, True, False],
                        name='outlier', index=exp_index)
        obs = transformer(exp)
        obs = pd.Series.from_csv(str(obs), sep='\t', header=0)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_pd_series(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, pd.Series, 'outliers.tsv')
        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series(['True', 'False', 'True', 'False'],
                        name='outlier', index=exp_index)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_metadata(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, qiime2.Metadata, 'outliers.tsv')

        exp_index = pd.Index(['a', 'b', 'c', 'd'], name='id')
        exp = pd.DataFrame([['True'], ['False'], ['True'], ['False']],
                           columns=['outlier'], index=exp_index, dtype='str')
        exp = qiime2.Metadata(exp)
        self.assertEqual(obs, exp)

    # test predictions format
    def test_Predictions_format_validate_positive(self):
        filepath = self.get_data_path('predictions.tsv')
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

    def test_Predictions_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Predictions)

    def test_sample_data_Predictions_to_Predictions_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[Predictions], PredictionsDirectoryFormat)

    def test_pd_series_to_Predictions_format(self):
        transformer = self.get_transformer(pd.Series, PredictionsFormat)
        exp = pd.Series([1, 2, 3, 4],
                        name='prediction', index=['a', 'b', 'c', 'd'])
        obs = transformer(exp)
        obs = pd.Series.from_csv(str(obs), sep='\t', header=0)
        pdt.assert_series_equal(obs[:4], exp)

    def test_Predictions_format_to_pd_series(self):
        _, obs = self.transform_format(
            PredictionsFormat, pd.Series, 'predictions.tsv')
        exp_index = pd.Index(['10249.C001.10SS', '10249.C002.05SS',
                              '10249.C004.01SS', '10249.C004.11SS'],
                             name='id', dtype=object)
        exp = pd.Series(['4.5', '2.5', '0.5', '4.5'], name='prediction',
                        index=exp_index, dtype=object)
        pdt.assert_series_equal(obs[:4], exp)

    def test_Predictions_format_to_metadata(self):
        _, obs = self.transform_format(
            PredictionsFormat, qiime2.Metadata, 'predictions.tsv')
        exp_index = pd.Index(['10249.C001.10SS', '10249.C002.05SS',
                              '10249.C004.01SS', '10249.C004.11SS'],
                             name='id')
        exp = pd.DataFrame([4.5, 2.5, 0.5, 4.5], columns=['prediction'],
                           index=exp_index, dtype='str')
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
        obs = pd.DataFrame.from_csv(str(obs), sep='\t', header=0)
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
                           index=exp_index, dtype='str')
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
                           index=exp_index, dtype='str')
        pdt.assert_frame_equal(obs.to_dataframe()[:4], exp)

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
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.Metadata(md)
            return md

        def _load_nmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.NumericMetadataColumn(md[column])
            return md

        def _load_cmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.CategoricalMetadataColumn(md[column])
            return md

        self.table_chard_fp = _load_biom('chardonnay.table.qza')
        self.md_chard_fp = _load_md('chardonnay.map.txt')
        self.mdc_chard_fp = _load_cmc('chardonnay.map.txt', 'Region')
        self.table_ecam_fp = _load_biom('ecam-table-maturity.qza')
        self.md_ecam_fp = _load_md('ecam_map_maturity.txt')
        self.mdc_ecam_fp = _load_nmc('ecam_map_maturity.txt', 'month')
        self.exp_imp = pd.DataFrame.from_csv(
            self.get_data_path('importance.tsv'), sep='\t')
        self.exp_pred = pd.Series.from_csv(
            self.get_data_path('predictions.tsv'), sep='\t', header=0)

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

    # test that the plugin/visualizer work
    def test_classify_samples(self):
        tmpd = join(self.temp_dir.name, 'RandomForestClassifier')
        mkdir(tmpd)
        classify_samples(tmpd, self.table_chard_fp, self.mdc_chard_fp,
                         test_size=0.5, cv=3,
                         n_estimators=2, n_jobs=1,
                         estimator='RandomForestClassifier',
                         parameter_tuning=True,
                         optimize_feature_selection=True,
                         missing_samples='ignore')

    # test that each classifier works and delivers an expected accuracy result
    # when a random seed is set.
    def test_classifiers(self):
        for classifier in ['RandomForestClassifier', 'ExtraTreesClassifier',
                           'GradientBoostingClassifier', 'AdaBoostClassifier',
                           'LinearSVC', 'SVC', 'KNeighborsClassifier']:
            tmpd = join(self.temp_dir.name, classifier)
            mkdir(tmpd)
            estimator, pad, pt = _set_parameters_and_estimator(
                classifier, self.table_chard_fp, self.md_chard_fp, 'Region',
                n_estimators=10, n_jobs=1, cv=1,
                random_state=123, parameter_tuning=False, classification=True,
                missing_samples='ignore')
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', estimator,
                tmpd, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False, missing_samples='ignore')
            self.assertAlmostEqual(accuracy, seeded_results[classifier])
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
        y_pred, importances = classify_samples_ncv(
            self.table_chard_fp, self.mdc_chard_fp, random_state=123,
            n_estimators=2, n_jobs=1, missing_samples='ignore')

    # test ncv a second time with KNeighborsRegressor (no feature importance)
    def test_regress_samples_ncv_knn(self):
        y_pred, importances = regress_samples_ncv(
            self.table_ecam_fp, self.mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, stratify=False, parameter_tuning=False,
            estimator='KNeighborsRegressor', missing_samples='ignore')

    def test_regress_samples(self):
        tmpd = join(self.temp_dir.name, 'RandomForestRegressor')
        mkdir(tmpd)
        regress_samples(tmpd, self.table_ecam_fp, self.mdc_ecam_fp,
                        test_size=0.5, cv=3,
                        n_estimators=2, n_jobs=1,
                        estimator='RandomForestRegressor',
                        missing_samples='ignore')

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
        exp_imp = pd.DataFrame.from_csv(
            self.get_data_path('importance_cv.tsv'), sep='\t')
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
            tmpd = join(self.temp_dir.name, regressor)
            mkdir(tmpd)
            estimator, pad, pt = _set_parameters_and_estimator(
                regressor, self.table_ecam_fp, self.md_ecam_fp, 'month',
                n_estimators=10, n_jobs=1, cv=1,
                random_state=123, parameter_tuning=False, classification=False,
                missing_samples='ignore')
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_ecam_fp, self.md_ecam_fp, 'month', estimator,
                tmpd, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None, classification=False,
                calc_feature_importance=False, scoring=mean_squared_error,
                missing_samples='ignore')
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
        # zero samples (if mapping file and table have no common samples)
        with self.assertRaisesRegex(ValueError, "metadata"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_ecam_fp, self.md_chard_fp, 'Region', estimator,
                self.temp_dir.name, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False, missing_samples='ignore')
        # too few samples to stratify
        with self.assertRaisesRegex(ValueError, "metadata"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', estimator,
                self.temp_dir.name, test_size=0.9, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False, missing_samples='ignore')
        # regressor chosen for classification problem
        with self.assertRaisesRegex(ValueError, "convert"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', regressor,
                self.temp_dir.name, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False, missing_samples='ignore')
        # metadata is a subset of feature table ids... raise error or else
        # an inner merge is taken, causing samples to be silently dropped!
        with self.assertRaisesRegex(ValueError, 'Missing samples'):
            md = self.md_chard_fp.filter_ids(self.md_chard_fp.ids[:5])
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, md, 'Region', regressor,
                self.temp_dir.name, missing_samples='error')

    # test experimental functions
    def test_maturity_index(self):
        maturity_index(self.temp_dir.name, self.table_ecam_fp, self.md_ecam_fp,
                       column='month', group_by='delivery', random_state=123,
                       n_jobs=1, control='Vaginal', test_size=0.4,
                       missing_samples='ignore')

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


class SampleEstimatorTestBase(SampleClassifierTestPluginBase):
    package = 'q2_sample_classifier.tests'

    def setUp(self):
        super().setUp()

        def _load_biom(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(biom.Table)
            return table

        def _load_nmc(md_fp, column):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.NumericMetadataColumn(md[column])
            return md

        table_ecam_fp = _load_biom('ecam-table-maturity.qza')
        mdc_ecam_fp = _load_nmc('ecam_map_maturity.txt', 'month')

        pipeline, importances = fit_classifier(
            table_ecam_fp, mdc_ecam_fp, random_state=123,
            n_estimators=2, n_jobs=1, optimize_feature_selection=True,
            parameter_tuning=True, missing_samples='ignore')
        transformer = self.get_transformer(
            Pipeline, SampleEstimatorDirFmt)
        self._sklp = transformer(pipeline)
        sklearn_pipeline = self._sklp.sklearn_pipeline.view(PickleFormat)
        self.sklearn_pipeline = str(sklearn_pipeline)

    def _custom_setup(self, version):
        with open(os.path.join(self.temp_dir.name,
                               'sklearn_version.json'), 'w') as fh:
            fh.write(json.dumps({'sklearn-version': version}))
        shutil.copy(self.sklearn_pipeline, self.temp_dir.name)
        return SampleEstimatorDirFmt(
            self.temp_dir.name, mode='r')


class TestTypes(SampleClassifierTestPluginBase):
    def test_taxonomic_classifier_semantic_type_registration(self):
        self.assertRegisteredSemanticType(SampleEstimator)

    def test_taxonomic_classifier_semantic_type_to_format_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleEstimator, SampleEstimatorDirFmt)


class TestFormats(SampleEstimatorTestBase):
    def test_taxonomic_classifier_dir_fmt(self):
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
    'SVC': 0.545454545455,
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
    'SVR': 72.6666666667}
