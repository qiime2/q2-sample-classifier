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
import biom
import shutil
import json
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import skbio

import qiime2
from q2_types.feature_table import (FeatureTable, PercentileNormalized)

from qiime2.plugins import sample_classifier
from q2_sample_classifier.tests.test_base_class import \
    SampleClassifierTestPluginBase
from q2_sample_classifier.classify import (
    regress_samples_ncv, classify_samples_ncv, fit_classifier, fit_regressor,
    detect_outliers, split_table, predict_classification,
    predict_regression)
from q2_sample_classifier.utilities import (
    _set_parameters_and_estimator, _train_adaboost_base_estimator,
    _match_series_or_die, _extract_features)
from q2_sample_classifier import (
    SampleEstimatorDirFmt, PickleFormat)


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
            index_col=0, names=['feature', 'importance'])
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
            distance_matrix=dm, metadata=metadata, k=1, cv=3, random_state=123
        )
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
            distance_matrix=dm, metadata=metadata, k=1, cv=3, random_state=123
        )
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
            distance_matrix=dm, metadata=metadata, k=2, cv=3, random_state=123
        )
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

    # test if training classifier with pipeline classify_samples raises
    # warning when test_size = 0.0
    def test_classify_samples_w_all_train_set(self):
        with self.assertWarnsRegex(Warning, "not representative of "
                                   "your model's performance"):
            table_fp = self.get_data_path('chardonnay.table.qza')
            table = qiime2.Artifact.load(table_fp)
            sample_classifier.actions.classify_samples(
                table=table, metadata=self.mdc_chard_fp,
                test_size=0.0, cv=1, n_estimators=10, n_jobs=1,
                estimator='RandomForestClassifier', random_state=123,
                parameter_tuning=False, optimize_feature_selection=False,
                missing_samples='ignore')

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
            ['blue', 'red', 'red', 'blue', 'blue',
             'blue', 'blue', 'red', 'blue', 'blue'],
            index=pd.Index(['s4', 's6', 's1', 's10', 's5', 's8', 's2', 's9',
                            's3', 's7'], dtype='object', name='SampleID'),
            name='prediction')
        exp_importances = pd.DataFrame(
            [0.595111111111111, 0.23155555555555551, 0.17333333333333334],
            index=pd.Index(['o3', 'o1', 'o2'], name='feature'),
            columns=['importance'])
        exp_probabilities = pd.DataFrame(
            [[0.5, 0.5], [0., 1.], [0., 1.], [0.5, 0.5], [0.5, 0.5],
             [0.5, 0.5], [0.5, 0.5], [0., 1.], [1., 0.], [1., 0.]],
            index=pd.Index(['s4', 's6', 's1', 's10', 's5', 's8', 's2', 's9',
                            's3', 's7'], name='SampleID'),
            columns=['blue', 'red'])
        pdt.assert_series_equal(y_pred, exp_pred)
        pdt.assert_frame_equal(importances, exp_importances)
        pdt.assert_frame_equal(probabilities, exp_probabilities)

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
            # TODO: Remove this conditional when
            # https://github.com/qiime2/q2-sample-classifier/issues/193 is
            # closed
            if regressor == 'Ridge':
                self.assertAlmostEqual(
                    accuracy, seeded_results[regressor], places=0,
                    msg='Accuracy of %s regressor was %f, but expected %f' % (
                        regressor, accuracy, seeded_results[regressor]))
            else:
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
        X_train, X_test, y_train, y_test = split_table(
            self.table_chard_fp, self.mdc_chard_fp, test_size=0.5,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()) + len(X_test.ids()), 21)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 21)

    def test_split_table_no_split(self):
        X_train, X_test, y_train, y_test = split_table(
            self.table_chard_fp, self.mdc_chard_fp, test_size=0.0,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()), 21)
        self.assertEqual(y_train.shape[0], 21)

    def test_split_table_invalid_test_size(self):
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            X_train, X_test, y_train, y_test = split_table(
                self.table_chard_fp, self.mdc_chard_fp, test_size=1.0,
                random_state=123, stratify=True, missing_samples='ignore')

    def test_split_table_percnorm(self):
        X_train, X_test, y_train, y_test = split_table(
            self.table_percnorm, self.mdc_percnorm, test_size=0.5,
            random_state=123, stratify=True, missing_samples='ignore')
        self.assertEqual(len(X_train.ids()) + len(X_test.ids()), 4)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 4)

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
            # verify predictions:
            # test that expected number of correct results is achieved (these
            # are mostly quite high as we would expect (total n=21))
            correct_results = np.sum(pred == exp)
            self.assertEqual(
                correct_results, seeded_predict_results[classifier],
                msg='Accuracy of %s classifier was %f, but expected %f' % (
                    classifier, correct_results,
                    seeded_predict_results[classifier]))
            # verify probabilities
            # test whether all are in correct range (0 to 1)
            ls_pred_classes = prob.columns.tolist()
            ls_correct_range = [col for col in ls_pred_classes if
                                prob[col].between(
                                    0, 1, inclusive=True).all()]
            self.assertEqual(len(ls_correct_range), prob.shape[1],
                             msg='Predicted probabilities of class {}'
                             'are not in range [0,1]'.format(
                [col for col in ls_pred_classes
                 if col not in ls_correct_range]))

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
            # TODO: Remove this conditional when
            # https://github.com/qiime2/q2-sample-classifier/issues/193 is
            # closed
            if regressor == 'Ridge':
                self.assertAlmostEqual(
                    mse, seeded_predict_results[regressor], places=4,
                    msg='Accuracy of %s regressor was %f, but expected %f' % (
                        regressor, mse, seeded_predict_results[regressor]))
            else:
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


seeded_results = {
    'RandomForestClassifier': 0.63636363636363635,
    'ExtraTreesClassifier': 0.454545454545,
    'GradientBoostingClassifier': 0.272727272727,
    'AdaBoostClassifier': 0.272727272727,
    'LinearSVC': 0.818182,
    'SVC': 0.36363636363636365,
    'KNeighborsClassifier': 0.363636363636,
    'RandomForestRegressor': 23.226508,
    'ExtraTreesRegressor': 19.725397,
    'GradientBoostingRegressor': 34.157100,
    'AdaBoostRegressor': 30.920635,
    'Lasso': 722.827623,
    'Ridge': 521.195194222418,
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
    'Ridge': 2.694020055323081e-05,
    'ElasticNet': 0.0614243397637,
    'KNeighborsRegressor': 26.8625396825,
    'SVR': 37.86704865859832,
    'LinearSVR': 0.0099912565770459132}
