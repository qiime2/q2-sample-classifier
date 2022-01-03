# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import tarfile
import json

import pandas as pd
import numpy as np
import qiime2
import qiime2.plugin.model as model
import sklearn
import joblib
from sklearn.pipeline import Pipeline

from .plugin_setup import plugin
from ._format import (SampleEstimatorDirFmt, JSONFormat, BooleanSeriesFormat,
                      ImportanceFormat, PredictionsFormat, PickleFormat,
                      ProbabilitiesFormat)


def _read_dataframe(fh):
    # Using `dtype=object` and `set_index` to avoid type casting/inference
    # of any columns or the index.
    df = pd.read_csv(fh, sep='\t', header=0, dtype='str')
    df.set_index(df.columns[0], drop=True, append=False, inplace=True)
    df.index.name = 'id'
    return df


@plugin.register_transformer
def _1(data: pd.Series) -> (BooleanSeriesFormat):
    ff = BooleanSeriesFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _2(ff: BooleanSeriesFormat) -> (pd.Series):
    with ff.open() as fh:
        df = _read_dataframe(fh)
        return df.iloc[:, 0]


@plugin.register_transformer
def _3(ff: BooleanSeriesFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh))


@plugin.register_transformer
def _4(data: pd.Series) -> (PredictionsFormat):
    ff = PredictionsFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True)
    return ff


@plugin.register_transformer
def _5(ff: PredictionsFormat) -> (pd.Series):
    with ff.open() as fh:
        df = _read_dataframe(fh)
        return pd.to_numeric(df.iloc[:, 0], errors='ignore')


@plugin.register_transformer
def _6(ff: PredictionsFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh).apply(
            lambda x: pd.to_numeric(x, errors='ignore')))


@plugin.register_transformer
def _7(data: pd.DataFrame) -> (ImportanceFormat):
    ff = ImportanceFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', header=True, na_rep=np.nan)
    return ff


@plugin.register_transformer
def _8(ff: ImportanceFormat) -> (pd.DataFrame):
    with ff.open() as fh:
        return _read_dataframe(fh).apply(
            lambda x: pd.to_numeric(x, errors='raise'))


@plugin.register_transformer
def _9(ff: ImportanceFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh).apply(
            lambda x: pd.to_numeric(x, errors='raise')))


@plugin.register_transformer
def _10(data: pd.DataFrame) -> (ProbabilitiesFormat):
    ff = ProbabilitiesFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep='\t', na_rep=np.nan, header=True)
    return ff


@plugin.register_transformer
def _11(ff: ProbabilitiesFormat) -> (pd.DataFrame):
    with ff.open() as fh:
        return _read_dataframe(fh).apply(
            lambda x: pd.to_numeric(x, errors='raise'))


@plugin.register_transformer
def _12(ff: ProbabilitiesFormat) -> (qiime2.Metadata):
    with ff.open() as fh:
        return qiime2.Metadata(_read_dataframe(fh).apply(
            lambda x: pd.to_numeric(x, errors='raise')))


@plugin.register_transformer
def _a(dirfmt: SampleEstimatorDirFmt) -> Pipeline:
    sklearn_version = dirfmt.version_info.view(dict)['sklearn-version']
    if sklearn_version != sklearn.__version__:
        raise ValueError('The scikit-learn version (%s) used to generate this'
                         ' artifact does not match the current version'
                         ' of scikit-learn installed (%s). Please retrain your'
                         ' classifier for your current deployment to prevent'
                         ' data-corruption errors.'
                         % (sklearn_version, sklearn.__version__))

    sklearn_pipeline = dirfmt.sklearn_pipeline.view(PickleFormat)

    with tarfile.open(str(sklearn_pipeline)) as tar:
        tmpdir = model.DirectoryFormat()
        dirname = str(tmpdir)
        tar.extractall(dirname)
        pipeline = joblib.load(os.path.join(dirname, 'sklearn_pipeline.pkl'))
        for fn in tar.getnames():
            os.unlink(os.path.join(dirname, fn))

    return pipeline


@plugin.register_transformer
def _b(data: Pipeline) -> SampleEstimatorDirFmt:
    sklearn_pipeline = PickleFormat()
    with tarfile.open(str(sklearn_pipeline), 'w') as tar:
        tmpdir = model.DirectoryFormat()
        pf = os.path.join(str(tmpdir), 'sklearn_pipeline.pkl')
        for fn in joblib.dump(data, pf):
            tar.add(fn, os.path.basename(fn))
            os.unlink(fn)

    dirfmt = SampleEstimatorDirFmt()
    dirfmt.version_info.write_data(
        {'sklearn-version': sklearn.__version__}, dict)
    dirfmt.sklearn_pipeline.write_data(sklearn_pipeline, PickleFormat)

    return dirfmt


@plugin.register_transformer
def _d(fmt: JSONFormat) -> dict:
    with fmt.open() as fh:
        return json.load(fh)


@plugin.register_transformer
def _e(data: dict) -> JSONFormat:
    result = JSONFormat()
    with result.open() as fh:
        json.dump(data, fh)
    return result
