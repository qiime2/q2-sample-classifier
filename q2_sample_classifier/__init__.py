# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import importlib

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

_format = importlib.import_module('q2_sample_classifier._format')
BooleanSeriesFormat = _format.BooleanSeriesFormat
BooleanSeriesDirectoryFormat = _format.BooleanSeriesDirectoryFormat
PredictionsFormat = _format.PredictionsFormat
PredictionsDirectoryFormat = _format.PredictionsDirectoryFormat
ImportanceFormat = _format.ImportanceFormat
ImportanceDirectoryFormat = _format.ImportanceDirectoryFormat
SampleEstimatorDirFmt = _format.SampleEstimatorDirFmt
PickleFormat = _format.PickleFormat

_type = importlib.import_module('q2_sample_classifier._type')
BooleanSeries = _type.BooleanSeries
Predictions = _type.Predictions
Importance = _type.Importance
SampleEstimator = _type.SampleEstimator

importlib.import_module('q2_sample_classifier._transformer')
