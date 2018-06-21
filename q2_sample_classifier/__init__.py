# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from ._format import (
    BooleanSeriesFormat, BooleanSeriesDirectoryFormat,
    PredictionsFormat, PredictionsDirectoryFormat, ImportanceFormat,
    ImportanceDirectoryFormat, SampleEstimatorDirFmt, PickleFormat)
from ._type import BooleanSeries, Predictions, Importance, SampleEstimator
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions

__all__ = ['BooleanSeriesFormat', 'BooleanSeriesDirectoryFormat',
           'PredictionsFormat', 'PredictionsDirectoryFormat',
           'ImportanceFormat', 'ImportanceDirectoryFormat',
           'SampleEstimatorDirFmt', 'PickleFormat', 'BooleanSeries',
           'Predictions', 'Importance', 'SampleEstimator']
