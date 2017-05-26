# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

# Much of this type format is adapted from q2_types.sample_data, but to support
# a pandas.DataFrame instead of pandas.Series
import qiime2.plugin.model as model
from .plugin_setup import plugin


class CoordinatesFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            for line, _ in zip(fh, range(10)):
                cells = line.split('\t')
                if len(cells) < 2:
                    return False
            return True


CoordinatesDirectoryFormat = model.SingleFileDirectoryFormat(
    'CoordinatesDirectoryFormat', 'coordinates.tsv',
    CoordinatesFormat)


plugin.register_formats(CoordinatesFormat, CoordinatesDirectoryFormat)
