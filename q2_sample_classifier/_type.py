# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData
import qiime2.plugin.model as model

from q2_sample_classifier import plugin_setup


Coordinates = SemanticType('Coordinates', variant_of=SampleData.field['type'])

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


plugin_setup.plugin.register_formats(CoordinatesFormat, CoordinatesDirectoryFormat)

plugin_setup.plugin.register_semantic_types(Coordinates)

plugin_setup.plugin.register_semantic_type_to_format(
    SampleData[Coordinates],
    artifact_format=CoordinatesDirectoryFormat)
