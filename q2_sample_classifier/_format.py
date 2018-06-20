# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import importlib
import tarfile
import json

import qiime2.plugin.model as model
from qiime2.plugin import ValidationError


class BooleanSeriesFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            line = fh.readline()
            for line, _ in zip(fh, range(5)):
                cells = line.strip().split('\t')
                if len(cells) != 2 or str(cells[1]) not in ('True', 'False'):
                    return False
            return True


BooleanSeriesDirectoryFormat = model.SingleFileDirectoryFormat(
    'BooleanSeriesDirectoryFormat', 'outliers.tsv',
    BooleanSeriesFormat)


class PickleFormat(model.BinaryFileFormat):
    def sniff(self):
        return tarfile.is_tarfile(str(self))


# https://github.com/qiime2/q2-types/issues/49
class JSONFormat(model.TextFileFormat):
    def sniff(self):
        with self.open() as fh:
            try:
                json.load(fh)
                return True
            except json.JSONDecodeError:
                pass
        return False


class SampleEstimatorDirFmt(model.DirectoryFormat):
    version_info = model.File('sklearn_version.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


def _validate_file_not_empty(has_data):
    if not has_data:
        raise ValidationError(
            "There must be at least one data record present in the "
            "file in addition to the header line.")


class PredictionsFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # validate header
            # for now we will not validate any information in the header,
            # since the name of the predicted column should be flexible. The
            # header name written by methods in q2-sample-classifier will be
            # "predicted-*", but this should also accommodate user-defined
            # column names.
            line = fh.readline()

            # validate body
            has_data = False
            for line_number, line in enumerate(fh, start=2):
                cells = line.strip().split('\t')
                if len(cells) != 2:
                    raise ValidationError(
                        "Expected data record to be TSV with two "
                        "fields. Detected {0} fields at line {1}:\n\n{2!r}"
                        .format(len(cells), line_number, cells))
                has_data = True
                if n_records is not None and (line_number - 1) >= n_records:
                    break

            _validate_file_not_empty(has_data)

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


PredictionsDirectoryFormat = model.SingleFileDirectoryFormat(
    'PredictionsDirectoryFormat', 'predictions.tsv',
    PredictionsFormat)


class ImportanceFormat(model.TextFileFormat):
    def _validate(self, n_records=None):
        with self.open() as fh:
            # validate header
            # for now we will not validate any information in the header,
            # since column names, count etc are frequently unique to individual
            # estimators. Let's keep this flexible.
            line = fh.readline()

            # validate body
            has_data = False
            for line_number, line in enumerate(fh, start=2):
                cells = line.strip().split('\t')
                if len(cells) < 2:
                    raise ValidationError(
                        "Expected data record to be TSV with two or more "
                        "fields. Detected {0} fields at line {1}:\n\n{2!r}"
                        .format(len(cells), line_number, cells))
                # all values (except row name) should be numbers
                try:
                    [float(c) for c in cells[1:]]
                except ValueError:
                    raise ValidationError(
                        "Columns must contain only numeric values. "
                        "A non-numeric value ({0!r}) was detected at line "
                        "{1}.".format(cells[1], line_number))

                has_data = True
                if n_records is not None and (line_number - 1) >= n_records:
                    break

            _validate_file_not_empty(has_data)

    def _validate_(self, level):
        record_count_map = {'min': 5, 'max': None}
        self._validate(record_count_map[level])


ImportanceDirectoryFormat = model.SingleFileDirectoryFormat(
    'ImportanceDirectoryFormat', 'importance.tsv',
    ImportanceFormat)

plugin_setup = importlib.import_module('.plugin_setup', 'q2_sample_classifier')

plugin_setup.plugin.register_formats(
    SampleEstimatorDirFmt, BooleanSeriesFormat, BooleanSeriesDirectoryFormat,
    ImportanceFormat, ImportanceDirectoryFormat, PredictionsFormat,
    PredictionsDirectoryFormat)
