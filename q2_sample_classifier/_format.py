# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import tarfile
import json

import qiime2.plugin.model as model
from qiime2.plugin import ValidationError


def _validate_record_len(cells, current_line_number, exp_len):
    if len(cells) != exp_len:
        raise ValidationError(
            "Expected data record to be TSV with {0} "
            "fields. Detected {1} fields at line {2}:\n\n{3!r}"
            .format(exp_len, len(cells), current_line_number, cells))


def _validate_file_not_empty(has_data):
    if not has_data:
        raise ValidationError(
            "There must be at least one data record present in the "
            "file in addition to the header line.")


class BooleanSeriesFormat(model.TextFileFormat):
    def _validate_(self, level):
        n_records = {'min': 5, 'max': None}[level]
        with self.open() as fh:
            # validate header
            # for now we will not validate any information in the header.
            line = fh.readline()

            # validate body
            has_data = False
            for line_number, line in enumerate(fh, start=2):
                cells = line.strip().split('\t')
                _validate_record_len(cells, line_number, 2)
                if str(cells[1]) not in ('True', 'False'):
                    raise ValidationError(
                        "Expected data to be comprised of values `True` and "
                        "`False`, found {0} at line {1}."
                        .format(str(cells[1]), line_number))
                has_data = True
                if n_records is not None and (line_number - 1) >= n_records:
                    break

            _validate_file_not_empty(has_data)


BooleanSeriesDirectoryFormat = model.SingleFileDirectoryFormat(
    'BooleanSeriesDirectoryFormat', 'outliers.tsv',
    BooleanSeriesFormat)


# This is effectively an internal format - it isn't registered with the
# plugin, but rather used as part of a dir fmt. This format also exists
# in q2-feature-classifier.
class PickleFormat(model.BinaryFileFormat):
    def _validate_(self, level):
        if not tarfile.is_tarfile(str(self)):
            raise ValidationError(
                "Unable to load pickled file (not a tar file).")


# https://github.com/qiime2/q2-types/issues/49
# This is effectively an internal format - it isn't registered with the
# plugin, but rather used as part of a dir fmt. This format also exists
# in q2-feature-classifier.
class JSONFormat(model.TextFileFormat):
    def _validate_(self, level):
        with self.open() as fh:
            try:
                json.load(fh)
            except json.JSONDecodeError as e:
                raise ValidationError(e)


class SampleEstimatorDirFmt(model.DirectoryFormat):
    version_info = model.File('sklearn_version.json', format=JSONFormat)
    sklearn_pipeline = model.File('sklearn_pipeline.tar', format=PickleFormat)


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
                # we want to strip each cell, not the original line
                # otherwise empty cells are dropped, causing a TypeError
                cells = [c.strip() for c in line.split('\t')]
                _validate_record_len(cells, line_number, 2)
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


class _MultiColumnNumericFormat(model.TextFileFormat):
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
                # we want to strip each cell, not the original line
                # otherwise empty cells are dropped, causing a TypeError
                cells = [c.strip() for c in line.split('\t')]
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


class ImportanceFormat(_MultiColumnNumericFormat):
    pass


ImportanceDirectoryFormat = model.SingleFileDirectoryFormat(
    'ImportanceDirectoryFormat', 'importance.tsv',
    ImportanceFormat)


class ProbabilitiesFormat(_MultiColumnNumericFormat):
    pass


ProbabilitiesDirectoryFormat = model.SingleFileDirectoryFormat(
    'ProbabilitiesDirectoryFormat', 'class_probabilities.tsv',
    ProbabilitiesFormat)


TrueTargetsDirectoryFormat = model.SingleFileDirectoryFormat(
    'TrueTargetsDirectoryFormat', 'true_targets.tsv',
    PredictionsFormat)
