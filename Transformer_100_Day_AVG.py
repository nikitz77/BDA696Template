# HW3 - Spark Assignment
# BDA 696 -  Karenina Zaballa
# BOILERPLATE TRANSFORMER IN CLASS

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable


class R_AVG_100(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(R_AVG_100, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()
        # This is where the rolling average calculation actually occurs
        dataset = dataset.withColumn(
            output_col, dataset[input_cols[1]] / dataset[input_cols[2]]
        )
        return dataset
