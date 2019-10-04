/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.selector.ModelSelectorNames
import org.apache.spark.ml.param._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

case object DataSplitter {

  /**
   * Creates instance that will split data into training and test set
   *
   * @param seed                set for the random split
   * @param reserveTestFraction fraction of the data used for test
   * @return data splitter
   */
  def apply(
    seed: Long = SplitterParamsDefault.seedDefault,
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault,
    maxTrainingSample: Int = SplitterParamsDefault.MaxTrainingSampleDefault
  ): DataSplitter = {
    new DataSplitter()
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)
      .setMaxTrainingSample(maxTrainingSample)
  }
}

/**
 * Instance that will split the data into training and holdout for regressions
 *
 * @param uid
 */
class DataSplitter(uid: String = UID[DataSplitter]) extends Splitter(uid = uid) with DataSplitterParams {

  /**
   * Function to set the down sampling fraction and parameters before passing into the validation step
   *
   * @param data
   * @return Parameters set in examining data
   */
  override def preValidationPrepare(data: Dataset[Row]): PrevalidationVal = {
    val dataSetSize = data.count().toDouble
    val sampleF = getMaxTrainingSample / dataSetSize
    val downSampleFraction = math.min(sampleF, SplitterParamsDefault.DownSampleFractionDefault)
    summary = Option(DataSplitterSummary(downSampleFraction))
    setDownSampleFraction(downSampleFraction)
    PrevalidationVal(summary, None)
  }

  /**
   * Rebalance the training data within the validation step
   *
   * @param data to prepare for model training. first column must be the label as a double
   * @return balanced training set and a test set
   */
  override def validationPrepare(data: Dataset[Row]): Dataset[Row] = {

    val dataPrep = super.validationPrepare(data)

    // check if down sampling is needed
    val balanced: DataFrame = if (getDownSampleFraction < 1) {
      dataPrep.sample( false, getDownSampleFraction, getSeed)
    } else {
      dataPrep
    }
    balanced.persist()
  }
  override def copy(extra: ParamMap): DataSplitter = {
    val copy = new DataSplitter(uid)
    copyValues(copy, extra)
  }
}
trait DataSplitterParams extends Params {
  /**
   * Fraction to down sample data
   * Value should be in [0.0, 1.0]
   *
   * @group param
   */
  private[op] final val downSampleFraction = new DoubleParam(this, "downSampleFraction",
    "fraction to down sample data", ParamValidators.inRange(
      lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = true
    )
  )
  setDefault(downSampleFraction, SplitterParamsDefault.DownSampleFractionDefault)

  private[op] def setDownSampleFraction(value: Double): this.type = set(downSampleFraction, value)

  private[op] def getDownSampleFraction: Double = $(downSampleFraction)
}

/**
 * Summary for data splitter run for storage in metadata
 * @param downSamplingFraction down sampling fraction for training set
 */
case class DataSplitterSummary(downSamplingFraction: Double) extends SplitterSummary {

  /**
   * Converts to [[Metadata]]
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean): Metadata = {
    new MetadataBuilder()
      .putString(SplitterSummary.ClassName, this.getClass.getName)
      .putDouble(ModelSelectorNames.DownSample, downSamplingFraction)
      .build()
  }

}
