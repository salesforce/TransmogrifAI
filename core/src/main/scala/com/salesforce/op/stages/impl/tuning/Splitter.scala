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

import com.salesforce.op.stages.impl.MetadataLike
import com.salesforce.op.stages.impl.selector.ModelSelectorNames
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.Metadata
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.slf4j.LoggerFactory

import scala.util.Try

case class PrevalidationVal(summaryOpt: Option[SplitterSummary], dataFrame: Option[DataFrame])

/**
 * Abstract class that will carry on the creation of training set + test set
 */
abstract class Splitter(val uid: String) extends SplitterParams {
  @transient private[tuning] lazy val log = LoggerFactory.getLogger(this.getClass)

  @transient private[op] var summary: Option[SplitterSummary] = None

  /**
   * Function to use to create the training set and test set.
   *
   * @param data
   * @return (dataTrain, dataTest)
   */
  def split[T](data: Dataset[T]): (Dataset[T], Dataset[T]) = {
    val fraction = 1.0 - getReserveTestFraction
    val Array(dataTrain, dataTest) = data.randomSplit(Array(fraction, 1.0 - fraction), seed = $(seed))
    dataTrain -> dataTest
  }

  /**
   * Function to use to prepare the dataset for modeling within the validation step
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Training set test set
   */
  def validationPrepare(data: Dataset[Row]): Dataset[Row] = {
    checkPreconditions()
    data
  }


  /**
   * Function to set parameters before passing into the validation step
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Parameters set in examining data
   */
  def preValidationPrepare(data: Dataset[Row]): PrevalidationVal

  protected def checkPreconditions(): Unit =
    require(summary.nonEmpty, "Cannot call validationPrepare until preValidationPrepare has been called")

  /**
   * Add a splitter parameter to name the label column
   *
   * @param label
   * @return
   */
  def withLabelColumnName(label: String): Splitter = {
    if (!isSet(labelColumnName)) {
      set(labelColumnName, label)
    } else {
      log.warn(s"$labelColumnName on an existing Splitter instance can be set only once")
      this
    }
  }
}

trait SplitterParams extends Params {

  /**
   * Seed for data splitting
   *
   * @group param
   */
  final val seed = new LongParam(this, "seed", "seed for the splitting/balancing")
  setDefault(seed, SplitterParamsDefault.seedDefault)

  def setSeed(value: Long): this.type = set(seed, value)
  def getSeed: Long = $(seed)

  /**
   * Fraction of data to reserve for test
   * Default is 0.1
   *
   * @group param
   */
  final val reserveTestFraction = new DoubleParam(this, "reserveTestFraction", "fraction of data to reserve for test",
    ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = true, upperInclusive = false)
  )
  setDefault(reserveTestFraction, SplitterParamsDefault.ReserveTestFractionDefault)

  def setReserveTestFraction(value: Double): this.type = set(reserveTestFraction, value)
  def getReserveTestFraction: Double = $(reserveTestFraction)

  /**
   * Maximum size of dataset want to train on.
   * Value should be > 0.
   * Default is 1000000.
   *
   * @group param
   */
  final val maxTrainingSample = new IntParam(this, "maxTrainingSample",
    "maximum size of dataset want to train on", ParamValidators.inRange(
      lowerBound = 0, upperBound = 1 << 30, lowerInclusive = false, upperInclusive = true
    )
  )
  setDefault(maxTrainingSample, SplitterParamsDefault.MaxTrainingSampleDefault)

  def setMaxTrainingSample(value: Int): this.type = set(maxTrainingSample, value)

  def getMaxTrainingSample: Int = $(maxTrainingSample)

  final val labelColumnName = new Param[String](this, "labelColumnName",
    "label column name, column 0 if not specified")
  private[op] def getLabelColumnName = $(labelColumnName)
}

object SplitterParamsDefault {

  def seedDefault: Long = util.Random.nextLong

  val ReserveTestFractionDefault = 0.1
  val SampleFractionDefault = 0.1
  val MaxTrainingSampleDefault = 1E6.toInt
  val MaxLabelCategoriesDefault = 100
  val MinLabelFractionDefault = 0.0
  val DownSampleFractionDefault = 1.0
}

trait SplitterSummary extends MetadataLike

private[op] object SplitterSummary {
  val ClassName: String = "className"

  def fromMetadata(metadata: Metadata): Try[SplitterSummary] = Try {
    metadata.getString(ClassName) match {
      case s if s == classOf[DataSplitterSummary].getName => DataSplitterSummary(
        preSplitterDataCount = metadata.getLong(ModelSelectorNames.PreSplitterDataCount),
        downSamplingFraction = metadata.getDouble(ModelSelectorNames.DownSample)
      )
      case s if s == classOf[DataBalancerSummary].getName => DataBalancerSummary(
        positiveLabels = metadata.getLong(ModelSelectorNames.Positive),
        negativeLabels = metadata.getLong(ModelSelectorNames.Negative),
        desiredFraction = metadata.getDouble(ModelSelectorNames.Desired),
        upSamplingFraction = metadata.getDouble(ModelSelectorNames.UpSample),
        downSamplingFraction = metadata.getDouble(ModelSelectorNames.DownSample)
      )
      case s if s == classOf[DataCutterSummary].getName => DataCutterSummary(
        labelsKept = metadata.getDoubleArray(ModelSelectorNames.LabelsKept),
        labelsDropped = metadata.getDoubleArray(ModelSelectorNames.LabelsDropped),
        labelsDroppedTotal = metadata.getLong(ModelSelectorNames.LabelsDroppedTotal)
      )
      case s =>
        throw new RuntimeException(s"Unknown splitter summary class '$s'")
    }
  }
}

