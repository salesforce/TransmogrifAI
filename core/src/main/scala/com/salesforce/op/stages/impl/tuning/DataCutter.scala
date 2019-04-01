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
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.slf4j.LoggerFactory

import scala.util.Try

case object DataCutter {

  /**
   * Creates instance that will split data into training and test set filtering out any labels that don't
   * meet the minimum fraction cutoff or fall in the top N labels specified
   *
   * @param seed                set for the random split
   * @param reserveTestFraction fraction of the data used for test
   * @param maxLabelCategories  maximum number of label categories to include
   * @param minLabelFraction    minimum fraction of total labels that a category must have to be included
   * @return data splitter
   */
  def apply(
    seed: Long = SplitterParamsDefault.seedDefault,
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault,
    maxLabelCategories: Int = SplitterParamsDefault.MaxLabelCategoriesDefault,
    minLabelFraction: Double = SplitterParamsDefault.MinLabelFractionDefault
  ): DataCutter = {
    new DataCutter()
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)
      .setMaxLabelCategories(maxLabelCategories)
      .setMinLabelFraction(minLabelFraction)
  }
}

/**
 * Instance that will make a holdout set and prepare the data for multiclass modeling
 * Creates instance that will split data into training and test set filtering out any labels that don't
 * meet the minimum fraction cutoff or fall in the top N labels specified.
 *
 * @param uid
 */
class DataCutter(uid: String = UID[DataCutter]) extends Splitter(uid = uid) with DataCutterParams {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Function to set parameters before passing into the validation step
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Parameters set in examining data
   */
  override def preValidationPrepare(data: Dataset[Row]): Option[SplitterSummary] = {

    import data.sparkSession.implicits._

    val labels = data.map(r => r.getDouble(0) -> 1L)
    val labelCounts = labels.groupBy(labels.columns(0)).sum(labels.columns(1)).persist()
    val (resKeep, resDrop) = estimate(labelCounts)
    labelCounts.unpersist()
    setLabels(resKeep, resDrop)

    summary = Option(DataCutterSummary(labelsKept = getLabelsToKeep, labelsDropped = getLabelsToDrop))
    summary
  }

  /**
   * Removes labels that should not be used in modeling
   *
   * @param data first column must be the label as a double
   * @return Training set test set
   */
  override def validationPrepare(data: Dataset[Row]): Dataset[Row] = {
    val dataPrep = super.validationPrepare(data)
    Try {
      val labelSF: StructField = dataPrep.schema.head
      val labelColMeta = labelSF.metadata
      log.info(s"Label column metadata ${labelSF.name} $labelColMeta")
      labelColMeta
        .getMetadata("ml_attr")
        .getStringArray("vals")
        .map(_.toDouble)
    }
      .map(_.toSet)
      .recover { case nonFatal =>
        log.warn(s"Recovering non-fatal exception: $nonFatal", nonFatal)
        Set.empty
      }
      .map { valSet =>

        val labelSet = getLabelsToKeep.toSet
        if (valSet == labelSet) {
          log.info("Label sets identical in dataset metadata and datacutter, skipping label trim")
          dataPrep
        } else {
          log.info(s"Dropping rows with columns not in $labelSet")

          val labelColName = dataPrep.columns(0)
          val labelCol = dataPrep.col(labelColName)
          val metadata = NominalAttribute.defaultAttr
            .withName(labelColName)
            .withValues(getLabelsToKeep.map(_.toString))
            .toMetadata()
          dataPrep
            .filter(r => labelSet.contains(r.getDouble(0)))
            .withColumn(labelColName, labelCol.as("_", metadata))
        }
    }
      .get
  }

  /**
   * Estimate the labels to keep and update metadata
   *
   * @param labelCounts
   * @return Set of labels to keep & to drop
   */
  private[op] def estimate(labelCounts: DataFrame): (Set[Double], Set[Double]) = {
    val minLabelFract = getMinLabelFraction
    val maxLabels = getMaxLabelCategories

    val colCount = labelCounts.columns(1)
    val totalValues = labelCounts.agg(sum(colCount)).first().getLong(0).toDouble
    val labelsKeep = labelCounts
      .filter(r => (r.getLong(1) / totalValues) >= minLabelFract)
      .sort(col(colCount).desc)
      .take(maxLabels)
      .map(_.getDouble(0))

    val labelSet = labelsKeep.toSet
    val labelsDropped = labelCounts.filter(r => !labelSet.contains(r.getDouble(0))).collect().map(_.getDouble(0)).toSet

    if (labelSet.nonEmpty) {
      log.info(s"DataCutter is keeping labels: $labelSet and dropping labels: $labelsDropped")
    } else {
      throw new RuntimeException(s"DataCutter dropped all labels with param settings:" +
        s" minLabelFraction = $minLabelFract, maxLabelCategories = $maxLabels. \n" +
        s"Label counts were: ${labelCounts.collect().toSeq}")
    }
    labelSet -> labelsDropped
  }

  override def copy(extra: ParamMap): DataCutter = {
    val copy = new DataCutter(uid)
    copyValues(copy, extra)
  }
}

private[impl] trait DataCutterParams extends Params {

  final val maxLabelCategories = new IntParam(this, "maxLabelCategories",
    "maximum number of label categories for multiclass classification",
    ParamValidators.inRange(lowerBound = 1, upperBound = 1 << 30, lowerInclusive = false, upperInclusive = true)
  )
  setDefault(maxLabelCategories, SplitterParamsDefault.MaxLabelCategoriesDefault)

  def setMaxLabelCategories(value: Int): this.type = set(maxLabelCategories, value)

  def getMaxLabelCategories: Int = $(maxLabelCategories)

  final val minLabelFraction = new DoubleParam(this, "minLabelFraction",
    "minimum fraction of the data a label category must have", ParamValidators.inRange(
      lowerBound = 0.0, upperBound = 0.5, lowerInclusive = true, upperInclusive = false
    )
  )
  setDefault(minLabelFraction, SplitterParamsDefault.MinLabelFractionDefault)

  def setMinLabelFraction(value: Double): this.type = set(minLabelFraction, value)

  def getMinLabelFraction: Double = $(minLabelFraction)

  private[op] final val labelsToKeep = new DoubleArrayParam(this, "labelsToKeep",
    "labels to keep when applying the data cutter")

  private[op] def setLabels(keep: Set[Double], drop: Set[Double]): this.type = {
    set(labelsToKeep, keep.toArray.sorted)
    set(labelsToDrop, drop.toArray.sorted)
  }

  private[op] def getLabelsToKeep: Array[Double] = $(labelsToKeep)

  private[op] final val labelsToDrop = new DoubleArrayParam(this, "labelsDropped",
    "labels to drop when applying the data cutter")

  private[op] def getLabelsToDrop: Array[Double] = $(labelsToDrop)

}

/**
 * Summary of results for data cutter
 *
 * @param labelsKept    labels retained
 * @param labelsDropped labels dropped by data cutter
 */
case class DataCutterSummary
(
  labelsKept: Seq[Double],
  labelsDropped: Seq[Double]
) extends SplitterSummary {

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
      .putDoubleArray(ModelSelectorNames.LabelsKept, labelsKept.toArray)
      .putDoubleArray(ModelSelectorNames.LabelsDropped, labelsDropped.toArray)
      .build()
  }

}
