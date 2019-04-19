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
import org.apache.spark.ml.attribute.{MetadataHelper, NominalAttribute}
import org.apache.spark.ml.param._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

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

  /**
   * Function to set parameters before passing into the validation step
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Parameters set in examining data
   */
  override def preValidationPrepare(data: DataFrame): PrevalidationVal = {
    val labelColName = if (isSet(labelColumnName)) {
      getLabelColumnName
    } else {
      data.columns(0)
    }

    if (!isSet(labelsToKeep)) {
      val labelCounts = data.groupBy(labelColName).count().persist()
      val res = estimate(labelCounts)
      labelCounts.unpersist()
      setLabels(res.labelsKept.distinct, res.labelsDropped.distinct, res.labelsDroppedTotal)
    }

    val labelMetaArr = getLabelsFromMetadata(data)
    val labelSet = getLabelsToKeep.toSet

    log.info(s"Dropping rows with columns not in $labelSet")

    // Update metadata that spark.ml Classifier is using tp determine the number of classes
    val na = NominalAttribute.defaultAttr.withName(labelColName)
    val metadataNA = if (labelMetaArr.isEmpty) {
      log.info("setting num vals " + labelSet.max.toInt + 1)
      na.withNumValues(labelSet.max.toInt + 1)
    } else {
      val newLabelMetaArr = labelMetaArr
        .zipWithIndex
        .collect {
          case (label: String, idx: Int) if labelSet.contains(idx.toDouble) => label
        }

      na.withValues(newLabelMetaArr)
    }

    // filter low cardinality labels out of the dataframe to reduce the volume and  to keep
    // it in sync with the new metadata.
    val labelColIdx = data.columns.indexOf(labelColName)
    val dataPrep = data
      .filter(r => labelSet.contains(r.getDouble(labelColIdx)))
      .withColumn(labelColName, data(labelColName).as(labelColName, metadataNA.toMetadata))

    summary = Option(DataCutterSummary(
      labelsKept = getLabelsToKeep,
      labelsDropped = getLabelsToDrop,
      labelsDroppedTotal = getLabelsDroppedTotal
    ))
    PrevalidationVal(summary, Option(dataPrep))
  }


  def getLabelsFromMetadata(data: DataFrame): Array[String] = {
    val labelSF = data.schema.head
    val labelColMetadata = labelSF.metadata
    log.info(s"Raw label column metadata: $labelColMetadata")

    Try {
      labelColMetadata
        .getMetadata(MetadataHelper.attributeKeys.ML_ATTR)
        .getStringArray(MetadataHelper.attributeKeys.VALUES)
    }
    .recover { case nonFatal =>
      log.warn("Cannot retrieve categories from metadata using " +
        s"${MetadataHelper.attributeKeys.ML_ATTR}.${MetadataHelper.attributeKeys.VALUES}, " +
        "retrieving number of categories using " +
        s"${MetadataHelper.attributeKeys.ML_ATTR}.${MetadataHelper.attributeKeys.NUM_VALUES}",
        nonFatal)
      val numVals = labelColMetadata
        .getMetadata(MetadataHelper.attributeKeys.ML_ATTR)
        .getLong(MetadataHelper.attributeKeys.NUM_VALUES)
      (0 until numVals.toInt).map(_.toDouble.toString).toArray
    }
    .recover {
      case nonFatal =>
        log.warn("Using an empty label array", nonFatal)
        Array.empty[String]
    }
    .getOrElse(Array.empty[String])
  }

  /**
   * Estimate the labels to keep and update metadata
   *
   * @param labelCounts
   * @return Set of labels to keep & to drop
   */
  private[op] def estimate(labelCounts: DataFrame): DataCutterSummary = {
    val numDroppedToRecord = getNumDroppedLabelsForLogging

    val minLabelFract = getMinLabelFraction
    val maxLabels = getMaxLabelCategories

    val Seq(labelColIdx, countColIdx) = Seq(0, 1)
    val Seq(labelCol, countCol) = Seq(labelColIdx, countColIdx).map(idx => labelCounts.columns(idx))

    val numLabels = labelCounts.count()
    val totalValues = labelCounts.agg(sum(countCol)).first().getLong(labelColIdx).toDouble

    val labelsKept = labelCounts
      .filter(r => r.getLong(countColIdx).toDouble / totalValues >= minLabelFract)
      .sort(col(countCol).desc, col(labelCol))
      .take(maxLabels)
      .map(_.getDouble(labelColIdx))

    val labelsKeptSet = labelsKept.toSet

    val labelsDropped = labelCounts
      .filter(r => !labelsKeptSet.contains(r.getDouble(labelColIdx)))
      .sort(col(countCol).desc, col(labelCol))
      .take(numDroppedToRecord)
      .map(_.getDouble(labelColIdx))

    val labelsDroppedTotal = numLabels - labelsKept.length

    if (labelsKept.nonEmpty) {
      log.info(s"DataCutter is keeping labels: ${labelsKept.mkString(", ")}" +
        s" and dropping labels: ${labelsDropped.mkString(", ")}")
    } else {
      throw new RuntimeException(s"DataCutter dropped all labels with param settings:" +
        s" minLabelFraction = $minLabelFract, maxLabelCategories = $maxLabels. \n" +
        s"Label counts were: ${labelCounts.collect().toSeq}")
    }
    DataCutterSummary(labelsKept.toSeq, labelsDropped.toSeq, labelsDroppedTotal.toLong)
  }

  override def copy(extra: ParamMap): DataCutter = {
    val copy = new DataCutter(uid)
    copyValues(copy, extra)
  }
}

private[impl] trait DataCutterParams extends SplitterParams {

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

  private[op] def setLabels(keep: Seq[Double], dropTopK: Seq[Double], labelsDropped: Long): this.type = {
    set(labelsToKeep, keep.toArray)
      .set(labelsToDrop, dropTopK.toArray)
      .set(labelsDroppedTotal, labelsDropped)
  }

  private[op] def getLabelsToKeep: Array[Double] = $(labelsToKeep)

  private[op] final val labelsToDrop = new DoubleArrayParam(this, "labelsDropped",
    "the top of the labels to drop when applying the data cutter")

  private[op] def getLabelsToDrop: Array[Double] = $(labelsToDrop)

  private[op] final val labelsDroppedTotal = new LongParam(this, "labelsDroppedTotal",
    "the number of labels dropped")

  private[op] def getLabelsDroppedTotal: Long = $(labelsDroppedTotal)

  final val maxNamesForDroppedLabels = new IntParam(this, "maxNamesForDroppedLabels",
    "maximum number of dropped label categories to retain for logging",
    ParamValidators.inRange(lowerBound = 0, upperBound = 1000, lowerInclusive = true, upperInclusive = true)
  )
  setDefault(maxNamesForDroppedLabels, 10)

  private[op] def getNumDroppedLabelsForLogging: Int = $(maxNamesForDroppedLabels)
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
  labelsDropped: Seq[Double],
  labelsDroppedTotal: Long
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
      .putLong(ModelSelectorNames.LabelsDroppedTotal, labelsDroppedTotal)
      .build()
  }
}
