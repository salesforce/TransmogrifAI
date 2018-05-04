/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.ml.param._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.slf4j.LoggerFactory

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
   * function to use to prepare the dataset for modeling
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Training set test set
   */
  def prepare(data: Dataset[LabelFeaturesKey]): ModelData = {
    import data.sparkSession.implicits._

    val keep =
      if (!isSet(labelsToKeep) || !isSet(labelsToDrop)) {
        val labels = data.map(r => r._1 -> 1L)
        val labelCounts = labels.groupBy(labels.columns(0)).sum(labels.columns(1)).persist()
        val (resKeep, resDrop) = estimate(labelCounts)
        labelCounts.unpersist()
        setLabels(resKeep, resDrop)
        resKeep
      } else getLabelsToKeep.toSet

    val dataUse = data.filter(r => keep.contains(r._1))

    val labelsMeta = new MetadataBuilder()
      .putDoubleArray(ModelSelectorBaseNames.LabelsKept, getLabelsToKeep)
      .putDoubleArray(ModelSelectorBaseNames.LabelsDropped, getLabelsToDrop)

    new ModelData(dataUse, labelsMeta)
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

    if (labelSet.size > 1) {
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

  def setMaxLabelCategories(value: Int): this.type = {
    set(maxLabelCategories, value)
  }

  def getMaxLabelCategories: Int = $(maxLabelCategories)

  final val minLabelFraction = new DoubleParam(this, "minLabelFraction",
    "minimum fraction of the data a label category must have", ParamValidators.inRange(
      lowerBound = 0.0, upperBound = 0.5, lowerInclusive = true, upperInclusive = false
    )
  )
  setDefault(minLabelFraction, SplitterParamsDefault.MinLabelFractionDefault)

  def setMinLabelFraction(value: Double): this.type = {
    set(minLabelFraction, value)
  }

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
