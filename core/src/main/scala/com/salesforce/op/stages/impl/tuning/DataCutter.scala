/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import com.salesforce.op.stages.impl.tuning.SelectorData.LabelFeaturesKey
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.MetadataBuilder
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
      .setMaxLabelCategores(maxLabelCategories)
      .setMinLabelFraction(minLabelFraction)
  }
}

/**
 * Instance that will make a holdout set and prepare the data for multicalss modeling
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

    val minLabelFract = getMinLabelFraction
    val maxLabels = getMaxLabelCategories

    val labels = data.map(r => r._1 -> 1L)
    val labelCounts = labels.groupBy(labels.columns(0)).sum(labels.columns(1)).persist()
    val totalValues = labelCounts.groupBy().sum(labelCounts.columns(1)).first().getLong(0).toDouble
    val labelsKeep = labelCounts
      .filter(r => (r.getLong(1) / totalValues) >= minLabelFract)
      .sort($"sum(_2)".desc)
      .take(maxLabels)
      .map(_.getDouble(0))

    val labelSet = labelsKeep.toSet
    val labelsDropped = labelCounts.filter(r => !labelSet.contains(r.getDouble(0))).collect().map(_.getDouble(0))

    if (labelSet.size > 1) {
      log.info(s"DataCutter is keeping labels: $labelSet and dropping labels: ${labelsDropped.toSet}")
    } else {
      throw new RuntimeException(s"DataCutter dropped all labels with param settings:" +
        s" minLabelFraction = $minLabelFract, maxLabelCategories = $maxLabels. \n" +
        s"Label counts were: ${labelCounts.collect().toSeq}")
    }

    val dataUse = data.filter(r => labelSet.contains(r._1))

    val metadata = new MetadataBuilder()
    metadata.putDoubleArray(ModelSelectorBaseNames.LabelsKept, labelsKeep)
    metadata.putDoubleArray(ModelSelectorBaseNames.LabelsDropped, labelsDropped)

    new ModelData(dataUse, metadata)
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

  def setMaxLabelCategores(value: Int): this.type = {
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

}
