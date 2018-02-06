/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

/**
 * Case class of data used in model selectors for data prep and cross validation
 * @param label label trying to predict
 * @param features features used to predict
 * @param key unique key for entity trying to score
 */
case object SelectorData {
  type LabelFeaturesKey = (Double, Vector, String)
}

/**
 * Case class for Training & test sets
 *
 * @param train      training set is persisted at construction
 * @param metadata   metadata built at construction
 */
case class ModelData private(train: Dataset[_], metadata: Metadata) {
  def this(train: Dataset[_], metadata: MetadataBuilder) =
    this(train.persist(), metadata.build())
}

/**
 * Abstract class that will carry on the creation of training set + test set
 */
abstract class Splitter(val uid: String) extends SplitterParams {

  /**
   * Function to use to create the training set and test set.
   *
   * @param data
   * @return (dataTrain, dataTest)
   */
  def split(data: Dataset[_]): (Dataset[_], Dataset[_]) = {
    val fraction = 1.0 - getReserveTestFraction
    val Array(dataTrain, dataTest) = data.randomSplit(Array(fraction, 1.0 - fraction), seed = $(seed))
    dataTrain -> dataTest
  }

  /**
   * Function to use to prepare the dataset for modeling
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Training set test set
   */
  def prepare(data: Dataset[SelectorData.LabelFeaturesKey]): ModelData

}

private[impl] trait SplitterParams extends Params {

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
}

object SplitterParamsDefault {

  def seedDefault: Long = util.Random.nextLong

  val ReserveTestFractionDefault = 0.1
  val SampleFractionDefault = 0.1
  val MaxTrainingSampleDefault = 100000
  val MaxLabelCategoriesDefault = 100
  val MinLabelFractionDefault = 0.0
}
