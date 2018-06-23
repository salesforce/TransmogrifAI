/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.tuning

import org.apache.spark.ml.param._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}


/**
 * Case class for Training & test sets
 *
 * @param train      training set is persisted at construction
 * @param metadata   metadata built at construction
 */
case class ModelData private(train: Dataset[Row], metadata: Metadata) {
  def this(train: Dataset[Row], metadata: MetadataBuilder) =
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
  def split[T](data: Dataset[T]): (Dataset[T], Dataset[T]) = {
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
  def prepare(data: Dataset[Row]): ModelData

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
}

object SplitterParamsDefault {

  def seedDefault: Long = util.Random.nextLong

  val ReserveTestFractionDefault = 0.1
  val SampleFractionDefault = 0.1
  val MaxTrainingSampleDefault = 1E6.toInt
  val MaxLabelCategoriesDefault = 100
  val MinLabelFractionDefault = 0.0
}
