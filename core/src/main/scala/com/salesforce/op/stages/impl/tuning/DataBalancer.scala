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

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import org.apache.spark.ml.param._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.MetadataBuilder
import org.slf4j.LoggerFactory

case object DataBalancer {

  /**
   * Creates an instance that will balance the dataset by doing upsample and downsample. Splitting is also possible.
   *
   * @param sampleFraction      desired minimum fraction of the minority class after balancing
   * @param maxTrainingSample   maximum size of training set
   * @param seed                seed for spliting and balancing
   * @param reserveTestFraction fraction of the data used for test
   * @return data balancer
   */
  def apply(
    sampleFraction: Double = SplitterParamsDefault.SampleFractionDefault,
    maxTrainingSample: Int = SplitterParamsDefault.MaxTrainingSampleDefault,
    seed: Long = SplitterParamsDefault.seedDefault,
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault
  ): DataBalancer = {
    new DataBalancer()
      .setSampleFraction(sampleFraction)
      .setMaxTrainingSample(maxTrainingSample)
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)
  }

}

/**
 * Instance that will split the data into train and holdout and then balance the dataset
 * before modeling binary classifications
 *
 * @param uid
 */
class DataBalancer(uid: String = UID[DataBalancer]) extends Splitter(uid = uid) with DataBalancerParams {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)
  @transient private[op] val metadataBuilder = new MetadataBuilder()

  /**
   * Computes the upSample and downSample proportions.
   *
   * @param smallCount        size of minority class data
   * @param bigCount          size of majority class data
   * @param sampleF           targeted fraction of small data
   * @param maxTrainingSample maximum training size
   * @return downSample & upSample proportions
   */
  def getProportions(
    smallCount: Double,
    bigCount: Double,
    sampleF: Double,
    maxTrainingSample: Int
  ): (Double, Double) = {

    def checkUpSampleSize(multiplier: Int): Boolean = {
      ((multiplier * smallCount * (1 - sampleF) < sampleF * bigCount)
        && (maxTrainingSample * sampleF) > (smallCount * multiplier))
    }

    if (smallCount < (maxTrainingSample * sampleF)) {
      // check to make sure that upsampling will not make data too big
      val upSample =
        if (checkUpSampleSize(100)) 100.0
        else if (checkUpSampleSize(50)) 50.0
        else if (checkUpSampleSize(10)) 10.0
        else if (checkUpSampleSize(5)) 5.0
        else if (checkUpSampleSize(4)) 4.0
        else if (checkUpSampleSize(3)) 3.0
        else if (checkUpSampleSize(2)) 2.0
        else 1.0

      // then calculate appropriate subsample for big
      ((smallCount * upSample / sampleF - smallCount * upSample) / bigCount, upSample)
    } else {
      // downsample both big and small
      val upSample = (maxTrainingSample * sampleF) / smallCount
      ((1 - sampleF) * maxTrainingSample / bigCount, upSample)
    }
  }

  /**
   * Split into a training set and a test set and balance the training set
   *
   * @param data to prepare for model training. first column must be the label as a double
   * @return balanced training set and a test set
   */
  def prepare(data: Dataset[Row]): ModelData = {

    val ds = data.persist()

    val Array(negativeData, positiveData) = Array(0.0, 1.0).map(label => ds.filter(_.getDouble(0) == label).persist())
    val balancerSeed = getSeed

    prepareData(
      data = ds,
      positiveData = positiveData,
      negativeData = negativeData,
      seed = balancerSeed
    )

  }

  override def copy(extra: ParamMap): DataBalancer = {
    val copy = new DataBalancer(uid)
    copyValues(copy, extra)
  }



  /**
   * Estimate if data needs to be balanced or not. If so, computes sample fractions and sets the appropriate params
   *
   * @param data            input data
   * @param positiveData    data with positives only
   * @param negativeData    data with negatives only
   * @param seed            seed
   * @return balanced data
   */
  private[op] def estimate[T](
    data: Dataset[T],
    positiveData: Dataset[T],
    negativeData: Dataset[T],
    seed: Long
  ): Unit = {
    val positiveCount = positiveData.count()
    val negativeCount = negativeData.count()
    val totalCount = positiveCount + negativeCount
    val sampleF = getSampleFraction

    // feed metadata with counts and sample fraction
    metadataBuilder.putLong(ModelSelectorBaseNames.Positive, positiveCount)
    metadataBuilder.putLong(ModelSelectorBaseNames.Negative, negativeCount)
    metadataBuilder.putDouble(ModelSelectorBaseNames.Desired, sampleF)
    log.info(s"Data has $positiveCount positive and $negativeCount negative.")

    val (smallCount, bigCount) = {
      val isPosSmall = positiveCount < negativeCount
      setIsPositiveSmall(isPosSmall)
      if (isPosSmall) (positiveCount, negativeCount)
      else (negativeCount, positiveCount)
    }
    val maxTrainSample = getMaxTrainingSample

    if (smallCount < 100 || (smallCount + bigCount) < 500) {
      log.warn("!!!Attention!!! - there is not enough data to build a good model!")
    }

    if (smallCount.toDouble / totalCount.toDouble >= sampleF) {
      log.info(
        s"Not resampling data: $smallCount small count and $bigCount big count is greater than" +
          s" requested ${sampleF}"
      )
      // if data is too big downsample
      val fraction = if (maxTrainSample < totalCount) maxTrainSample / totalCount.toDouble else 1.0
      setAlreadyBalancedFraction(fraction)

    } else {
      log.info(s"Sampling data to get $sampleF split versus $smallCount small and $bigCount big")
      val (downSample, upSample) = getProportions(smallCount, bigCount, sampleF, maxTrainSample)

      setDownSampleFraction(downSample)
      setUpSampleFraction(upSample)

      // feed metadata with upsample and downsample
      metadataBuilder.putDouble(ModelSelectorBaseNames.UpSample, upSample)
      metadataBuilder.putDouble(ModelSelectorBaseNames.DownSample, downSample)

      val (posFraction, negFraction) =
        if (positiveCount < negativeCount) (upSample, downSample)
        else (downSample, upSample)

      val newPositiveCount = math.rint(positiveCount * posFraction)
      val newNegativeCount = math.rint(negativeCount * negFraction)
      log.info(s"After sampling see " +
        s"$newPositiveCount positives and $newNegativeCount negatives, " +
        s"sample fraction is ${
          math.min(newPositiveCount, newNegativeCount) / (newPositiveCount + newNegativeCount)
        }."
      )

      if (upSample >= 1.0) {
        log.info(s"Upsampling by a factor $upSample. Downsampling by a factor $downSample.")
      } else {
        log.info(s"Both downsampling by a factor $upSample for small and by a factor $downSample for big.\n" +
          s"To make upsampling happen, please increase the max training sample size '${maxTrainingSample.name}'")
      }

    }
  }
  /**
   * Preparing data
   *
   * @param data            input data
   * @param positiveData    data with positives only
   * @param negativeData    data with negatives only
   * @param seed            seed
   * @return balanced data
   */
  private[op] def prepareData[T](
    data: Dataset[T],
    positiveData: Dataset[T],
    negativeData: Dataset[T],
    seed: Long
  ): ModelData = {

    if (!(isSet(isPositiveSmall) || isSet(downSampleFraction) ||
      isSet(upSampleFraction) || isSet(alreadyBalancedFraction))) {
      estimate(data = data, positiveData = positiveData, negativeData = negativeData, seed = seed)
    }

    // If these conditions are met, that means that we have enough information to balance the data : upSample,
    // downSample and which class is in minority
    if (isSet(isPositiveSmall) && isSet(downSampleFraction) && isSet(upSampleFraction)) {
      val (down, up) = ($(downSampleFraction), $(upSampleFraction))
      log.info(s"Sample fractions: downSample of ${down}, upSample of ${up}")
      val (smallData, bigData) = if ($(isPositiveSmall)) (positiveData, negativeData) else (negativeData, positiveData)
      new ModelData(rebalance(smallData, up, bigData, down, seed).toDF(), metadataBuilder)
    } else { // Data is already balanced, but need to be sampled
      val fraction = $(alreadyBalancedFraction)
      log.info(s"Data is already balanced, yet it will be sampled by a fraction of $fraction")
      val balanced = sampleBalancedData(fraction = fraction, seed = seed,
        data = data, positiveData = positiveData, negativeData = negativeData).toDF()
      new ModelData(balanced, metadataBuilder)
    }
  }


  /**
   *
   * @param smallData          data with the minority class
   * @param upSampleFraction   fraction to sample minority data
   * @param bigData            data with the majority class
   * @param downSampleFraction fraction to sample minority data
   *
   * @return balanced small and big data split into training and test sets
   *         with downSample and upSample proportions
   */
  private[op] def rebalance[T](
    smallData: Dataset[T],
    upSampleFraction: Double,
    bigData: Dataset[T],
    downSampleFraction: Double,
    seed: Long
  ): Dataset[T] = {

    import smallData.sparkSession.implicits._
    val bigDataTrain = bigData.sample(withReplacement = false, downSampleFraction, seed = seed)
    val smallDataTrain = upSampleFraction match {
      case u if u > 1.0 => smallData.sample(withReplacement = true, u, seed = seed)
      case 1.0 => smallData // if upSample == 1.0, no need to upSample
      case u => smallData.sample(withReplacement = false, u, seed = seed) // downsample instead
    }

    smallDataTrain.union(bigDataTrain)
  }

  /**
   * Sample already balanced data
   *
   * @param fraction subsample to take
   * @param seed seed to use in sampling
   * @param data full dataset in case no sampling is needed
   * @param positiveData positive data for stratified sampling
   * @param negativeData negative data for stratified sampling
   * @return
   */
  private[op] def sampleBalancedData[T](
    fraction: Double,
    seed: Long,
    data: Dataset[T],
    positiveData: Dataset[T],
    negativeData: Dataset[T]
  ): Dataset[T] = {
    fraction match {
      case 1.0 => data // we don't sample
      // stratified sampling
      case r => negativeData.sample(withReplacement = false, fraction = r, seed = seed)
        .union(positiveData.sample(withReplacement = false, fraction = r, seed = seed))
    }
  }
}

trait DataBalancerParams extends Params {

  /**
   * Targeted sample fraction for the class in minority.
   * Value should be in ]0.0, 1.0[
   * Default is 0.1.
   *
   * @group param
   */
  final val sampleFraction = new DoubleParam(this, "sampleFraction",
    "proportion of the data used for training", ParamValidators.inRange(
      lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = false
    )
  )
  setDefault(sampleFraction, SplitterParamsDefault.SampleFractionDefault)

  def setSampleFraction(value: Double): this.type = {
    set(sampleFraction, value)
  }

  def getSampleFraction: Double = $(sampleFraction)

  /**
   * Maximum size of dataset want to train on.
   * Value should be > 0.
   * Default is 5000.
   *
   * @group param
   */
  final val maxTrainingSample = new IntParam(this, "maxTrainingSample",
    "maximum size of dataset want to train on", ParamValidators.inRange(
      lowerBound = 0, upperBound = 1 << 30, lowerInclusive = false, upperInclusive = true
    )
  )
  setDefault(maxTrainingSample, SplitterParamsDefault.MaxTrainingSampleDefault)

  def setMaxTrainingSample(value: Int): this.type = {
    set(maxTrainingSample, value)
  }

  def getMaxTrainingSample: Int = $(maxTrainingSample)

  /**
   * Fraction to sample minority data
   * Value should be > 0.0
   *
   * @group param
   */
  private[op] final val upSampleFraction = new DoubleParam(this, "upSampleFraction",
    "fraction to sample minority data", ParamValidators.gt(0.0) // it can be a downSample fraction
  )

  private[op] def setUpSampleFraction(value: Double): this.type = {
    set(upSampleFraction, value)
  }

  private[op] def getUpSampleFraction: Double = $(upSampleFraction)


  /**
   * Fraction to sample majority data
   * Value should be in ]0.0, 1.0]
   *
   * @group param
   */
  private[op] final val downSampleFraction = new DoubleParam(this, "downSampleFraction",
    "fraction to sample majority data", ParamValidators.inRange(
      lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = true
    )
  )

  private[op] def setDownSampleFraction(value: Double): this.type = {
    set(downSampleFraction, value)
  }

  private[op] def getDownSampleFraction: Double = $(downSampleFraction)

  /**
   * Whether or not positive data is in minority
   * Value should be in true or false
   *
   * @group param
   */
  private[op] final val isPositiveSmall = new BooleanParam(this, "isPositiveSmall",
    "whether or not positive data is in minority")

  private[op] def setIsPositiveSmall(value: Boolean): this.type = {
    set(isPositiveSmall, value)
  }

  private[op] def getIsPositiveSmall: Boolean = $(isPositiveSmall)

  /**
   * Sampling fraction in case the data is already balanced, but the size is greater than maxTrainingSample
   * Value should be in ]0.0, 1.0]
   *
   * @group param
   */
  private[op] final val alreadyBalancedFraction = new DoubleParam(this, "alreadyBalancedFraction",
    "sampling fraction in case the data is already balanced, but the size is greater than maxTrainingSample",
    ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0, lowerInclusive = false, upperInclusive = true)
  )

  private[op] def setAlreadyBalancedFraction(value: Double): this.type = {
    set(alreadyBalancedFraction, value)
  }

  private[op] def getAlreadyBalancedFraction: Double = $(alreadyBalancedFraction)
}
