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
 * Instance that will balance the dataset before splitting
 *
 * @param uid
 */
private[op] class DataBalancer(uid: String = UID[DataBalancer]) extends Splitter(uid = uid) with DataBalancerParams {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Computes the upSample and downSample proportions.
   *
   * @param smallCount        size of minority class data
   * @param bigCount          size of majority class data
   * @param sampleF           targeted fraction of small data
   * @param maxTrainingSample maximum training size
   * @return downSample & upSample proportions
   */
  private[op] def getProportions(
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
   *
   * @param smallData      data with the minority class
   * @param smallCount     smallData size
   * @param bigData        data with the majority class
   * @param bigCount       bigData size
   * @param sampleF        targeted sample fraction of the training dataset
   * @param maxTrainSample maximum size of the dataset
   * @return balanced small and big data split into training and test sets
   *         with downSample and upSample proportions and a boolean of whether or not logging the new counts
   */
  private[op] def getTrainingSplit(
    smallData: Dataset[_],
    smallCount: Long,
    bigData: Dataset[_],
    bigCount: Long,
    sampleF: Double,
    maxTrainSample: Int
  ): (Dataset[LabelFeaturesKey], Double, Double) = {

    import smallData.sparkSession.implicits._
    val balancerSeed = getSeed

    log.info(s"Sampling data to get $sampleF split versus $smallCount small and $bigCount big")

    val trainProportion = 1.0 - getReserveTestFraction

    // get downSample and upSample proportion of the training set
    val (downSample, upSample) = getProportions(smallCount * trainProportion,
      bigCount * trainProportion, sampleF, maxTrainSample)

    val bigDataTrain = bigData.sample(withReplacement = false, downSample, seed = balancerSeed)

    val smallDataTrain = upSample match {
      case u if u > 1.0 => smallData.sample(withReplacement = true, u, seed = balancerSeed)
      case 1.0 => smallData // if upSample == 1.0, no need to upSample
      case u => smallData.sample(withReplacement = false, u, seed = balancerSeed) // downsample instead
    }

    val train = smallDataTrain.as[LabelFeaturesKey].union(bigDataTrain.as[LabelFeaturesKey])

    (train, downSample, upSample)
  }


  /**
   * Split into a training set and a test set and balance the training set
   *
   * @param data
   * @return balanced training set and a test set
   */
  final override def prepare(data: Dataset[LabelFeaturesKey]): ModelData = {

    val ds = data.persist()

    val Array(negativeData, positiveData) = Array(0.0, 1.0).map(label => ds.filter(_._1 == label).persist())

    val positiveCount = positiveData.count()
    val negativeCount = negativeData.count()
    val totalCount = positiveCount + negativeCount

    // feed metadata with counts and sample fraction
    val metaDataBuilder = new MetadataBuilder()
    metaDataBuilder.putLong(ModelSelectorBaseNames.Positive, positiveCount)
    metaDataBuilder.putLong(ModelSelectorBaseNames.Negative, negativeCount)
    metaDataBuilder.putDouble(ModelSelectorBaseNames.Desired, $(sampleFraction))


    log.info(s"Data has $positiveCount positive and $negativeCount negative.")

    val (smallCount, smallData, bigCount, bigData) =
      if (positiveCount < negativeCount) (positiveCount, positiveData, negativeCount, negativeData)
      else (negativeCount, negativeData, positiveCount, positiveData)

    val trainData = {

      if (smallCount.toDouble / totalCount.toDouble >= $(sampleFraction)) {
        // if the current fraction is superior than the one expected
        log.info(
          s"Not resampling data: $smallCount small count and $bigCount big count is greater than" +
            s" requested ${$(sampleFraction)}"
        )

        // if data is too big downsample
        val maxTrainingSample = getMaxTrainingSample
        val balancerSeed = getSeed

          if (maxTrainingSample < totalCount) {
            data.sample(withReplacement = false, maxTrainingSample / totalCount.toDouble, seed = balancerSeed)
          } else data

      } else {

        if (smallCount < 100 || (smallCount + bigCount) < 500) {
          log.warn("!!!Attention!!! - there is not enough data to build a good model!")
        }

        val (train, downSample, upSample) =
          getTrainingSplit(smallData, smallCount, bigData, bigCount, getSampleFraction, getMaxTrainingSample)

        // feed metadata with upsample and downsample
        metaDataBuilder.putDouble(ModelSelectorBaseNames.UpSample, upSample)
        metaDataBuilder.putDouble(ModelSelectorBaseNames.DownSample, downSample)

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

        train
      }
    }

    new ModelData(trainData, metaDataBuilder)
  }

  final override def copy(extra: ParamMap): DataBalancer = {
    val copy = new DataBalancer(uid)
    copyValues(copy, extra)
  }
}

private[impl] trait DataBalancerParams extends Params {

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
}
