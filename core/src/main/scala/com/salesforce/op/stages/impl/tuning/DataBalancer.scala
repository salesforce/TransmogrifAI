/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import org.slf4j.LoggerFactory


/**
 * Case class for Training & test sets
 *
 * @param train       training set
 * @param test        test set
 * @param metadata    metadata
 * @param hasLeakage if ther is leakage after resampling
 */
case class ModelSplitData(train: Dataset[_], test: Dataset[_], metadata: Metadata, hasLeakage: Boolean)

/**
 * Abstract class that will carry on the creation of training set + test set
 *
 * @param uid
 */
abstract class Splitter(val uid: String) extends SplitterParams {

  /**
   * function to use to create the training set and test set.
   *
   * @param data
   * @return Training set test set
   */
  private[impl] def split(data: Dataset[(Double, Vector, Double)]): ModelSplitData

  /**
   * Splits randomly a dataset into two given a fraction.
   *
   * @param dataset  data to split
   * @param fraction proportion of data to split
   * @return two datasets of sizes (fraction * originalSize, (1 - fraction) * originalSize)
   */
  private[op] def randomSplit(dataset: Dataset[_], fraction: Double): (Dataset[_], Dataset[_]) = {
    val splitData = dataset.randomSplit(Array(fraction, 1.0 - fraction), seed = $(seed))
    (splitData(0), splitData(1))
  }

}

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
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault
  ): DataSplitter = {
    new DataSplitter()
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)

  }
}

/**
 * Instance that will only split the data
 *
 * @param uid
 */
class DataSplitter(uid: String = UID[DataSplitter]) extends Splitter(uid = uid) {
  final override private[impl] def split(dataset: Dataset[(Double, Vector, Double)]): ModelSplitData = {
    val (dataTrain, dataTest) = randomSplit(dataset, 1 - getReserveTestFraction)
    ModelSplitData(dataTrain.persist(), dataTest.persist(), new MetadataBuilder().build(), false)
  }

  final override def copy(extra: ParamMap): DataSplitter = {
    val copy = new DataSplitter(uid)
    copyValues(copy, extra)
  }
}

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
    new DataBalancer().setSampleFraction(sampleFraction)
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
class DataBalancer(uid: String = UID[DataBalancer]) extends Splitter(uid = uid) with DataBalancerParams {

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
  private[op] def getProportions(smallCount: Double, bigCount: Double, sampleF: Double,
    maxTrainingSample: Int): (Double, Double) = {

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
  private[op] def getTrainingSplit(smallData: Dataset[_], smallCount: Long,
    bigData: Dataset[_], bigCount: Long, sampleF: Double, maxTrainSample: Int):
  (Dataset[_], Dataset[_], Dataset[_], Dataset[_],
    Double, Double)
  = {

    val defaultTestFraction = $(reserveTestFraction)
    val balancerSeed = $(seed)

    log.info(s"Sampling data to get $sampleF split versus $smallCount small and $bigCount big")

    val trainProportion = 1.0 - defaultTestFraction
    // get downSample and upSample proportion of the training set
    val (downSample, upSample) = getProportions(smallCount * trainProportion,
      bigCount * trainProportion, sampleF, maxTrainSample)


    val (bigDataTrainRaw, bigDataTest) = randomSplit(bigData, trainProportion)
    val bigDataTrain = bigDataTrainRaw.sample(withReplacement = false, downSample, seed = balancerSeed)

    val (smallDataTrainRaw, smallDataTest) = randomSplit(smallData, trainProportion) // Do split before upsampling

    val smallDataTrain = upSample match {
      case u if (u > 1.0) => smallDataTrainRaw.sample(withReplacement = true, u, seed = balancerSeed)
      case 1.0 => smallDataTrainRaw // if upSample == 1.0, no need to upSample
      case u => smallDataTrainRaw.sample(withReplacement = false, u, seed = balancerSeed) // downsample instead
    }

    (bigDataTrain, bigDataTest, smallDataTrain, smallDataTest, downSample, upSample)
  }


  /**
   * Split into a training set and a test set and balance the training set
   *
   * @param dataset
   * @return balanced training set and a test set
   */
  final override private[impl] def split(dataset: Dataset[(Double, Vector, Double)]): ModelSplitData = {
    // scalastyle:off
    import dataset.sparkSession.implicits._
    // scalastyle:on

    val ds = dataset.persist()

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

    val (smallCount, bigCount) = if (positiveCount < negativeCount) (positiveCount, negativeCount)
    else (negativeCount, positiveCount)

    val (trainData, testData, hasLeakage) =
      if (smallCount.toDouble / totalCount.toDouble >= $(sampleFraction)) {
        // if the current fraction is superior than the one expected
        log.info(
          s"Not resampling data: $smallCount small count and $bigCount big count is greater than" +
            s" requested ${$(sampleFraction)}"
        )

        // either take sample max or size or reserve test fraction
        val sampleProportion = math.min($(maxTrainingSample).toDouble / totalCount, 1.0 - $(reserveTestFraction))
        val (train, test) = randomSplit(ds, sampleProportion)

        (train, test, false)

      } else {

        if (smallCount < 100 || (smallCount + bigCount) < 500) {
          log.warn("!!!Attention!!! - there is not enough data to build a good model!")
        }

        val (smallData, bigData) = if (positiveCount < negativeCount) (positiveData, negativeData)
        else (negativeData, positiveData)

        val (bigDataTrain, bigDataTest, smallDataTrain, smallDataTest, downSample, upSample) =
          getTrainingSplit(smallData, smallCount, bigData, bigCount, $(sampleFraction), $(maxTrainingSample))

        // feed metadata with upsample and downsample
        metaDataBuilder.putDouble(ModelSelectorBaseNames.UpSample, upSample)
        metaDataBuilder.putDouble(ModelSelectorBaseNames.DownSample, downSample)


        val (train, test) = (
          smallDataTrain.as[(Double, Vector, Double)].union(bigDataTrain.as[(Double, Vector, Double)]),
          bigDataTest.as[(Double, Vector, Double)].union(smallDataTest.as[(Double, Vector, Double)])
        )


        val hasLeakage = upSample > 1.0



        val (posFraction, negFraction) =
          if (positiveCount < negativeCount) (upSample, downSample)
          else (downSample, upSample)

        val (newPositiveCount, newNegativeCount) = (
          math.rint(positiveCount * posFraction),
          math.rint(negativeCount * negFraction)
        )

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

        (train, test, hasLeakage)
      }

    ModelSplitData(trainData.persist(), testData.persist(), metaDataBuilder.build(), hasLeakage)

  }

  final override def copy(extra: ParamMap): DataBalancer = {
    val copy = new DataBalancer(uid)
    copyValues(copy, extra)
  }
}

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
 * Instance that will balance the dataset before splitting.
 * Creates instance that will split data into training and test set filtering out any labels that don't
 * meet the minimum fraction cutoff or fall in the top N labels specified.
 *
 * @param uid
 */
class DataCutter(uid: String = UID[DataCutter]) extends Splitter(uid = uid) with DataCutterParams {

  @transient private lazy val log = LoggerFactory.getLogger(this.getClass)

  /**
   * function to use to create the training set and test set.
   *
   * @param data
   * @return Training set test set
   */
  override private[impl] def split(data: Dataset[(Double, Vector, Double)]) = {
    val minLabelFract = getMinLabelFraction
    val maxLabels = getMaxLabelCategories

    // scalastyle:off
    import data.sparkSession.implicits._
    // scalastyle:on

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

    val (dataTrain, dataTest) = randomSplit(dataUse, 1 - getReserveTestFraction)
    ModelSplitData(dataTrain.persist(), dataTest.persist(), metadata.build(), false)
  }

  override def copy(extra: ParamMap): DataCutter = {
    val copy = new DataCutter(uid)
    copyValues(copy, extra)
  }
}


private[impl] trait SplitterParams extends Params {


  /**
   * Seed for data splitting
   *
   * @group param
   */
  final val seed = new LongParam(this, "seed", "seed for the splitting/balancing")
  setDefault(seed, SplitterParamsDefault.seedDefault)

  def setSeed(value: Long): this.type = {
    set(seed, value)
  }

  def getSeed: Long = $(seed)

  /**
   * Fraction of data to reserve for test
   * Default is 0.1
   *
   * @group param
   */
  final val reserveTestFraction = new DoubleParam(this, "reserveTestFraction", "fraction of data to reserve for test",
    ParamValidators.inRange(lowerBound = 0, upperBound = 1, lowerInclusive = true, upperInclusive = false)
  )
  setDefault(reserveTestFraction, SplitterParamsDefault.ReserveTestFractionDefault)

  def setReserveTestFraction(value: Double): this.type = {
    set(reserveTestFraction, value)
  }

  def getReserveTestFraction: Double = $(reserveTestFraction)
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

object SplitterParamsDefault {

  def seedDefault: Long = util.Random.nextLong

  val ReserveTestFractionDefault = 0.1
  val SampleFractionDefault = 0.1
  val MaxTrainingSampleDefault = 100000
  val MaxLabelCategoriesDefault = 100
  val MinLabelFractionDefault = 0.0
}
