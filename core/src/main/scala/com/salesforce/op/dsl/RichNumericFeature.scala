/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature._
import com.salesforce.op.stages.impl.preparators.{CorrelationType, SanityChecker}
import com.salesforce.op.stages.impl.regression.IsotonicRegressionCalibrator
import com.salesforce.op.utils.tuples.RichTuple._

import scala.language.postfixOps
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/**
 * Enrichment functions for Numeric Feature
 */
trait RichNumericFeature {

  /**
   * Enrichment functions for Numeric Feature
   *
   * @param f FeatureLike
   * @tparam I input type
   */
  implicit class RichNumericFeature[I <: OPNumeric[_] : TypeTag : ClassTag](val f: FeatureLike[I]) {

    /**
     * Apply Divide transformer shortcut function
     *
     * Divide function truth table (Real as example):
     *
     * Real.empty / Real.empty = Real.empty
     * Real.empty / Real(x)    = Real.empty
     * Real(x)    / Real.empty = Real.empty
     * Real(x)    / Real(y)    = Real(x / y) filter ("is not NaN or Infinity")
     *
     * @param that feature to divide by
     * @tparam I2 that feature output type
     * @return transformed feature
     */
    def /[I2 <: OPNumeric[_] : TypeTag](that: FeatureLike[I2]): FeatureLike[Real] = {
      f.transformWith[I2, Real](
        stage = new BinaryLambdaTransformer[I, I2, Real](
          operationName = "divide",
          transformFn = (i1: I, i2: I2) => {
            val result = for {
              x <- i1.toDouble
              y <- i2.toDouble
            } yield x / y

            result filter Number.isValid toReal
          }
        ),
        f = that
      )
    }

    /**
     * Apply Multiply transformer shortcut function
     *
     * Multiply function truth table (Real as example):
     *
     * Real.empty * Real.empty = Real.empty
     * Real.empty * Real(x)    = Real.empty
     * Real(x)    * Real.empty = Real.empty
     * Real(x)    * Real(y)    = Real(x * y) filter ("is not NaN or Infinity")
     *
     * @param that feature to divide by
     * @tparam I2 that feature output type
     * @return transformed feature
     */
    def *[I2 <: OPNumeric[_] : TypeTag](that: FeatureLike[I2]): FeatureLike[Real] = {
      f.transformWith[I2, Real](
        stage = new BinaryLambdaTransformer[I, I2, Real](
          operationName = "multiply",
          transformFn = (i1: I, i2: I2) => {
            val result = for {
              x <- i1.toDouble
              y <- i2.toDouble
            } yield x * y

            result filter Number.isValid toReal
          }
        ),
        f = that
      )
    }

    /**
     * Apply Plus transformer shortcut function
     *
     * Plus function truth table (Real as example):
     *
     * Real.empty + Real.empty = Real.empty
     * Real.empty + Real(x)    = Real(x)
     * Real(x)    + Real.empty = Real(x)
     * Real(x)    + Real(y)    = Real(x + y)
     *
     * @param that feature to divide by
     * @tparam I2 that feature output type
     * @return transformed feature
     */
    def +[I2 <: OPNumeric[_] : TypeTag](that: FeatureLike[I2]): FeatureLike[Real] = {
      f.transformWith[I2, Real](
        stage = new BinaryLambdaTransformer[I, I2, Real](
          operationName = "plus",
          transformFn = (i1: I, i2: I2) => (i1.toDouble -> i2.toDouble).map(_ + _).toReal
        ),
        f = that
      )
    }

    /**
     * Apply Minus transformer shortcut function
     *
     * Minus function truth table (Real as example):
     *
     * Real.empty - Real.empty = Real.empty
     * Real.empty - Real(x)    = Real(-x)
     * Real(x)    - Real.empty = Real(x)
     * Real(x)    - Real(y)    = Real(x - y)
     *
     * @param that feature to divide by
     * @tparam I2 that feature output type
     * @return transformed feature
     */
    def -[I2 <: OPNumeric[_] : TypeTag](that: FeatureLike[I2]): FeatureLike[Real] = {
      f.transformWith[I2, Real](
        stage = new BinaryLambdaTransformer[I, I2, Real](
          operationName = "minus",
          transformFn = (i1: I, i2: I2) => {
            val optZ = (i1.toDouble, i2.toDouble) match {
              case (Some(x), Some(y)) => Some(x - y)
              case (Some(x), None) => Some(x)
              case (None, Some(y)) => Some(-y)
              case (None, None) => None
            }
            optZ.toReal
          }
        ),
        f = that
      )
    }

    /**
     * Apply Divide scalar transformer shortcut function
     *
     * @param v scalar value
     * @param n value converter
     * @tparam N value type
     * @return transformed feature
     */
    def /[N](v: N)(implicit n: Numeric[N]): FeatureLike[Real] = {
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "divideS",
          transformFn = r => r.toDouble.map(_ / n.toDouble(v)).filter(Number.isValid).toReal)
      )
    }

    /**
     * Apply Multiply scalar transformer shortcut function
     *
     * @param v scalar value
     * @param n value converter
     * @tparam N value type
     * @return transformed feature
     */
    def *[N](v: N)(implicit n: Numeric[N]): FeatureLike[Real] = {
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "multiplyS",
          transformFn = r => r.toDouble.map(_ * n.toDouble(v)).filter(Number.isValid).toReal)
      )
    }

    /**
     * Apply Plus scalar transformer shortcut function
     *
     * @param v scalar value
     * @param n value converter
     * @tparam N value type
     * @return transformed feature
     */
    def +[N](v: N)(implicit n: Numeric[N]): FeatureLike[Real] = {
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "plusS",
          transformFn = r => r.toDouble.map(_ + n.toDouble(v)).toReal)
      )
    }

    /**
     * Apply Minus scalar transformer shortcut function
     *
     * @param v scalar value
     * @param n value converter
     * @tparam N value type
     * @return transformed feature
     */
    def -[N](v: N)(implicit n: Numeric[N]): FeatureLike[Real] = {
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "minusS",
          transformFn = r => r.toDouble.map(_ - n.toDouble(v)).toReal)
      )
    }

  }

  /**
   * Enrichment functions for Real Feature
   *
   * @param f FeatureLike
   */
  implicit class RichRealFeature[T <: Real : TypeTag](val f: FeatureLike[T])(implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Fill missing values with mean
     *
     * @param default default value is the whole feature is filled with missing values
     * @return transformed feature of type RealNN
     */
    def fillMissingWithMean(default: Double = 0.0): FeatureLike[RealNN] = {
      f.transformWith(new FillMissingWithMean[Double, T]().setDefaultValue(default))
    }

    /**
     * Apply NumericBucketizer transformer shortcut function
     *
     * @param trackNulls   option to keep track of values that were missing
     * @param splits       sorted list of split points for bucketizing
     * @param bucketLabels sorted list of labels for the buckets
     */
    def bucketize(
      trackNulls: Boolean,
      splits: Array[Double] = NumericBucketizer.Splits,
      bucketLabels: Option[Array[String]] = None
    ): FeatureLike[OPVector] = {
      f.transformWith(new NumericBucketizer[Double, T]().setBuckets(splits, bucketLabels).setTrackNulls(trackNulls))
    }

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label      label feature
     * @param trackNulls option to keep track of values that were missing
     */
    def autoBucketize(label: FeatureLike[RealNN], trackNulls: Boolean): FeatureLike[OPVector] = {
      new DecisionTreeNumericBucketizer[Double, T]().setTrackNulls(trackNulls).setInput(label, f).getOutput()
    }

    /**
     * Apply real vectorizer: Converts a sequence of Real features into a vector feature.
     *
     * @param others       other features of same type
     * @param fillValue    value to pull in place of nulls
     * @param trackNulls   keep tract of when nulls occur by adding a second column to the vector with a null indicator
     * @param fillWithMean replace missing values with mean (as apposed to constant provided in fillValue)
     * @return
     */
    def vectorize
    (
      fillValue: Double,
      fillWithMean: Boolean,
      trackNulls: Boolean,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      val stage = new RealVectorizer[T]()
        .setInput(f +: others)
        .setTrackNulls(trackNulls)
      if (fillWithMean) stage.setFillWithMean else stage.setFillWithConstant(fillValue)
      stage.getOutput()
    }

  }


  /**
   * Enrichment functions for Real non nullable Feature
   *
   * @param f FeatureLike
   */
  implicit class RichRealNNFeature(val f: FeatureLike[RealNN]) {
    /**
     * Z-normalization shortcut function using OpStandardScaler.
     */
    def zNormalize(): FeatureLike[RealNN] = {
      f.transformWith(new OpScalarStandardScaler())
    }

    /**
     * Apply PercentileBucketizer transformer shortcut function. Will rescale values into the
     * specified number of bins (default it 100)
     *
     * @param buckets number of bins to scale into
     */
    def toPercentile(buckets: Int = 100): FeatureLike[RealNN] = {
      f.transformWith(stage = new PercentileCalibrator().setExpectedNumBuckets(buckets))
    }

    /**
     * Apply standard isotonic regression transformer shortcut function.
     *
     * @param label      feature to calibrate against
     * @param isIsotonic increasing default true or decreasing
     * @return recalibrated feature
     */
    def toIsotonicCalibrated(label: FeatureLike[RealNN], isIsotonic: Boolean = true): FeatureLike[RealNN] = {
      val estimator = new IsotonicRegressionCalibrator().setIsotonic(isIsotonic)
      label.transformWith[RealNN, RealNN](stage = estimator, f = f)
    }

    /**
     * Apply [[OpIndexToStringNoFilter]] transformer.
     *
     * A transformer that maps a feature of indices back to a new feature of corresponding text values.
     * The index-string mapping is either from the ML attributes of the input feature,
     * or from user-supplied labels (which take precedence over ML attributes).
     *
     * @see [[OpStringIndexerNoFilter]] for converting text into indices
     *
     * @param labels        Optional array of labels specifying index-string mapping.
     *                      If not provided or if empty, then metadata from input feature is used instead.
     * @param unseenName    name to give strings that appear in transform but not in fit
     * @param handleInvalid how to transform values not seen in fitting
     * @return deindexed text feature
     */
    def deindexed(
      labels: Array[String] = Array.empty,
      unseenName: String = OpIndexToStringNoFilter.unseenDefault,
      handleInvalid: IndexToStringHandleInvalid = IndexToStringHandleInvalid.NoFilter
    ): FeatureLike[Text] = {
      handleInvalid match {
        case IndexToStringHandleInvalid.NoFilter => f.transformWith(
          new OpIndexToStringNoFilter().setLabels(labels).setUnseenName(unseenName)
        )
        case IndexToStringHandleInvalid.Error => f.transformWith(new OpIndexToString().setLabels(labels))
      }
    }

    /**
     * Apply [[SanityChecker]] estimator.
     * It checks for potential problems with computed features in a supervized learning setting.
     *
     * @param featureVector     feature vector
     * @param checkSample       Rate to downsample the data for statistical calculations (note: actual sampling
     *                          will not be exact due to Spark's dataset sampling behavior)
     * @param sampleSeed        Seed to use when sampling
     * @param sampleLowerLimit  Lower limit on number of samples in downsampled data set (note: sample limit
     *                          will not be exact, due to Spark's dataset sampling behavior)
     * @param sampleUpperLimit  Upper limit on number of samples in downsampled data set (note: sample limit
     *                          will not be exact, due to Spark's dataset sampling behavior)
     * @param maxCorrelation    Maximum correlation (absolute value) allowed between a feature in the
     *                          feature vector and the label
     * @param minCorrelation    Minimum correlation (absolute value) allowed between a feature in the
     *                          feature vector and the label
     * @param correlationType   Which coefficient to use for computing correlation
     * @param minVariance       Minimum amount of variance allowed for each feature and label
     * @param removeBadFeatures If set to true, this will automatically remove all the bad features
     *                          from the feature vector
     * @return sanity checked feature vector
     */
    // scalastyle:off
    def sanityCheck(
      featureVector: FeatureLike[OPVector],
      checkSample: Double = SanityChecker.CheckSample,
      sampleSeed: Long = SanityChecker.SampleSeed,
      sampleLowerLimit: Int = SanityChecker.SampleLowerLimit,
      sampleUpperLimit: Int = SanityChecker.SampleUpperLimit,
      maxCorrelation: Double = SanityChecker.MaxCorrelation,
      minCorrelation: Double = SanityChecker.MinCorrelation,
      maxCramersV: Double = SanityChecker.MaxCramersV,
      correlationType: CorrelationType = SanityChecker.CorrelationType,
      minVariance: Double = SanityChecker.MinVariance,
      removeBadFeatures: Boolean = SanityChecker.RemoveBadFeatures,
      categoricalLabel: Option[Boolean] = None
    ): FeatureLike[OPVector] = {
      // scalastyle:on
      val checker = new SanityChecker()
        .setCheckSample(checkSample)
        .setSampleSeed(sampleSeed)
        .setSampleLowerLimit(sampleLowerLimit)
        .setSampleUpperLimit(sampleUpperLimit)
        .setMaxCorrelation(maxCorrelation)
        .setMinCorrelation(minCorrelation)
        .setMaxCramersV(maxCramersV)
        .setCorrelationType(correlationType)
        .setMinVariance(minVariance)
        .setRemoveBadFeatures(removeBadFeatures)
        .setInput(f, featureVector)

      categoricalLabel.foreach(checker.setCategoricalLabel)

      checker.getOutput()
    }

    /**
     * Apply real vectorizer: Converts a sequence of RealNN features into a vector feature.
     *
     * @param others other features of same type
     * @return
     */
    def vectorize(others: Array[FeatureLike[RealNN]] = Array.empty): FeatureLike[OPVector] =
      new RealNNVectorizer().setInput(f +: others).getOutput()

  }


  /**
   * Enrichment functions for Binary Feature
   *
   * @param f FeatureLike
   */
  implicit class RichBinaryFeature(val f: FeatureLike[Binary]) {

    /**
     * Fill missing values with mean
     *
     * @param default default value is the whole feature is filled with missing values
     * @return transformed feature of type RealNN
     */
    def fillMissingWithMean(default: Double = 0.0): FeatureLike[RealNN] = {
      f.transformWith(new FillMissingWithMean[Boolean, Binary]().setDefaultValue(default))
    }

    /**
     * Apply binary vectorizer
     *
     * @param others     other features of same type
     * @param fillValue  value to pull in place of nulls
     * @param trackNulls keep tract of when nulls occurs by adding a second column to the vector with a null indicator
     * @return
     */
    def vectorize
    (
      fillValue: Boolean,
      trackNulls: Boolean,
      others: Array[FeatureLike[Binary]] = Array.empty
    ): FeatureLike[OPVector] =
      new BinaryVectorizer().setInput(f +: others).setFillValue(fillValue).setTrackNulls(trackNulls).getOutput()
  }


  /**
   * Enrichment functions for Integral Feature
   *
   * @param f FeatureLike
   */
  implicit class RichIntegralFeature[T <: Integral : TypeTag](val f: FeatureLike[T])
    (implicit val ttiv: TypeTag[T#Value]) {

    /**
     * Fill missing values with mean
     *
     * @param default default value is the whole feature is filled with missing values
     * @return transformed feature of type RealNN
     */
    def fillMissingWithMean(default: Double = 0.0): FeatureLike[RealNN] = {
      f.transformWith(new FillMissingWithMean[Long, T]().setDefaultValue(default))
    }

    /**
     * Apply NumericBucketizer transformer shortcut function
     *
     * @param trackNulls   option to keep track of values that were missing
     * @param splits       sorted list of split points for bucketizing
     * @param bucketLabels sorted list of labels for the buckets
     */
    def bucketize(
      trackNulls: Boolean,
      splits: Array[Double] = NumericBucketizer.Splits,
      bucketLabels: Option[Array[String]] = None
    ): FeatureLike[OPVector] = {
      f.transformWith(new NumericBucketizer[Long, T]().setBuckets(splits, bucketLabels).setTrackNulls(trackNulls))
    }

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label      label feature
     * @param trackNulls option to keep track of values that were missing
     */
    def autoBucketize(label: FeatureLike[RealNN], trackNulls: Boolean): FeatureLike[OPVector] = {
      new DecisionTreeNumericBucketizer[Long, T]().setTrackNulls(trackNulls).setInput(label, f).getOutput()
    }

    /**
     * Apply integral vectorizer: Converts a sequence of Integral features into a vector feature.
     *
     * @param others       other features of same type
     * @param fillValue    value to pull in place of nulls
     * @param trackNulls   keep tract of when nulls occur by adding a second column to the vector with a null indicator
     * @param fillWithMode replace missing values with mode (as apposed to constant provided in fillValue)
     * @return
     */
    def vectorize
    (
      fillValue: Long,
      fillWithMode: Boolean,
      trackNulls: Boolean,
      others: Array[FeatureLike[T]] = Array.empty
    ): FeatureLike[OPVector] = {
      val stage = new IntegralVectorizer().setInput(f +: others).setTrackNulls(trackNulls)
      if (fillWithMode) stage.setFillWithMode else stage.setFillWithConstant(fillValue)
      stage.getOutput()
    }
  }

}

object Number extends Serializable {
  def isValid(x: Double): Boolean = !x.isNaN && !x.isInfinity
}
