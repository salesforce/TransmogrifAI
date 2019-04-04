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

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature._
import com.salesforce.op.stages.impl.preparators.{CorrelationType, CorrelationExclusion, SanityChecker}
import com.salesforce.op.stages.impl.regression.IsotonicRegressionCalibrator
import com.salesforce.op.utils.tuples.RichTuple._
import com.salesforce.op.utils.numeric.Number

import scala.language.postfixOps
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

// scalastyle:off
object RichNumericFeatureLambdas {
  def divide = (i1: OPNumeric[_], i2: OPNumeric[_]) => {
    val result = for {
      x <- i1.toDouble
      y <- i2.toDouble
    } yield x / y

    result filter Number.isValid toReal
  }

  def plus = (i1: OPNumeric[_], i2: OPNumeric[_]) => (i1.toDouble -> i2.toDouble).map(_ + _).toReal

  def minus = (i1: OPNumeric[_], i2: OPNumeric[_]) => {
    val optZ = (i1.toDouble, i2.toDouble) match {
      case (Some(x), Some(y)) => Some(x - y)
      case (Some(x), None) => Some(x)
      case (None, Some(y)) => Some(-y)
      case (None, None) => None
    }
    optZ.toReal
  }

  def multiply = (i1: OPNumeric[_], i2: OPNumeric[_]) => {
    val result = for {
      x <- i1.toDouble
      y <- i2.toDouble
    } yield x * y

    result filter Number.isValid toReal
  }

  def multiplyS(nd: Double) = (r: OPNumeric[_]) => r.toDouble.map(_ * nd).filter(Number.isValid).toReal

  def divideS(nd: Double) = (r: OPNumeric[_]) => r.toDouble.map(_ / nd).filter(Number.isValid).toReal

  def plusS(nd: Double) = (r: OPNumeric[_]) => r.toDouble.map(_ + nd).filter(Number.isValid).toReal

  def minusS(nd: Double) = (r: OPNumeric[_]) => r.toDouble.map(_ - nd).filter(Number.isValid).toReal
}

// scalastyle:on
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
          transformFn = RichNumericFeatureLambdas.divide
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
          transformFn = RichNumericFeatureLambdas.multiply
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
          transformFn = RichNumericFeatureLambdas.plus
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
          transformFn = RichNumericFeatureLambdas.minus
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
      val nd = n.toDouble(v)
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "divideS",
          transformFn = RichNumericFeatureLambdas.divideS(nd),
          lambdaCtorArgs = Array(nd)
        )
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
      val nd = n.toDouble(v)
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "multiplyS",
          transformFn = RichNumericFeatureLambdas.multiplyS(nd),
          lambdaCtorArgs = Array(nd)
        )
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
      val nd = n.toDouble(v)
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "plusS",
          transformFn = RichNumericFeatureLambdas.plusS(nd),
          lambdaCtorArgs = Array(nd)
        )
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
      val nd = n.toDouble(v)
      f.transformWith(
        new UnaryLambdaTransformer[I, Real](
          operationName = "minusS",
          transformFn = RichNumericFeatureLambdas.minusS(nd),
          lambdaCtorArgs = Array(nd)
        )
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
     * @param trackNulls     option to keep track of values that were missing
     * @param trackInvalid   option to keep track of invalid values,
     *                       eg. NaN, -/+Inf or values that fall outside the buckets
     * @param splits         sorted list of split points for bucketizing
     * @param splitInclusion should the splits be left or right inclusive.
     *                       Meaning if x1 and x2 are split points, then for Left the bucket interval is [x1, x2)
     *                       and for Right the bucket interval is (x1, x2].
     * @param bucketLabels   sorted list of labels for the buckets
     */
    def bucketize(
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      splits: Array[Double] = NumericBucketizer.Splits,
      splitInclusion: Inclusion = NumericBucketizer.SplitInclusion,
      bucketLabels: Option[Array[String]] = None
    ): FeatureLike[OPVector] = {
      f.transformWith(
        new NumericBucketizer[T]()
          .setBuckets(splits, bucketLabels)
          .setTrackNulls(trackNulls)
          .setSplitInclusion(splitInclusion)
          .setTrackInvalid(trackInvalid)
      )
    }

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label        label feature
     * @param trackNulls   option to keep track of values that were missing
     * @param trackInvalid option to keep track of invalid values,
     *                     eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain  minimum info gain, one of the stopping criteria of the Decision Tree
     */
    def autoBucketize(
      label: FeatureLike[RealNN],
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = DecisionTreeNumericBucketizer.MinInfoGain
    ): FeatureLike[OPVector] = {
      new DecisionTreeNumericBucketizer[Double, T]()
        .setInput(label, f)
        .setTrackInvalid(trackInvalid)
        .setTrackNulls(trackNulls)
        .setMinInfoGain(minInfoGain).getOutput()
    }

    /**
     * Apply real vectorizer: Converts a sequence of Real features into a vector feature.
     *
     * @param others       other features of same type
     * @param fillValue    value to pull in place of nulls
     * @param trackNulls   keep tract of when nulls occur by adding a second column to the vector with a null indicator
     * @param fillWithMean replace missing values with mean (as apposed to constant provided in fillValue)
     * @param trackInvalid option to keep track of invalid values,
     *                     eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain  minimum info gain, one of the stopping criteria of the Decision Tree for the autoBucketizer
     * @param label        optional label column to be passed into autoBucketizer if present
     * @return a vector feature containing the raw Features with filled missing values and the bucketized
     *         features if a label argument is passed
     */
    def vectorize
    (
      fillValue: Double,
      fillWithMean: Boolean,
      trackNulls: Boolean,
      others: Array[FeatureLike[T]] = Array.empty,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = TransmogrifierDefaults.MinInfoGain,
      label: Option[FeatureLike[RealNN]] = None
    ): FeatureLike[OPVector] = {
      val features = f +: others
      val stage = new RealVectorizer[T]().setInput(features).setTrackNulls(trackNulls)
      if (fillWithMean) stage.setFillWithMean else stage.setFillWithConstant(fillValue)
      val filledValues = stage.getOutput()
      label match {
        case None =>
          filledValues
        case Some(lbl) =>
          val bucketized = features.map(
            _.autoBucketize(label = lbl, trackNulls = false, trackInvalid = trackInvalid, minInfoGain = minInfoGain)
          )
          new VectorsCombiner().setInput(filledValues +: bucketized).getOutput()
      }
    }

    /**
     * Apply ScalerTransformer shortcut.  Applies the scaling function defined by the scalingType and scalingArg params
     *
     * @param scalingType type of scaling function
     * @param scalingArgs arguments to define the scaling function
     * @tparam O Output feature type
     * @return the descaled input cast to type O
     */
    def scale[O <: Real : TypeTag](
      scalingType: ScalingType,
      scalingArgs: ScalingArgs
    ): FeatureLike[O] = {
      new ScalerTransformer[T, O](scalingType = scalingType, scalingArgs = scalingArgs).setInput(f).getOutput()
    }

    /**
     * Apply DescalerTransformer shortcut.  Applies the inverse of the scaling function found in
     * the metadata of the the input feature: scaledFeature
     *
     * @param scaledFeature the feature containing metadata for constructing the scaling used to make this column
     * @tparam I feature type of the input feature: scaledFeature
     * @tparam O output feature type
     * @return the scaled input cast to type O
     */
    def descale[I <: Real : TypeTag, O <: Real : TypeTag](scaledFeature: FeatureLike[I]): FeatureLike[O] = {
      new DescalerTransformer[T, I, O]().setInput(f, scaledFeature).getOutput()
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
     * @param featureVector          feature vector
     * @param checkSample            Rate to downsample the data for statistical calculations (note: actual sampling
     *                               will not be exact due to Spark's dataset sampling behavior)
     * @param sampleSeed             Seed to use when sampling
     * @param sampleLowerLimit       Lower limit on number of samples in downsampled data set (note: sample limit
     *                               will not be exact, due to Spark's dataset sampling behavior)
     * @param sampleUpperLimit       Upper limit on number of samples in downsampled data set (note: sample limit
     *                               will not be exact, due to Spark's dataset sampling behavior)
     * @param maxCorrelation         Maximum correlation (absolute value) allowed between a feature in the
     *                               feature vector and the label
     * @param minCorrelation         Minimum correlation (absolute value) allowed between a feature in the
     *                               feature vector and the label
     * @param correlationType        Which coefficient to use for computing correlation
     * @param minVariance            Minimum amount of variance allowed for each feature and label
     * @param removeBadFeatures      If set to true, this will automatically remove all the bad features
     *                               from the feature vector
     * @param removeFeatureGroup     remove all features descended from a parent feature
     * @param protectTextSharedHash  protect text shared hash from related null indicators and other hashes
     * @param maxRuleConfidence      Maximum allowed confidence of association rules in categorical variables.
     *                               A categorical variable will be removed if there is a choice where the maximum
     *                               confidence is above this threshold, and the support for that choice is above the
     *                               min rule support parameter, defined below.
     * @param minRequiredRuleSupport Categoricals can be removed if an association rule is found between one of the
     *                               choices and a categorical label where the confidence of that rule is above
     *                               maxRuleConfidence and the support fraction of that choice is above minRuleSupport.
     * @param featureLabelCorrOnly   If true, then only calculate correlations between features and label instead of
     *                               the entire correlation matrix which includes all feature-feature correlations
     * @param correlationExclusion   Setting for what categories of feature vector columns to exclude from the
     *                               correlation calculation (eg. hashed text features)
     * @param categoricalLabel       If true, treat label as categorical. If not set, check number of distinct labels to
     *                               decide whether a label should be treated categorical.
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
      correlationType: CorrelationType = SanityChecker.CorrelationTypeDefault,
      minVariance: Double = SanityChecker.MinVariance,
      removeBadFeatures: Boolean = SanityChecker.RemoveBadFeatures,
      removeFeatureGroup: Boolean = SanityChecker.RemoveFeatureGroup,
      protectTextSharedHash: Boolean = SanityChecker.ProtectTextSharedHash,
      maxRuleConfidence: Double = SanityChecker.MaxRuleConfidence,
      minRequiredRuleSupport: Double = SanityChecker.MinRequiredRuleSupport,
      featureLabelCorrOnly: Boolean = SanityChecker.FeatureLabelCorrOnly,
      correlationExclusion: CorrelationExclusion = SanityChecker.CorrelationExclusionDefault,
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
        .setRemoveFeatureGroup(removeFeatureGroup)
        .setProtectTextSharedHash(protectTextSharedHash)
        .setMaxRuleConfidence(maxRuleConfidence)
        .setMinRequiredRuleSupport(minRequiredRuleSupport)
        .setFeatureLabelCorrOnly(featureLabelCorrOnly)
        .setCorrelationExclusion(correlationExclusion)
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
     * @param trackNulls     option to keep track of values that were missing
     * @param trackInvalid   option to keep track of invalid values,
     *                       eg. NaN, -/+Inf or values that fall outside the buckets
     * @param splits         sorted list of split points for bucketizing
     * @param splitInclusion should the splits be left or right inclusive.
     *                       Meaning if x1 and x2 are split points, then for Left the bucket interval is [x1, x2)
     *                       and for Right the bucket interval is (x1, x2].
     * @param bucketLabels   sorted list of labels for the buckets
     */
    def bucketize(
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      splits: Array[Double] = NumericBucketizer.Splits,
      splitInclusion: Inclusion = NumericBucketizer.SplitInclusion,
      bucketLabels: Option[Array[String]] = None
    ): FeatureLike[OPVector] = {
      f.transformWith(
        new NumericBucketizer[T]()
          .setBuckets(splits, bucketLabels)
          .setTrackNulls(trackNulls)
          .setSplitInclusion(splitInclusion)
          .setTrackInvalid(trackInvalid)
      )
    }

    /**
     * Apply a smart bucketizer transformer
     *
     * @param label        label feature
     * @param trackNulls   option to keep track of values that were missing
     * @param trackInvalid option to keep track of invalid values,
     *                     eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain  minimum info gain, one of the stopping criteria of the Decision Tree
     */
    def autoBucketize(
      label: FeatureLike[RealNN],
      trackNulls: Boolean,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = DecisionTreeNumericBucketizer.MinInfoGain
    ): FeatureLike[OPVector] = {
      new DecisionTreeNumericBucketizer[Long, T]()
        .setInput(label, f)
        .setTrackInvalid(trackInvalid)
        .setTrackNulls(trackNulls)
        .setMinInfoGain(minInfoGain).getOutput()
    }

    /**
     * Apply integral vectorizer: Converts a sequence of Integral features into a vector feature.
     *
     * @param others       other features of same type
     * @param fillValue    value to pull in place of nulls
     * @param trackNulls   keep tract of when nulls occur by adding a second column to the vector with a null indicator
     * @param fillWithMode replace missing values with mode (as apposed to constant provided in fillValue)
     * @param trackInvalid option to keep track of invalid values,
     *                     eg. NaN, -/+Inf or values that fall outside the buckets
     * @param minInfoGain  minimum info gain, one of the stopping criteria of the Decision Tree for the autoBucketizer
     * @param label        optional label column to be passed into autoBucketizer if present
     * @return a vector feature containing the raw Features with filled missing values and the bucketized
     *         features if a label argument is passed
     */
    def vectorize
    (
      fillValue: Long,
      fillWithMode: Boolean,
      trackNulls: Boolean,
      others: Array[FeatureLike[T]] = Array.empty,
      trackInvalid: Boolean = TransmogrifierDefaults.TrackInvalid,
      minInfoGain: Double = TransmogrifierDefaults.MinInfoGain,
      label: Option[FeatureLike[RealNN]] = None
    ): FeatureLike[OPVector] = {
      val features = f +: others
      val stage = new IntegralVectorizer[T]().setInput(features).setTrackNulls(trackNulls)
      if (fillWithMode) stage.setFillWithMode else stage.setFillWithConstant(fillValue)
      val filledValues = stage.getOutput()
      label match {
        case None =>
          filledValues
        case Some(lbl) =>
          val bucketized = features.map(
            _.autoBucketize(label = lbl, trackNulls = false, trackInvalid = trackInvalid, minInfoGain = minInfoGain)
          )
          new VectorsCombiner().setInput(filledValues +: bucketized).getOutput()
      }
    }
  }

}
