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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureLike, OPFeature, TransientFeature}
import com.salesforce.op.stages.OpPipelineStageBase
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata, SequenceAggregators}
import com.salesforce.op.utils.text.TextUtils
import org.slf4j.LoggerFactory
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.{Metadata, StructField}
import org.apache.spark.sql.{Dataset, Encoders}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._

/**
 * Transmogrifier Defaults trait allows injection of params into Transmogrifier
 */
private[op] trait TransmogrifierDefaults {
  val NullString: String = OpVectorColumnMetadata.NullString
  val OtherString: String = OpVectorColumnMetadata.OtherString
  val DefaultNumOfFeatures: Int = 512
  val MaxNumOfFeatures: Int = 16384
  val DateListDefault: DateListPivot = DateListPivot.SinceLast
  val ReferenceDate: org.joda.time.DateTime = DateTimeUtils.now()
  val TopK: Int = 20
  val MinSupport: Int = 10
  val FillValue: Int = 0
  val BinaryFillValue: Boolean = false
  val HashWithIndex: Boolean = false
  val PrependFeatureName: Boolean = true
  val HashSpaceStrategy: HashSpaceStrategy = com.salesforce.op.stages.impl.feature.HashSpaceStrategy.Auto
  val CleanText: Boolean = true
  val CleanKeys: Boolean = false
  val HashAlgorithm: HashAlgorithm = com.salesforce.op.stages.impl.feature.HashAlgorithm.MurMur3
  val BinaryFreq: Boolean = false
  val FillWithMode: Boolean = true
  val FillWithMean: Boolean = true
  val TrackNulls: Boolean = true
  val TrackInvalid: Boolean = false
  val TrackTextLen: Boolean = false
  val MinDocFrequency: Int = 0
  val MaxPercentCardinality = OpOneHotVectorizer.MaxPctCardinality
  // Default is to fill missing Geolocations with the mean, but if fillWithConstant is chosen, use this
  val DefaultGeolocation: Geolocation = Geolocation(0.0, 0.0, GeolocationAccuracy.Unknown)
  val MinInfoGain: Double = DecisionTreeNumericBucketizer.MinInfoGain
  val MaxCategoricalCardinality = 30
  val CircularDateRepresentations: Seq[TimePeriod] = Seq(TimePeriod.HourOfDay, TimePeriod.DayOfWeek,
    TimePeriod.DayOfMonth, TimePeriod.DayOfYear)

  val DefaultRegion: String = PhoneNumberParser.DefaultRegion
  val AutoDetectLanguage: Boolean = TextTokenizer.AutoDetectLanguage
  val MinTokenLength: Int = TextTokenizer.MinTokenLength
  val ToLowercase: Boolean = TextTokenizer.ToLowercase
}

private[op] object TransmogrifierDefaults extends TransmogrifierDefaults

private[op] case object Transmogrifier {
  val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Vectorize features by type applying default vectorizers
   *
   * @param features input features
   * @param defaults transmogrifier defaults (allows params injection)
   * @param label    optional label feature to be passed into stages that require the label column
   * @return vectorized features grouped by type
   */
  def transmogrify(
    features: Seq[FeatureLike[_]],
    label: Option[FeatureLike[RealNN]] = None
  )(implicit defaults: TransmogrifierDefaults): Iterable[FeatureLike[OPVector]] = {
    import defaults._
    def castSeqAs[U <: FeatureType](f: Seq[FeatureLike[_]]) = f.map(_.asInstanceOf[FeatureLike[U]])

    def castAs[U <: FeatureType](f: Seq[FeatureLike[_]]): (FeatureLike[U], Array[FeatureLike[U]]) = {
      val casted = castSeqAs[U](f)
      casted.head -> casted.tail.toArray
    }

    val featuresByType = features.groupBy(_.wtt.tpe).toSeq.sortBy(_._1.toString) // make features creation deterministic

    featuresByType.flatMap { case (featureType, g) =>
      val res = featureType match {

        // Vector
        case t if t =:= weakTypeOf[OPVector] =>
          castSeqAs[OPVector](g)

        // Lists
        case t if t =:= weakTypeOf[TextList] =>
          val (f, other) = castAs[TextList](g)
          f.vectorize(numTerms = DefaultNumOfFeatures, binary = BinaryFreq, minDocFreq = MinDocFrequency,
            others = other)
        case t if t =:= weakTypeOf[DateList] =>
          val (f, other) = castAs[DateList](g)
          f.vectorize(dateListPivot = DateListDefault, trackNulls = TrackNulls, referenceDate = ReferenceDate,
            others = other)
        case t if t =:= weakTypeOf[DateTimeList] =>
          val (f, other) = castAs[DateTimeList](g)
          f.vectorize(dateListPivot = DateListDefault, trackNulls = TrackNulls, referenceDate = ReferenceDate,
            others = other)
        case t if t =:= weakTypeOf[Geolocation] =>
          val (f, other) = castAs[Geolocation](g)
          f.vectorize(fillWithMean = FillWithMean, trackNulls = TrackNulls, fillValue = DefaultGeolocation,
            others = other)

        // Maps
        case t if t =:= weakTypeOf[Base64Map] =>
          val (f, other) = castAs[Base64Map](g) // TODO make better default
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[BinaryMap] =>
          val (f, other) = castAs[BinaryMap](g)
          f.vectorize(defaultValue = FillValue, cleanKeys = CleanKeys, others = other, trackNulls = TrackNulls)
        case t if t =:= weakTypeOf[ComboBoxMap] =>
          val (f, other) = castAs[ComboBoxMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[CurrencyMap] =>
          val (f, other) = castAs[CurrencyMap](g)
          f.vectorize(defaultValue = FillValue, fillWithMean = FillWithMean, cleanKeys = CleanKeys, others = other,
            trackNulls = TrackNulls, trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, label = label)
        case t if t =:= weakTypeOf[DateMap] =>
          val (f, other) = castAs[DateMap](g)
          f.vectorize(defaultValue = FillValue, cleanKeys = CleanKeys, others = other, trackNulls = TrackNulls,
            referenceDate = ReferenceDate, circularDateReps = CircularDateRepresentations)
        case t if t =:= weakTypeOf[DateTimeMap] =>
          val (f, other) = castAs[DateTimeMap](g)
          f.vectorize(defaultValue = FillValue, cleanKeys = CleanKeys, others = other, trackNulls = TrackNulls,
            referenceDate = ReferenceDate, circularDateReps = CircularDateRepresentations)
        case t if t =:= weakTypeOf[EmailMap] =>
          val (f, other) = castAs[EmailMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[IDMap] =>
          val (f, other) = castAs[IDMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[IntegralMap] =>
          val (f, other) = castAs[IntegralMap](g)
          f.vectorize(defaultValue = FillValue, fillWithMode = FillWithMode, cleanKeys = CleanKeys, others = other,
            trackNulls = TrackNulls, trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, label = label)
        case t if t =:= weakTypeOf[MultiPickListMap] =>
          val (f, other) = castAs[MultiPickListMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[PercentMap] =>
          val (f, other) = castAs[PercentMap](g)
          f.vectorize(defaultValue = FillValue, fillWithMean = FillWithMean, cleanKeys = CleanKeys, others = other,
            trackNulls = TrackNulls, trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, label = label)
        case t if t =:= weakTypeOf[PhoneMap] =>
          val (f, other) = castAs[PhoneMap](g) // TODO make better default
          f.vectorize(defaultRegion = DefaultRegion, others = other, trackNulls = TrackNulls)
        case t if t =:= weakTypeOf[PickListMap] =>
          val (f, other) = castAs[PickListMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[RealMap] =>
          val (f, other) = castAs[RealMap](g)
          f.vectorize(defaultValue = FillValue, fillWithMean = FillWithMean, cleanKeys = CleanKeys, others = other,
            trackNulls = TrackNulls, trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, label = label)
        case t if t =:= weakTypeOf[TextAreaMap] =>
          val (f, other) = castAs[TextAreaMap](g)
          f.smartVectorize(maxCategoricalCardinality = MaxCategoricalCardinality,
            numHashes = DefaultNumOfFeatures, autoDetectLanguage = AutoDetectLanguage,
            minTokenLength = MinTokenLength, toLowercase = ToLowercase,
            prependFeatureName = PrependFeatureName, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls)
        case t if t =:= weakTypeOf[TextMap] =>
          val (f, other) = castAs[TextMap](g)
          f.smartVectorize(maxCategoricalCardinality = MaxCategoricalCardinality,
            numHashes = DefaultNumOfFeatures, autoDetectLanguage = AutoDetectLanguage,
            minTokenLength = MinTokenLength, toLowercase = ToLowercase,
            prependFeatureName = PrependFeatureName, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls)
        case t if t =:= weakTypeOf[URLMap] =>
          val (f, other) = castAs[URLMap](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[CountryMap] =>
          val (f, other) = castAs[CountryMap](g) // TODO make Country specific transformer
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[StateMap] =>
          val (f, other) = castAs[StateMap](g) // TODO make State specific transformer
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[CityMap] =>
          val (f, other) = castAs[CityMap](g) // TODO make City specific transformer
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[PostalCodeMap] =>
          val (f, other) = castAs[PostalCodeMap](g) // TODO make PostalCode specific transformer
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[StreetMap] =>
          val (f, other) = castAs[StreetMap](g) // TODO make Street specific transformer
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, cleanKeys = CleanKeys,
            others = other, trackNulls = TrackNulls, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[GeolocationMap] =>
          val (f, other) = castAs[GeolocationMap](g)
          f.vectorize(cleanKeys = CleanKeys, others = other, trackNulls = TrackNulls)

        // Numerics
        case t if t =:= weakTypeOf[Binary] =>
          val (f, other) = castAs[Binary](g)
          f.vectorize(fillValue = BinaryFillValue, trackNulls = TrackNulls, others = other)
        case t if t =:= weakTypeOf[Currency] =>
          val (f, other) = castAs[Currency](g)
          f.vectorize(fillValue = FillValue, fillWithMean = FillWithMean, trackNulls = TrackNulls,
            trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, others = other, label = label)
        case t if t =:= weakTypeOf[Date] =>
          val (f, other) = castAs[Date](g)
          f.vectorize(dateListPivot = DateListDefault, referenceDate = ReferenceDate, trackNulls = TrackNulls,
            circularDateReps = CircularDateRepresentations, others = other)
        case t if t =:= weakTypeOf[DateTime] =>
          val (f, other) = castAs[DateTime](g)
          f.vectorize(dateListPivot = DateListDefault, referenceDate = ReferenceDate, trackNulls = TrackNulls,
            circularDateReps = CircularDateRepresentations, others = other)
        case t if t =:= weakTypeOf[Integral] =>
          val (f, other) = castAs[Integral](g)
          f.vectorize(fillValue = FillValue, fillWithMode = FillWithMode, trackNulls = TrackNulls,
            trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, others = other, label = label)
        case t if t =:= weakTypeOf[Percent] =>
          val (f, other) = castAs[Percent](g)
          f.vectorize(fillValue = FillValue, fillWithMean = FillWithMean, trackNulls = TrackNulls,
            trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, others = other, label = label)
        case t if t =:= weakTypeOf[Real] =>
          val (f, other) = castAs[Real](g)
          f.vectorize(fillValue = FillValue, fillWithMean = FillWithMean, trackNulls = TrackNulls,
            trackInvalid = TrackInvalid, minInfoGain = MinInfoGain, others = other, label = label)
        case t if t =:= weakTypeOf[RealNN] =>
          val (f, other) = castAs[RealNN](g)
          f.vectorize(other)

        // Sets
        case t if t =:= weakTypeOf[MultiPickList] =>
          val (f, other) = castAs[MultiPickList](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)

        // Text
        case t if t =:= weakTypeOf[Base64] =>
          val (f, other) = castAs[Base64](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[ComboBox] =>
          val (f, other) = castAs[ComboBox](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[Email] =>
          val (f, other) = castAs[Email](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[ID] =>
          val (f, other) = castAs[ID](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[Phone] =>
          val (f, other) = castAs[Phone](g)
          f.vectorize(defaultRegion = DefaultRegion, others = other)
        case t if t =:= weakTypeOf[PickList] =>
          val (f, other) = castAs[PickList](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[Text] =>
          val (f, other) = castAs[Text](g)
          f.smartVectorize(maxCategoricalCardinality = MaxCategoricalCardinality,
            trackNulls = TrackNulls, numHashes = DefaultNumOfFeatures,
            hashSpaceStrategy = defaults.HashSpaceStrategy, autoDetectLanguage = AutoDetectLanguage,
            minTokenLength = MinTokenLength, toLowercase = ToLowercase,
            prependFeatureName = PrependFeatureName, detectSensitive = true, others = other)
        case t if t =:= weakTypeOf[TextArea] =>
          val (f, other) = castAs[TextArea](g)
          f.smartVectorize(maxCategoricalCardinality = MaxCategoricalCardinality,
            trackNulls = TrackNulls, numHashes = DefaultNumOfFeatures,
            hashSpaceStrategy = defaults.HashSpaceStrategy, autoDetectLanguage = AutoDetectLanguage,
            minTokenLength = MinTokenLength, toLowercase = ToLowercase,
            prependFeatureName = PrependFeatureName, detectSensitive = true, others = other)
        case t if t =:= weakTypeOf[URL] =>
          val (f, other) = castAs[URL](g)
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, trackNulls = TrackNulls,
            others = other, maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[Country] =>
          val (f, other) = castAs[Country](g) // TODO make do something smart for Country
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[State] =>
          val (f, other) = castAs[State](g) // TODO make do something smart for State
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[City] =>
          val (f, other) = castAs[City](g) // TODO make do something smart for City
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[PostalCode] =>
          val (f, other) = castAs[PostalCode](g) // TODO make do something smart for PostalCode
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)
        case t if t =:= weakTypeOf[Street] =>
          val (f, other) = castAs[Street](g) // TODO make do something smart for Street
          f.vectorize(topK = TopK, minSupport = MinSupport, cleanText = CleanText, others = other,
            maxPctCardinality = MaxPercentCardinality)

        // Unknown
        case t => throw new IllegalArgumentException(s"No vectorizer available for type $t")
      }

      res match {
        case r: Seq[_] => r.asInstanceOf[Seq[FeatureLike[OPVector]]]
        case r => Seq(r.asInstanceOf[FeatureLike[OPVector]])
      }
    }
  }

  /**
   * Extract feature history map from array of input features
   *
   * @param tf            array of transient features
   * @param thisStageName this stage name
   * @return map from feature name to feature history
   */
  def inputFeaturesToHistory(tf: Array[TransientFeature], thisStageName: String): Map[String, FeatureHistory] =
    tf.map(f => f.name -> FeatureHistory(originFeatures = f.originFeatures, stages = f.stages :+ thisStageName)).toMap

}


trait VectorizerDefaults extends OpPipelineStageBase {
  self: PipelineStage =>

  implicit def booleanToDouble(v: Boolean): Double = if (v) 1.0 else 0.0

  // TODO once track nulls is everywhere put track nulls param here and avoid making the metadata twice
  abstract override def onSetInput(): Unit = {
    super.onSetInput()
    setMetadata(vectorMetadataFromInputFeatures.toMetadata)
  }

  private def vectorMetadata(withNullTracking: Boolean): OpVectorMetadata = {
    val tf = getTransientFeatures()
    val cols =
      if (withNullTracking) tf.flatMap { f => Seq(f.toColumnMetaData(), f.toColumnMetaData(isNull = true)) }
      else tf.map { f => f.toColumnMetaData() }
    OpVectorMetadata(vectorOutputName, cols, Transmogrifier.inputFeaturesToHistory(tf, stageName))
  }

  /**
   * Compute the output vector metadata only from the input features. Vectorizers use this to derive
   * the full vector, including pivot columns or indicator features.
   *
   * @return Vector metadata from input features
   */
  protected def vectorMetadataFromInputFeatures: OpVectorMetadata = vectorMetadata(withNullTracking = false)

  protected def vectorMetadataWithNullIndicators: OpVectorMetadata = vectorMetadata(withNullTracking = true)

  /**
   * Get the name of the output vector
   *
   * @return Output vector name as a string
   */
  protected def vectorOutputName: String = (getOutput(): Array[OPFeature]).head.name

  /**
   * Get the metadata describing the output vector
   *
   * This does ''not'' trigger [[onGetMetadata()]]
   *
   * @return Metadata of output vector
   */
  protected def outputVectorMeta: OpVectorMetadata = OpVectorMetadata(
    StructField(
      vectorOutputName,
      SQLDataTypes.VectorType,
      metadata = $(outputMetadata)
    )
  )
}

case object VectorizerUtils {
  /**
   * Function to reindex a sequence of vectorized categoricals or maps when flattening
   *
   * @param seq sequence of all previous vectorized values
   * @return next index for concatenating the sequence
   */
  def nextIndex(seq: Seq[(Int, Double)]): Int = seq.lastOption.map(_._1 + 1).getOrElse(0)

  /**
   * Function to flatten sequences of (Index, Value) tuples and reindex them sequentially
   * Example:
   * > reindex(Seq(Seq((0,2.0), (1,1.0)), Seq((0,2.0), (5,1.0))))
   * res: Seq((0,2.0), (1,1.0), (2,2.0), (7,1.0))
   *
   * @param seq sequence to flatten
   * @return flattened and reindex values
   */
  def reindex(seq: Seq[Seq[(Int, Double)]]): Seq[(Int, Double)] = {
    val acc = new ArrayBuffer[(Int, Double)](seq.length)
    var next = 0
    seq.foreach(curr => {
      var ind = next
      curr.foreach(c => {
        ind = c._1 + next
        acc += ind -> c._2
      })
      next = ind + 1
    })
    acc
  }

  /**
   * Create a one-hot vector
   *
   * @param pos  position to put 1.0 in the vector
   * @param size size of the one-hot vector
   * @return one-hot vector with 1.0 in position value
   */
  def oneHot(pos: Int, size: Int): Array[Double] = {
    require(pos < size && pos >= 0, s"One-hot index lies outside the bounds of the vector: pos = $pos, size = $size")
    val arr = new Array[Double](size)
    arr(pos) = 1.0
    arr
  }

  /**
   * Function to convert sequences of (Index, Value) tuples into a compressed sparse vector
   *
   * @param seq Input sequence containing tuples of indicies and values
   * @return the vector representation of those values
   */
  def makeSparseVector(seq: Seq[(Int, Double)]): Vector = {
    val size = nextIndex(seq)
    if (size == 0) Vectors.dense(Array.empty[Double])
    else Vectors.sparse(size, seq).compressed
  }


}


/**
 * Param that decides whether or not the values that were missing are tracked
 */
trait TrackNullsParam extends Params {
  final val trackNulls = new BooleanParam(
    parent = this, name = "trackNulls", doc = "option to keep track of values that were missing"
  )
  setDefault(trackNulls, TransmogrifierDefaults.TrackNulls)

  /**
   * Option to keep track of values that were missing
   */
  def setTrackNulls(v: Boolean): this.type = set(trackNulls, v)
}

/**
 * Param that decides whether or not the values that are considered invalid are tracked
 */
trait TrackInvalidParam extends Params {
  final val trackInvalid = new BooleanParam(
    parent = this, name = "trackInvalid", doc = "option to keep track of invalid values"
  )
  setDefault(trackInvalid, TransmogrifierDefaults.TrackInvalid)

  /**
   * Option to keep track of invalid values
   */
  def setTrackInvalid(v: Boolean): this.type = set(trackInvalid, v)
}

/**
 * Param that decides whether or not lengths of text are tracked during vectorization
 */
trait TrackTextLenParam extends Params {
  final val trackTextLen = new BooleanParam(
    parent = this, name = "trackTextLen", doc = "option to keep track of text lengths"
  )
  setDefault(trackTextLen, TransmogrifierDefaults.TrackTextLen)

  /**
   * Option to keep track of text lengths
   */
  def setTrackTextLen(v: Boolean): this.type = set(trackTextLen, v)
}

trait CleanTextFun {
  def cleanTextFn(s: String, shouldClean: Boolean): String = if (shouldClean) TextUtils.cleanString(s) else s
}

trait CleanTextMapFun extends CleanTextFun {

  def cleanMap[V](m: Map[String, V], shouldCleanKey: Boolean, shouldCleanValue: Boolean): Map[String, V] = {
    if (!shouldCleanKey && !shouldCleanValue) m
    else {
      m.map {
        case (k: String, v: String) => cleanTextFn(k, shouldCleanKey) -> cleanTextFn(v, shouldCleanValue)
        case (k: String, v: Traversable[_]) =>
          if (v.headOption.exists(_.isInstanceOf[String])) {
            cleanTextFn(k, shouldCleanKey) -> v.asInstanceOf[Traversable[String]].map(cleanTextFn(_, shouldCleanValue))
          } else {
            cleanTextFn(k, shouldCleanKey) -> v
          }
        case (k: String, v) => cleanTextFn(k, shouldCleanKey) -> v
      }.asInstanceOf[Map[String, V]]
    }
  }

}

trait TextParams extends Params {
  final val cleanText = new BooleanParam(
    parent = this, name = "cleanText", doc = "ignore capitalization and punctuation in grouping categories"
  )
  setDefault(cleanText, TransmogrifierDefaults.CleanText)
  def setCleanText(clean: Boolean): this.type = set(cleanText, clean)
}


trait PivotParams extends TextParams {
  final val topK = new IntParam(
    parent = this, name = "topK", doc = "number of elements to keep for each vector",
    isValid = ParamValidators.gt(0L)
  )
  setDefault(topK, TransmogrifierDefaults.TopK)
  def setTopK(numberToKeep: Int): this.type = set(topK, numberToKeep)
}

trait MinSupportParam extends Params {
  final val minSupport = new IntParam(
    parent = this, name = "minSupport", doc = "minimum number of occurrences an element must have to appear in pivot",
    isValid = ParamValidators.gtEq(0L)
  )
  setDefault(minSupport, TransmogrifierDefaults.MinSupport)
  def setMinSupport(min: Int): this.type = set(minSupport, min)
}


trait SaveOthersParams extends Params {
  final val unseenName: Param[String] = new Param(this, "unseenName",
    "Name to give indexes which do not have a label name associated with them."
  )
  setDefault(unseenName, TransmogrifierDefaults.OtherString)
  def getUnseenName: String = $(unseenName)
  def setUnseenName(unseenNameIn: String): this.type = set(unseenName, unseenNameIn)
}


trait MapPivotParams extends Params {
  self: CleanTextMapFun =>

  final val cleanKeys = new BooleanParam(
    parent = this, name = "cleanKeys", doc = "ignore capitalization and punctuation in grouping map keys"
  )
  setDefault(cleanKeys, TransmogrifierDefaults.CleanKeys)

  def setCleanKeys(clean: Boolean): this.type = set(cleanKeys, clean)

  final val whiteListKeys = new StringArrayParam(
    parent = this, name = "whiteListKeys", doc = "list of map keys to include in pivot"
  )
  setDefault(whiteListKeys, Array[String]())

  final def setWhiteListKeys(keys: Array[String]): this.type = set(whiteListKeys, keys)

  final val blackListKeys = new StringArrayParam(
    parent = this, name = "blackListKeys", doc = "list of map keys to exclude from pivot"
  )
  setDefault(blackListKeys, Array[String]())

  final def setBlackListKeys(keys: Array[String]): this.type = set(blackListKeys, keys)

  protected def filterKeys[V](m: Map[String, V], shouldCleanKey: Boolean, shouldCleanValue: Boolean): Map[String, V] = {
    val map = cleanMap[V](m, shouldCleanKey, shouldCleanValue)
    val (whiteList, blackList) = (
      $(whiteListKeys).map(cleanTextFn(_, shouldCleanKey)),
      $(blackListKeys).map(cleanTextFn(_, shouldCleanKey))
    )
    if (whiteList.nonEmpty) {
      map.filter { case (k, v) => whiteList.contains(k) && !blackList.contains(k) }
    } else if (blackList.nonEmpty) {
      map.filter { case (k, v) => !blackList.contains(k) }
    } else {
      map
    }
  }


}


trait MapStringPivotHelper extends SaveOthersParams {
  self: MapPivotParams =>

  type MapMap = SequenceAggregators.MapMap
  type SeqSeqTupArr = Seq[Seq[(String, Array[String])]]
  type SeqMapMap = SequenceAggregators.SeqMapMap

  protected def getCategoryMaps[V]
  (
    in: Dataset[Seq[Map[String, V]]],
    convertToMapOfMaps: Map[String, V] => MapMap,
    shouldCleanKeys: Boolean,
    shouldCleanValues: Boolean
  ): Dataset[SeqMapMap] = {
    implicit val seqMapMapEncoder = Encoders.kryo[SeqMapMap]
    in.map(seq =>
      seq.map { kc =>
        val filteredMap = filterKeys[V](kc, shouldCleanKey = shouldCleanKeys, shouldCleanValue = shouldCleanValues)
        convertToMapOfMaps(filteredMap)
      }
    )
  }

  protected def getTopValues(categoryMaps: Dataset[SeqMapMap], inputSize: Int, topK: Int, minSup: Int): SeqSeqTupArr = {
    val sumAggr = SequenceAggregators.SumSeqMapMap(size = inputSize)
    val countOccurrences: SeqMapMap = categoryMaps.select(sumAggr.toColumn).first()
    // Top K values for each categorical input
    countOccurrences.map {
      _.map { case (k, v) => k ->
        v.toArray
          .filter(_._2 >= minSup)
          .sortBy(v => -v._2 -> v._1)
          .take(topK)
          .map(_._1)
      }.toSeq
    }
  }

  protected def makeVectorColumnMetadata
  (
    topValues: SeqSeqTupArr,
    inputFeatures: Array[TransientFeature],
    unseenName: String,
    trackNulls: Boolean = false
  ): Array[OpVectorColumnMetadata] = {
    for {
      (f, kvPairs) <- inputFeatures.zip(topValues)
      (key, values) <- kvPairs
      value <- values.view ++ Seq(unseenName) ++ // view here to avoid copying the array when appending the string
        (if (trackNulls) Seq(TransmogrifierDefaults.NullString) else Nil)
    } yield OpVectorColumnMetadata(
      parentFeatureName = Seq(f.name),
      parentFeatureType = Seq(f.typeName),
      grouping = Option(key),
      indicatorValue = Option(value)
    )
  }

  protected def makeOutputVectorMetadata
  (
    topValues: SeqSeqTupArr,
    inputFeatures: Array[TransientFeature],
    operationName: String,
    outputName: String,
    stageName: String,
    trackNulls: Boolean = false // todo remove default and use this for other maps
  ): OpVectorMetadata = {
    val otherValueString = $(unseenName)
    val cols = makeVectorColumnMetadata(topValues, inputFeatures, otherValueString, trackNulls)
    OpVectorMetadata(outputName, cols, Transmogrifier.inputFeaturesToHistory(inputFeatures, stageName))
  }
}
