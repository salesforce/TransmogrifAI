/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPMap, _}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata, SequenceAggregators}
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.sql.types.MetadataBuilder
import org.apache.spark.sql.{Dataset, Encoders}
import org.joda.time.{DateTime, DateTimeZone, Days}

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for vectorizing OPMap[A] features. Individual vectorizers for different feature types need to implement
 * the getFillByKey function (which calculates any fill values that differ by key - means, modes, etc.) and the
 * makeModel function (which specifies which type of model will be returned).
 *
 * @param uid           uid for instance
 * @param operationName unique name of the operation this stage performs
 * @param convertFn     maps input type into a Map[String, Double] on the way to conversion to OPVector
 * @param tti           type tag for input
 * @param ttiv          type tag for input value
 * @tparam A value type for underlying map
 * @tparam T input feature type to vectorize into an OPVector
 */
abstract class OPMapVectorizer[A, T <: OPMap[A]]
(
  uid: String = UID[OPMapVectorizer[A, T]],
  operationName: String,
  val convertFn: T#Value => RealMap#Value
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends SequenceEstimator[T, OPVector](operationName = operationName, uid = uid) with
    MapVectorizerFuns[Double, RealMap] with NumericMapDefaultParam with TrackNullsParam {

  private implicit val seqRealMapEncoder = Encoders.kryo[Seq[RealMap#Value]]
  protected val shouldCleanValues = true

  def fillByKey(dataset: Dataset[Seq[T#Value]]): Seq[Map[String, Double]] = Seq.empty

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[A, T]

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val allKeys = getKeyValues(
      in = dataset.map(v => v.map(convertFn)),
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues
    )

    val meta = if ($(trackNulls)) makeVectorMetaWithNullIndicators(allKeys) else makeVectorMetadata(allKeys)
    setMetadata(meta.toMetadata)


    val args = OPMapVectorizerModelArgs(
      allKeys = allKeys,
      fillByKey = fillByKey(dataset),
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      defaultValue = $(defaultValue),
      trackNulls = $(trackNulls)
    )
    makeModel(args, operationName, uid)
  }
}

/**
 * Class for vectorizing BinaryMap features. Fills missing keys with args.defaultValue, which does not depend on the
 * key, so getFillByKey returns an empty sequence.
 *
 * @param uid uid for instance
 * @param tti type tag for input
 * @tparam T input feature type to vectorize into an OPVector
 */
class BinaryMapVectorizer[T <: OPMap[Boolean]](uid: String = UID[BinaryMapVectorizer[T]])(implicit tti: TypeTag[T])
  extends OPMapVectorizer[Boolean, T](uid = uid, operationName = "vecBinMap", convertFn = booleanToRealMap) {
  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[Boolean, T] =
    new BinaryMapVectorizerModel(args, operationName = operationName, uid = uid)
}

/**
 * Class for vectorizing IntegralMap features. Fills missing keys with the mode for that key.
 *
 * @param uid uid for instance
 * @param tti type tag for input
 * @tparam T input feature type to vectorize into an OPVector
 */
class IntegralMapVectorizer[T <: OPMap[Long]](uid: String = UID[IntegralMapVectorizer[T]])(implicit tti: TypeTag[T])
  extends OPMapVectorizer[Long, T](uid = uid, operationName = "vecIntMap", convertFn = intMapToRealMap) {

  def setFillWithMode(shouldFill: Boolean): this.type = set(withConstant, !shouldFill)

  override def fillByKey(dataset: Dataset[Seq[T#Value]]): Seq[Map[String, Double]] = {
    val size = getInputFeatures().length
    val cleanedData = dataset.map(_.map(
      cleanMap(_, shouldCleanKey = $(cleanKeys), shouldCleanValue = shouldCleanValues)
    ))

    if ($(withConstant)) Seq.empty
    else {
      val modeAggr = SequenceAggregators.ModeSeqMapLong(size = size)
      cleanedData.select(modeAggr.toColumn).first()
    }.map(convertFn)
  }

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[Long, T] =
    new IntegralMapVectorizerModel(args, operationName = operationName, uid = uid)
}

/**
 * Class for vectorizing DateMap features. Fills missing keys with args.defaultValue, which does not depend on the
 * key, so getFillByKey returns an empty sequence.
 *
 * @param uid uid for instance
 * @param tti type tag for input
 * @tparam T input feature type to vectorize into an OPVector
 */
class DateMapVectorizer[T <: OPMap[Long]](uid: String = UID[DateMapVectorizer[T]])(implicit tti: TypeTag[T])
  extends OPMapVectorizer[Long, T](uid = uid, operationName = "vecDateMap", convertFn = intMapToRealMap) {

  val referenceDate = new Param[DateTime](parent = this, name = "referenceDate",
    doc = "Reference date used to compare time to")
  def setReferenceDate(date: DateTime): this.type = set(referenceDate, date)
  def getReferenceDate(): DateTime = $(referenceDate)

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[Long, T] =
    new DateMapVectorizerModel(
      args.copy(referenceDate = Some(getReferenceDate())),
      operationName = operationName,
      uid = uid
    )

}

class TextMapHashingVectorizer[T <: OPMap[String]]
(
  uid: String = UID[TextMapHashingVectorizer[T]]
)(implicit tti: TypeTag[T])
  extends OPMapVectorizer[String, T](
    uid = uid, operationName = "vecHashTextMap",
    convertFn = _.map { case (k, v) => k -> 1.0 }
  ) with TextParams {

  final val prependFeatureName = new BooleanParam(
    parent = this, name = "prependFeatureName",
    doc = s"if true, prepends a input feature name to each token of that feature"
  )
  setDefault(prependFeatureName, true)

  def setPrependFeatureName(v: Boolean): this.type = set(prependFeatureName, v)

  def getFillByKey(dataset: Dataset[Seq[T#Value]]): Seq[Map[String, Double]] = Seq.empty

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[String, T] =
    new TextMapHashingVectorizerModel[T](
      args = args.copy(shouldCleanValues = $(cleanText)),
      shouldPrependFeatureName = $(prependFeatureName),
      operationName = operationName,
      uid = uid
    )
}


/**
 * Class for vectorizing RealMap features. Fills missing keys with the mean for that key.
 *
 * @param uid uid for instance
 * @param tti type tag for input
 * @tparam T input feature type to vectorize into an OPVector
 */
class RealMapVectorizer[T <: OPMap[Double]](uid: String = UID[RealMapVectorizer[T]])(implicit tti: TypeTag[T])
  extends OPMapVectorizer[Double, T](uid = uid, operationName = "vecRealMap", convertFn = identity) {

  def setFillWithMean(shouldFill: Boolean): this.type = set(withConstant, !shouldFill)

  override def fillByKey(dataset: Dataset[Seq[T#Value]]): Seq[Map[String, Double]] = {
    val size = getInputFeatures().length
    val cleanedData = dataset.map(_.map(
      cleanMap(_, shouldCleanKey = $(cleanKeys), shouldCleanValue = shouldCleanValues)
    ))

    if ($(withConstant)) Seq.empty
    else {
      val meanAggr = SequenceAggregators.MeanSeqMapDouble(size = size)
      cleanedData.select(meanAggr.toColumn).first()
    }
  }

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[Double, T] =
    new RealMapVectorizerModel(args, operationName = operationName, uid = uid)
}

sealed trait NumericMapDefaultParam extends Params {

  final val defaultValue = new DoubleParam(
    parent = this, name = "defaultValue", doc = "value to give missing keys when pivoting"
  )
  setDefault(defaultValue, 0.0)

  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  final val withConstant = new BooleanParam(
    parent = this, name = "fillWithConstant", doc = "boolean to enable filling missing key values with a constant"
  )
  setDefault(withConstant, true)

  def setFillWithConstant(value: Double): this.type = {
    set(defaultValue, value)
    set(withConstant, true)
  }
}


trait MapVectorizerFuns[A, T <: OPMap[A]] extends VectorizerDefaults with MapPivotParams with CleanTextMapFun {
  self: PipelineStage =>

  protected def getKeyValues
  (
    in: Dataset[Seq[T#Value]],
    shouldCleanKeys: Boolean,
    shouldCleanValues: Boolean
  ): Seq[Seq[String]] = {
    val size = getInputFeatures().length
    val sumAggr = SequenceAggregators.SumSeqSet(size = size)
    implicit val enc = sumAggr.outputEncoder
    in.map(_.map(kb => filterKeys(kb, shouldCleanKey = shouldCleanKeys, shouldCleanValue = shouldCleanValues).keySet))
      .select(sumAggr.toColumn)
      .first()
      .map(_.toSeq)
  }

  protected def makeVectorMetadata(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val meta = vectorMetadataFromInputFeatures
    val cols = for {
      (keys, col) <- allKeys.zip(meta.columns)
      key <- keys
    } yield new OpVectorColumnMetadata(
      parentFeatureName = col.parentFeatureName,
      parentFeatureType = col.parentFeatureType,
      indicatorGroup = Option(key),
      indicatorValue = None
    )
    meta.withColumns(cols.toArray)
  }


  protected def makeVectorMetaWithNullIndicators(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val vectorMeta = makeVectorMetadata(allKeys)
    val updatedCols = vectorMeta.columns.flatMap { col =>
      Seq(
        col,
        OpVectorColumnMetadata(
          parentFeatureName = col.parentFeatureName,
          parentFeatureType = col.parentFeatureType,
          indicatorGroup = col.indicatorGroup,
          indicatorValue = Some(TransmogrifierDefaults.NullString)
        )
      )
    }
    vectorMeta.withColumns(updatedCols)
  }
}

/**
 * OPMap vectorizer model arguments
 *
 * @param allKeys           all keys per feature
 * @param fillByKey         fill values for features
 * @param shouldCleanKeys   should clean map keys
 * @param shouldCleanValues should clean map values
 * @param defaultValue      default value to replace with
 * @param trackNulls        add column to track null values for each map key
 * @param referenceDate     reference date to subtract off before converting times into to vector
 */
sealed case class OPMapVectorizerModelArgs
(
  allKeys: Seq[Seq[String]],
  fillByKey: Seq[Map[String, Double]],
  shouldCleanKeys: Boolean,
  shouldCleanValues: Boolean,
  defaultValue: Double,
  trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
  referenceDate: Option[DateTime] = None
)

sealed abstract class OPMapVectorizerModel[A, I <: OPMap[A]]
(
  val args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I])
  extends SequenceModel[I, OPVector](operationName = operationName, uid = uid) with CleanTextMapFun {

  protected def convertFn: I#Value => RealMap#Value

  def transformFn: Seq[I] => OPVector = row => {
    val converted = row.map(v => convertFn(v.value))
    val eachPivoted: Array[Double] =
      converted.zipWithIndex.flatMap {
        case (map, i) =>
          val keys = args.allKeys(i)
          val fills: Map[String, Double] = if (args.fillByKey.isEmpty) Map.empty else args.fillByKey(i)
          val cleaned = cleanMap(map, shouldCleanKey = args.shouldCleanKeys, shouldCleanValue = args.shouldCleanValues)
          if (args.trackNulls) {
            keys.flatMap(k => cleaned.get(k) match {
              case Some(v) => Seq(v, 0.0)
              case None => Seq(fills.getOrElse(k, args.defaultValue), 1.0)
            })
          } else {
            keys.map(k => cleaned.getOrElse(k, fills.getOrElse(k, args.defaultValue)))
          }
      }.toArray
    Vectors.dense(eachPivoted).compressed.toOPVector
  }

}

final class BinaryMapVectorizerModel[T <: OPMap[Boolean]] private[op]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Boolean, T](args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Boolean] => Map[String, Double] = booleanToRealMap
}

final class IntegralMapVectorizerModel[T <: OPMap[Long]] private[op]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Long, T](args = args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Long] => Map[String, Double] = intMapToRealMap
}

final class DateMapVectorizerModel[T <: OPMap[Long]] private[op]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Long, T](args = args, operationName = operationName, uid = uid) {
  val timeZone: DateTimeZone = DateTimeUtils.DefaultTimeZone
  val reference = args.referenceDate.getOrElse(TransmogrifierDefaults.ReferenceDate)

  def convertFn: DateMap#Value => RealMap#Value = (dt: DateMap#Value) =>
    dt.mapValues(v => Days.daysBetween(new DateTime(v, timeZone), reference).getDays.toDouble)
}

final class RealMapVectorizerModel[T <: OPMap[Double]] private[op]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Double, T](args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Double] => Map[String, Double] = identity
}

object TextMapHashingVectorizerNames {
  val MapKeys = "mapKeys"
}

final class TextMapHashingVectorizerModel[T <: OPMap[String]] private[op]
(
  args: OPMapVectorizerModelArgs,
  val shouldPrependFeatureName: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[String, T](args, operationName = operationName, uid = uid)
    with TextTokenizerParams with HashingFun {

  private val hashingParams = HashingFunctionParams(
    hashWithIndex = TransmogrifierDefaults.HashWithIndex,
    // Need to not prepend the feature name in the tests, so allow this to be settable
    prependFeatureName = shouldPrependFeatureName,
    numFeatures = TransmogrifierDefaults.DefaultNumOfFeatures,
    numInputs = 1, // All tokens are combined into a single TextList before hashing
    maxNumOfFeatures = TransmogrifierDefaults.MaxNumOfFeatures,
    forceSharedHashSpace = TransmogrifierDefaults.ForceSharedHashSpace,
    binaryFreq = TransmogrifierDefaults.BinaryFreq,
    hashAlgorithm = TransmogrifierDefaults.HashAlgorithm
  )

  def convertFn: Map[String, String] => Map[String, Double] = _.map { case (k, v) => k -> 1.0 }

  // Try to use old vectorizer on each of the items, mapped to Text features
  override def transformFn: Seq[T] => OPVector = row => {
    val tokenSeq = row.zipWithIndex.flatMap {
      case (map, i) =>
        val keys = args.allKeys(i)
        val cleaned = cleanMap(map.v, shouldCleanKey = args.shouldCleanKeys, shouldCleanValue = args.shouldCleanValues)
        val mapValues = cleaned.map { case (k, v) => v.toText }
        mapValues.map(tokenizeFn(_)).toSeq
    }
    val allTokens = tokenSeq.flatMap(_.value).toTextList

    // TODO make sure this also works when we are not using the shared hash space
    hash(hashingParams, Seq(allTokens))
  }

  // TODO: Set this metadata in the estimator
  override def onGetMetadata(): Unit = {
    val metaBuilder = new MetadataBuilder()
    // Add all the maps' keys to the metadata
    metaBuilder.withMetadata(makeVectorMetadata(hashingParams, outputName).toMetadata)
    metaBuilder.putStringArray(TextMapHashingVectorizerNames.MapKeys, args.allKeys.flatten.toArray)

    setMetadata(metaBuilder.build())
  }

}
