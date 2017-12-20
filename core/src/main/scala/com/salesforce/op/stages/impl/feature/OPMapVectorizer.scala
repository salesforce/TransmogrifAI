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
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Params}
import org.apache.spark.sql.{Dataset, Encoders}
import org.joda.time.{DateTime, DateTimeZone, Days}

import scala.reflect.runtime.universe.TypeTag

/**
 * Base class for vectorizing OPMap[A] features. Individual vectorizers for different feature types need to implement
 * the getFillByKey function (which calculates any fill values that differ by key - means, modes, etc.) and the
 * makeModel funtion (which specifies which type of model will be returned).
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
    MapVectorizerFuncs[Double, RealMap] with NumericMapDefaultParam {

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
    setMetadata(makeVectorMeta(allKeys).toMetadata)

    val args = OPMapVectorizerModelArgs(
      allKeys = allKeys,
      fillByKey = fillByKey(dataset),
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      defaultValue = $(defaultValue)
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

  def makeModel(args: OPMapVectorizerModelArgs, operationName: String, uid: String): OPMapVectorizerModel[Long, T] =
    new DateMapVectorizerModel(args, operationName = operationName, uid = uid)

}

/**
 * Converts a sequence of KeyString features into a vector keeping the top K most common occurrences of each
 * key in the maps for that feature (ie the final vector has length k * number of keys * number of features).
 * Each key found will also generate an other column which will capture values that do not make the cut or where not
 * seen in training. Note that any keys not seen in training will be ignored.
 *
 * @param uid uid for instance
 */
class TextMapPivotVectorizer[T <: OPMap[String]]
(
  uid: String = UID[TextMapPivotVectorizer[T]]
)(implicit tti: TypeTag[T])
  extends SequenceEstimator[T, OPVector](operationName = "vecTextMap", uid = uid)
    with VectorizerDefaults with PivotParams with MapPivotParams with TextParams
    with MapStringPivotHelper with CleanTextMapFun with MinSupportParam {

  def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = $(cleanText)

    def convertToMapOfMaps(mapIn: Map[String, String]): MapMap = {
      mapIn.map { case (k, v) => k -> Map(v -> 1L) }
    }

    val categoryMaps: Dataset[SeqMapMap] =
      getCategoryMaps(dataset, convertToMapOfMaps, shouldCleanKeys, shouldCleanValues)

    val topValues: SeqSeqTupArr = getTopValues(categoryMaps, inN.length, $(topK), $(minSupport))

    val vectorMeta = createOutputVectorMetadata(topValues, inN, operationName, outputName, stageName)
    setMetadata(vectorMeta.toMetadata)

    new TextMapPivotVectorizerModel[T](
      topValues = topValues,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      operationName = operationName,
      uid = uid
    )
  }
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


trait MapVectorizerFuncs[A, T <: OPMap[A]] extends VectorizerDefaults with MapPivotParams with CleanTextMapFun {
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

  protected def makeVectorMeta(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
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
}

/**
 * OPMap vectorizer model arguments
 *
 * @param allKeys           all keys per feature
 * @param fillByKey         fill values for features
 * @param shouldCleanKeys   should clean map keys
 * @param shouldCleanValues should clean map values
 * @param defaultValue      default value to replace with
 */
sealed case class OPMapVectorizerModelArgs
(
  allKeys: Seq[Seq[String]],
  fillByKey: Seq[Map[String, Double]],
  shouldCleanKeys: Boolean,
  shouldCleanValues: Boolean,
  defaultValue: Double
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
          keys.map(k => cleaned.getOrElse(k, fills.getOrElse(k, args.defaultValue)))
      }.toArray
    Vectors.dense(eachPivoted).compressed.toOPVector
  }

}

private final class BinaryMapVectorizerModel[T <: OPMap[Boolean]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Boolean, T](args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Boolean] => Map[String, Double] = booleanToRealMap
}

private final class IntegralMapVectorizerModel[T <: OPMap[Long]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Long, T](args = args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Long] => Map[String, Double] = intMapToRealMap
}

private final class DateMapVectorizerModel[T <: OPMap[Long]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Long, T](args = args, operationName = operationName, uid = uid) {
  val timeZone: DateTimeZone = DateTimeUtils.DefaultTimeZone
  val referenceDate: DateTime = new DateTime(TransmogrifierDefaults.ReferenceDate.getMillis, timeZone)

  def convertFn: DateMap#Value => RealMap#Value = (dt: DateMap#Value) =>
    dt.mapValues(v => Days.daysBetween(new DateTime(v, timeZone), referenceDate).getDays.toDouble)
}

private final class RealMapVectorizerModel[T <: OPMap[Double]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Double, T](args, operationName = operationName, uid = uid) {
  def convertFn: Map[String, Double] => Map[String, Double] = identity
}

private final class TextMapPivotVectorizerModel[T <: OPMap[String]]
(
  val topValues: Seq[Seq[(String, Array[String])]],
  val shouldCleanKeys: Boolean,
  val shouldCleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextMapFun {

  def transformFn: Seq[T] => OPVector = row => {
    // Combine top values for each feature with map feature
    val eachPivoted =
      row.zip(topValues).map { case (map, topMap) =>
        val cleanedMap = cleanMap(map.value, shouldCleanKeys, shouldCleanValues)
        topMap.map { case (mapKey, top) =>
          val sizeOfVector = top.length
          cleanedMap.get(mapKey) match {
            case None => Seq(sizeOfVector -> 0.0)
            case Some(cv) =>
              top.indexOf(cv) match {
                case i if i < 0 => Seq(sizeOfVector -> 1.0)
                case i => Seq(i -> 1.0, sizeOfVector -> 0.0)
              }
          }
        }
      }
    // Fix indices for sparse vector
    val reindexed = reindex(eachPivoted.map(reindex))
    makeSparseVector(reindexed).toOPVector
  }
}
