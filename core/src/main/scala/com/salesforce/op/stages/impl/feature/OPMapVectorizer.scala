/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata, SequenceAggregators}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.sql.{Dataset, Encoders}

import scala.reflect.runtime.universe.TypeTag


abstract class OPMapVectorizer[A, I <: OPMap[A]]
(
  operationName: String = "vecMap",
  uid: String = UID[OPMapVectorizer[_, _]]
)(implicit tti: TypeTag[I],
  ttiv: TypeTag[I#Value],
  val convert: I#Value => RealMap#Value
) extends SequenceEstimator[I, OPVector](operationName = operationName, uid = uid)
  with VectorizerDefaults with MapPivotParams with CleanTextMapFun {
  private implicit val seqSetEncoder = Encoders.kryo[Seq[Set[String]]]
  private implicit val seqRealMapEncoder = Encoders.kryo[Seq[RealMap#Value]]

  final val defaultValue = new DoubleParam(
    parent = this, name = "defaultValue", doc = "value to give missing keys when pivoting"
  )
  setDefault(defaultValue, 0.0)
  def setDefaultValue(value: Double): this.type = set(defaultValue, value)

  private def getKeyValues
  (
    in: Dataset[Seq[RealMap#Value]],
    shouldCleanKeys: Boolean,
    shouldCleanValues: Boolean
  ): Seq[Seq[String]] = {
    val inputSize = getInputFeatures().length
    in.map(_.map(kb => filterKeys(kb, shouldCleanKeys, shouldCleanValues).keySet))
      .select(SequenceAggregators.SumSeqSet(size = inputSize))
      .first()
      .map(_.toSeq)
  }

  private def makeVectorMeta(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val meta = vectorMetadataFromInputFeatures
    val cols = for {
      (keys, col) <- allKeys.zip(meta.columns)
      key <- keys
    } yield new OpVectorColumnMetadata(
      parentFeatureName = col.parentFeatureName,
      parentFeatureType = col.parentFeatureType,
      indicatorGroup = None,
      indicatorValue = Option(key)
    )
    meta.withColumns(cols.toArray)
  }

  def fitFn(dataset: Dataset[Seq[I#Value]]): SequenceModel[I, OPVector] = {
    val shouldCleanKeys = $(cleanKeys)
    val shouldCleanValues = true

    val allKeys = getKeyValues(dataset.map(v => v.map(convert)), shouldCleanKeys, shouldCleanValues)
    setMetadata(makeVectorMeta(allKeys).toMetadata)

    val args = OPMapVectorizerModelArgs(
      allKeys = allKeys,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues,
      defaultValue = $(defaultValue)
    )
    model(args, operationName = operationName, uid = uid)
  }

  protected def model(args: OPMapVectorizerModelArgs, operationName: String, uid: String): SequenceModel[I, OPVector]

}


class BinaryMapVectorizer(uid: String = UID[BinaryMapVectorizer])
  extends OPMapVectorizer[Boolean, BinaryMap](operationName = "vecBinMap", uid = uid) {
  def model(args: OPMapVectorizerModelArgs, operationName: String, uid: String): SequenceModel[BinaryMap, OPVector] =
    new BinaryMapVectorizerModel(args, operationName, uid)
}

class IntegralMapVectorizer[T <: OPMap[Long]]
(
  uid: String = UID[IntegralMapVectorizer[T]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OPMapVectorizer[Long, T](operationName = "vecIntMap", uid = uid) {
  def model(args: OPMapVectorizerModelArgs, operationName: String, uid: String): SequenceModel[T, OPVector] =
    new IntegralMapVectorizerModel(args, operationName, uid)
}

class RealMapVectorizer[T <: OPMap[Double]]
(
  uid: String = UID[RealMapVectorizer[T]]
)(implicit tti: TypeTag[T], ttiv: TypeTag[T#Value])
  extends OPMapVectorizer[Double, T](operationName = "vecRealMap", uid = uid) {
  def model(args: OPMapVectorizerModelArgs, operationName: String, uid: String): SequenceModel[T, OPVector] =
    new RealMapVectorizerModel(args, operationName, uid)
}

case class OPMapVectorizerModelArgs
(
  allKeys: Seq[Seq[String]],
  shouldCleanKeys: Boolean,
  shouldCleanValues: Boolean,
  defaultValue: Double
)

// TODO: can we make this base model and concrete model classes prettier?
sealed abstract class OPMapVectorizerModel[A, I <: OPMap[A]]
(
  val args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String,
  val convert: I#Value => RealMap#Value
)(implicit tti: TypeTag[I])
  extends SequenceModel[I, OPVector](operationName = operationName, uid = uid) with CleanTextMapFun {

  def transformFn: Seq[I] => OPVector = row => {
    val converted = row.map(v => convert(v.value))
    val eachPivoted: Array[Double] =
      converted.zip(args.allKeys).flatMap { case (map, keys) =>
        val cleanedMap = cleanMap(map, args.shouldCleanKeys, args.shouldCleanValues)
        keys.map(k => cleanedMap.getOrElse(k, args.defaultValue))
      }.toArray
    Vectors.dense(eachPivoted).compressed.toOPVector
  }

}

private final class BinaryMapVectorizerModel
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
) extends OPMapVectorizerModel[Boolean, BinaryMap](args, operationName = operationName, uid = uid, booleanToRealMap)

private final class IntegralMapVectorizerModel[T <: OPMap[Long]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Long, T](args = args, operationName = operationName, uid = uid, intMapToRealMap)

private final class RealMapVectorizerModel[T <: OPMap[Double]]
(
  args: OPMapVectorizerModelArgs,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends OPMapVectorizerModel[Double, T](args = args, operationName = operationName, uid = uid, identity)





