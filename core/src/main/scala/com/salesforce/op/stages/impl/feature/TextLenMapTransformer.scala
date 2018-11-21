package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{OPVector, _}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Transformer for generating using text length per row
 */
class TextLenMapTransformer[T <: OPMap[String]]
(
  uid: String = UID[TextLenMapTransformer[_]]
)(implicit tti: TypeTag[T]) extends SequenceEstimator[T, OPVector](
  operationName = "textLenMap", uid = uid) with VectorizerDefaults with MapVectorizerFuns[String, T] {
  protected val shouldCleanValues = true

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {

    val shouldCleanKeys = $(cleanKeys)
    val allKeys: Seq[Seq[String]] = getKeyValues(
      in = dataset,
      shouldCleanKeys = shouldCleanKeys,
      shouldCleanValues = shouldCleanValues
    )

    // Make the metadata
    val transFeat = getTransientFeatures()
    val colMeta = for {
      (tf, keys) <- transFeat.zip(allKeys)
      key <- keys
    } yield OpVectorColumnMetadata(
      parentFeatureName = Seq(tf.name),
      parentFeatureType = Seq(tf.typeName),
      grouping = Option(key),
      descriptorValue = Option(OpVectorColumnMetadata.TextLen)
    )

    setMetadata(OpVectorMetadata(vectorOutputName, colMeta,
      Transmogrifier.inputFeaturesToHistory(transFeat, stageName))
      .toMetadata)

    new TextLenMapModel[T](allKeys, shouldCleanKeys, shouldCleanValues,
      operationName = operationName, uid = uid)
  }
}

final class TextLenMapModel[T <: OPMap[String]] private[op]
(
  val allKeys: Seq[Seq[String]],
  val cleanKeys: Boolean,
  val cleanValues: Boolean,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T])
  extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
    with VectorizerDefaults with CleanTextMapFun with TextTokenizerParams {

  def transformFn: Seq[T] => OPVector = row => {
    row.zipWithIndex.flatMap {
      case (map, i) =>
        val keys = allKeys(i)
        val cleaned = cleanMap(map.v, shouldCleanKey = cleanKeys, shouldCleanValue = cleanValues)
        val tokenMap = cleaned.mapValues { v => v.toText }.mapValues(tokenize(_).tokens)

        // Need to check if key is present, and also that our tokenizer will not remove the value
        keys.map { k =>
          if (cleaned.contains(k) && tokenMap(k).nonEmpty) {
            tokenMap(k).value.map(_.length).sum.toDouble
          } else {
            0.0
          }
        }
    }.toOPVector
  }

}
