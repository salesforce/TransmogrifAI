package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.features.types.{OPVector, TextList}
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.Vectors

import scala.reflect.runtime.universe.TypeTag

/**
 * Transformer for generating using text length per row
 */
class TextLenTransformer[T <: TextList]
(
  uid: String = UID[TextLenTransformer[_]]
)(implicit tti: TypeTag[T], val ttiv: TypeTag[T#Value]) extends SequenceTransformer[T, OPVector](
  operationName = "textLen", uid = uid) with VectorizerDefaults {
  override def transformFn: Seq[T] => OPVector = in => {
    val output = in.flatMap { f =>
      if (f.isEmpty) Seq(0.0) else f.value.map(_.length.toDouble)
    }
    Vectors.dense(output.toArray).toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val tf = getTransientFeatures()
    val colMeta = tf.map(_.toColumnTextLenData)

    setMetadata(
      OpVectorMetadata(vectorOutputName, colMeta, Transmogrifier.inputFeaturesToHistory(tf, stageName)).toMetadata
    )
  }
}
