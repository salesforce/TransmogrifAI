/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe.TypeTag

/**
 * Creates null indicator columns for a sequence of input TextList features, originally for use as a separate stage in
 * null tracking for hashed text features (easier to do outside the hashing vectorizer since we can make a null
 * indicator column for each input feature without having to add lots of complex logic in the hashing vectorizer to
 * deal with metadata for shared vs. separate hash spaces.
 */
class TextListNullTransformer[T <: TextList]
(
  uid: String = UID[TextListNullTransformer[_]]
)(implicit tti: TypeTag[T], val ttiv: TypeTag[T#Value]) extends SequenceTransformer[T, OPVector](
  operationName = "textListNull", uid = uid) with VectorizerDefaults {

  override def transformFn: (Seq[T]) => OPVector = in => {
    val arr = ArrayBuffer.empty[Double]
    in.foreach(f => if (f.isEmpty) arr.append(1.0) else arr.append(0.0))
    Vectors.dense(arr.toArray).toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val tf = getTransientFeatures()
    val colMeta = tf.map(_.toColumnMetaData(isNull = true))

    setMetadata(
      OpVectorMetadata(vectorOutputName, colMeta, Transmogrifier.inputFeaturesToHistory(tf, stageName)).toMetadata
    )
  }
}
