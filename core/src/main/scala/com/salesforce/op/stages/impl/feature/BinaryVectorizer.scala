/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.BooleanParam

import scala.collection.mutable.ArrayBuffer

/**
 * Vectorizes Binary inputs where each input is transformed into 2 vector elements
 * where the first element is [1 -> true] or [0 -> false] and the second element
 * is [1 -> filled value] or [0 -> original value]. The vector representation for
 * each input is concatenated into a final vector representation.
 *
 * Example:
 *
 * Data: Seq[(Binary, Binary)] = ((Some(false), None)) => f1, f2
 * new BinaryVectorizer().setInput(f1, f2).setFillValue(10)
 *
 * will produce
 * Array(0.0, 0.0, 10.0, 1.0)
 *
 * @param uid uid for instance
 */
class BinaryVectorizer
(
  uid: String = UID[BinaryVectorizer]
) extends SequenceTransformer[Binary, OPVector](operationName = "vecBin", uid = uid)
  with VectorizerDefaults with TrackNullsParam {

  final val fillValue = new BooleanParam(
    parent = this, name = "fillValue", doc = "value to replace nulls"
  )
  setDefault(fillValue, false)

  def setFillValue(v: Boolean): this.type = set(fillValue, v)

  override def transformFn: (Seq[Binary]) => OPVector = in => {
    val (isTrackNulls, theFillValue) = ($(trackNulls), $(fillValue))
    val arr = ArrayBuffer.empty[Double]

    if (isTrackNulls) {
      in.foreach(_.value match {
        case None => arr.append(theFillValue, 1.0)
        case Some(v) => arr.append(v, 0.0)
      })
    } else in.foreach(b => arr.append(b.value.getOrElse[Boolean](theFillValue)))

    Vectors.dense(arr.toArray).toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    if ($(trackNulls)) {
      setMetadata(vectorMetadataWithNullIndicators.toMetadata)
    }
  }
}

