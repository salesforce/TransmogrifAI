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

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
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
    } else {
      setMetadata(vectorMetadataFromInputFeatures.toMetadata)
    }

  }
}

