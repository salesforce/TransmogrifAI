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
import com.salesforce.op.features.types.{OPVector, TextList}
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors

import scala.reflect.runtime.universe.TypeTag

/**
 * Sequence transformer for generating a sequence of text lengths from a sequence of TextList values (eg. tokenized
 * raw text)
 */
class TextLenTransformer[T <: TextList]
(
  uid: String = UID[TextLenTransformer[_]]
)(implicit tti: TypeTag[T], val ttiv: TypeTag[T#Value]) extends SequenceTransformer[T, OPVector](
  operationName = "textLen", uid = uid) with VectorizerDefaults with TextTokenizerParams
  with TextParams {

  override def transformFn: Seq[T] => OPVector = in => {
    val output = in.map(_.value.map(_.length).sum.toDouble)
    Vectors.dense(output.toArray).toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val tf = getTransientFeatures()
    val colMeta = tf.map(f => new OpVectorColumnMetadata(
        parentFeatureName = Seq(f.name),
        parentFeatureType = Seq(f.typeName),
        grouping = Some(f.name),
        descriptorValue = Option(OpVectorColumnMetadata.TextLenString)
      )
    )

    setMetadata(
      OpVectorMetadata(vectorOutputName, colMeta, Transmogrifier.inputFeaturesToHistory(tf, stageName)).toMetadata
    )
  }
}
