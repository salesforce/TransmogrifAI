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

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldCleanValues = false
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
        // val tokenMap = cleaned.mapValues(tokenize(_).tokens)

        // Need to check if key is present, and also that our tokenizer will not remove the value
        keys.map { k =>
          cleaned.get(k).map(_.length).getOrElse(0).toDouble
        }
    }.toOPVector
  }

}
