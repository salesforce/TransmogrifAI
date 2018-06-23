/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Creates null indicator columns for a sequence of input TextMap features, originally for use as a separate stage in
 * null tracking for hashed text features (easier to do outside the hashing vectorizer since we can make a null
 * indicator column for each input feature without having to add lots of complex logic in the hashing vectorizer to
 * deal with metadata for shared vs. separate hash spaces.
 */
class TextMapNullEstimator[T <: OPMap[String]]
(
  uid: String = UID[TextMapNullEstimator[_]]
)(implicit tti: TypeTag[T]) extends SequenceEstimator[T, OPVector](
  operationName = "textMapNull", uid = uid) with VectorizerDefaults with MapVectorizerFuns[String, T] {

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
      indicatorGroup = Option(key),
      indicatorValue = Option(OpVectorColumnMetadata.NullString)
    )

    setMetadata(OpVectorMetadata(vectorOutputName, colMeta,
      Transmogrifier.inputFeaturesToHistory(transFeat, stageName))
      .toMetadata)

    new TextMapNullModel[T](allKeys, shouldCleanKeys, shouldCleanValues,
      operationName = operationName, uid = uid)
  }
}

final class TextMapNullModel[T <: OPMap[String]] private[op]
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
        keys.map(k => if (cleaned.contains(k) && tokenMap(k).nonEmpty) 0.0 else 1.0)
    }.toOPVector
  }

}
