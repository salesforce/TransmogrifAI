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
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.util.ClosureUtils

import scala.util.Failure

/**
 * Allows columns to be dropped from a feature vector based on properties of the
 * metadata about what is contained in each column (will work only on vectors)
 * created with [[OpVectorMetadata]]
 * @param matchFn function that goes from [[OpVectorColumnMetadata]] to boolean for dropping
 *                columns (cases that evaluate to true will be dropped)
 * @param uid     uid for instance
 */
class DropIndicesByTransformer
(
  val matchFn: OpVectorColumnMetadata => Boolean,
  uid: String = UID[DropIndicesByTransformer]
) extends UnaryTransformer[OPVector, OPVector](operationName = "dropIndicesBy", uid = uid) {

  ClosureUtils.checkSerializable(matchFn) match {
    case Failure(e) => throw new IllegalArgumentException("Provided function is not serializable", e)
    case _ =>
  }

  @transient private lazy val vectorMetadata = OpVectorMetadata(getInputSchema()(in1.name))
  @transient private lazy val columnMetadataToKeep = vectorMetadata.columns.collect { case cm if !matchFn(cm) => cm }
  @transient private lazy val indicesToKeep = columnMetadataToKeep.map(_.index)

  override def transformFn: OPVector => OPVector = v => {
    val values = new Array[Double](indicesToKeep.length)
    v.value.foreachActive((i, v) => {
      val k = indicesToKeep.indexOf(i)
      if (k >= 0) values(k) = v
    })
    Vectors.dense(values).compressed.toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val newMetaData = OpVectorMetadata(getOutputFeatureName, columnMetadataToKeep, vectorMetadata.history).toMetadata
    setMetadata(newMetaData)
  }
}
