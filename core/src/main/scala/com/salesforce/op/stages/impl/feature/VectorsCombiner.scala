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
import com.salesforce.op.features.TransientFeature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer
import scala.util.Try


/**
 * Takes in a sequence of vectors and combines them into a single vector
 *
 * @param uid uid for instance
 */
class VectorsCombiner(uid: String = UID[VectorsCombiner])
  extends SequenceEstimator[OPVector, OPVector](operationName = "combVec", uid = uid) {

  def fitFn(dataset: Dataset[Seq[OPVector#Value]]): SequenceModel[OPVector, OPVector] = {
    updateMetadata(dataset)
    new VectorsCombinerModel(operationName = operationName, uid = uid)
  }

  /**
   * Function makes sure that the attribute names for each element of the vectors combined match the elements
   *
   * @param data input dataset of vectors
   */
  private def updateMetadata(data: Dataset[Seq[OPVector#Value]]): Unit = {
    val schema = getInputSchema()
    lazy val firstRow = data.first()

    def vectorSize(f: TransientFeature, index: Int): Int = Try {
      AttributeGroup.fromStructField(schema(f.name)).numAttributes.get // see it there is an attribute group size
    } getOrElse firstRow(index).size // get the size from the data

    val attributes = inN.zipWithIndex.map {
      case (f, i) => Try(OpVectorMetadata(schema(f.name))).getOrElse(f.toVectorMetaData(vectorSize(f, i)))
    }

    val outMeta = OpVectorMetadata.flatten(getOutputFeatureName, attributes)
    setMetadata(outMeta.toMetadata)
  }

}

final class VectorsCombinerModel private[op] (operationName: String, uid: String)
  extends SequenceModel[OPVector, OPVector](operationName = operationName, uid = uid) {
  def transformFn: Seq[OPVector] => OPVector = VectorsCombiner.combineOP
}

case object VectorsCombiner {

  /**
   * Combine multiple OP vectors into one
   *
   * @param vectors input vectors
   * @return result vector
   */
  def combineOP(vectors: Seq[OPVector]): OPVector = {
    new OPVector(combine(vectors.view.map(_.value)))
  }

  /**
   * Combine multiple vectors into one
   *
   * @param vectors input vectors
   * @return result vector
   */
  def combine(vectors: Seq[Vector]): Vector = {
    val indices = ArrayBuffer.empty[Int]
    val values = ArrayBuffer.empty[Double]

    val size = vectors.foldLeft(0)((size, vector) => {
      vector.foreachActive { case (i, v) =>
        if (v != 0.0) {
          indices += size + i
          values += v
        }
      }
      size + vector.size
    })
    Vectors.sparse(size, indices.toArray, values.toArray).compressed
  }

}

