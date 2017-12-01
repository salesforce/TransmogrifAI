/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ArrayBuffer


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
    val attributes = inN.map(f => OpVectorMetadata(schema(f.name)))
    val outMeta = OpVectorMetadata.flatten(outputName, attributes)
    setMetadata(outMeta.toMetadata)
  }

}

private final class VectorsCombinerModel(operationName: String, uid: String)
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

