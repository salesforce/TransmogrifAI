/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import org.apache.spark.ml.linalg.Vectors

/**
 * Converts a sequence of real non nullable features into a vector feature
 *
 * @param uid uid for instance
 */
class RealNNVectorizer
(
  uid: String = UID[RealNNVectorizer]
) extends SequenceTransformer[RealNN, OPVector](operationName = "vecNum", uid = uid)
  with VectorizerDefaults {

  /**
   * Function used to convert input to output
   */
  override def transformFn: (Seq[RealNN]) => OPVector = in => {
    val ins = in.map(_.value.get) // assumes a non nullable real (RealNN)
    Vectors.dense(ins.toArray).toOPVector
  }

}
