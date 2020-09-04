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
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.stages.base.unary.UnaryTransformer

/**
* Joins probability score with label from string indexer stage
*
* @input i: RealNN - output feature from OPStringIndexer
* @input probs: OPVector - vector of probabilities from multiclass model
* @return Map(label -> probability)
*/
class MultiLabelJoiner
(
  operationName: String = classOf[MultiLabelJoiner].getSimpleName,
  uid: String = UID[MultiLabelJoiner]
) extends BinaryTransformer[RealNN, OPVector, RealMap](operationName = operationName, uid = uid) {

  private lazy val labels = {
    val schema = getInputSchema
    val meta = schema(in1.name).metadata
    meta.getMetadata("ml_attr").getStringArray("vals")
  }

  override def transformFn: (RealNN, OPVector) => RealMap = (i: RealNN, probs: OPVector) =>
    labels.zip(probs.value.toArray).toMap.toRealMap
}

/**
 * Sorts the label probability map and returns the topN.
 *
 * @topN: Int - maximum number of label/probability pairs to return
 * @labelProbMap: RealMap - Map(label -> probability)
 * @returns Map(label -> probability)
 */
class TopNLabelProbMap
(
  topN: Int,
  operationName: String = classOf[TopNLabelProbMap].getSimpleName,
  uid: String = UID[TopNLabelJoiner]
) extends UnaryTransformer[RealMap, RealMap](operationName = operationName, uid = uid) {

  override def transformFn: RealMap => RealMap = TopNLabelJoiner(topN)
}

/**
 * Joins probability score with label from string indexer stage
 * and
 * Sorts by highest score and returns up topN.
 * and
 * Filters out the class - UnseenLabel
 *
 * @input topN: Int - maximum number of label/probability pairs to return
 * @input i: RealNN - output feature from OPStringIndexer
 * @input probs: OPVector - vector of probabilities from multiclass model
 * @returns Map(label -> probability)
 */
class TopNLabelJoiner
(
  topN: Int,
  operationName: String = classOf[TopNLabelJoiner].getSimpleName,
  uid: String = UID[TopNLabelJoiner]
) extends MultiLabelJoiner(operationName = operationName, uid = uid) {

  override def transformFn: (RealNN, OPVector) => RealMap = (i: RealNN, probs: OPVector) => {
    val labelProbMap = super.transformFn(i, probs).value
    val filteredLabelProbMap = labelProbMap.filterKeys(_ != OpStringIndexerNoFilter.UnseenNameDefault)
    TopNLabelJoiner(topN)(filteredLabelProbMap.toRealMap)
  }

}

object TopNLabelJoiner {

  /**
   * Sorts the label probability map and returns the topN
   * @topN - maximum number of label/probability pairs to return
   * @labelProbMap - Map(label -> probability)
   */
  def apply(topN: Int)(labelProbMap: RealMap): RealMap = {
    labelProbMap
      .value.toArray
      .sortBy(-_._2)
      .take(topN)
      .toMap.toRealMap
  }

}

