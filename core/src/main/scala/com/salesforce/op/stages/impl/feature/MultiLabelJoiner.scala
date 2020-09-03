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

