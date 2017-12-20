/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.classification.Impurity
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.RichNode._
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

/**
 * Smart bucketizer for numeric values based on a Decision Tree classifier.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti2          type tag for numeric feature type
 * @param nev           numeric evidence for feature type value
 * @tparam N  numeric feature type value
 * @tparam I2 numeric feature type
 */
class DecisionTreeNumericBucketizer[N, I2 <: OPNumeric[N]]
(
  operationName: String = "dtNumBuck",
  uid: String = UID[DecisionTreeNumericBucketizer[_, _]]
)(
  implicit tti2: TypeTag[I2],
  val nev: Numeric[N]
) extends BinaryEstimator[RealNN, I2, OPVector](operationName = operationName, uid = uid)
  with SmartNumericBucketizerParams with TrackNullsParam {

  def fitFn(dataset: Dataset[(Option[Double], Option[N])]): BinaryModel[RealNN, I2, OPVector] = {
    import dataset.sparkSession.implicits._

    // drop the missing values and vectorize
    val data = dataset.filter(_._2.isDefined).map { case (response, in2) =>
      response -> Vectors.dense(Array(nev.toDouble(in2.get)))
    }.toDF(in1.name, in2.name)

    val decisionTree = new DecisionTreeClassifier()
      .setImpurity(getImpurity)
      .setMaxDepth(getMaxDepth)
      .setMaxBins(getMaxBins)
      .setMinInstancesPerNode(getMinInstancesPerNode)
      .setMinInfoGain(getMinInfoGain)
      .setLabelCol(in1.name)
      .setFeaturesCol(in2.name)

    val decisionTreeModel = decisionTree.fit(data)
    val splits = Double.NegativeInfinity +: decisionTreeModel.rootNode.splits :+ Double.PositiveInfinity

    // TODO: additionaly add logic to inspect the gain from the trained decision tree model
    // and figure out if we need to split at all, otherwise simply produce an empty vector
    val finalSplits = if (NumericBucketizer.checkSplits(splits)) splits else Array.empty[Double]
    val shouldSplit = finalSplits.length > 2

    new SmartNumericBucketizerModel[N, I2](
      shouldSplit = shouldSplit, splits = finalSplits,
      trackNulls = $(trackNulls), operationName = operationName, uid = uid
    )
  }
}

private final class SmartNumericBucketizerModel[N, I2 <: OPNumeric[N]]
(
  val shouldSplit: Boolean,
  val splits: Array[Double],
  val trackNulls: Boolean,
  operationName: String,
  uid: String
)(
  implicit tti2: TypeTag[I2],
  val nev: Numeric[N]
) extends BinaryModel[RealNN, I2, OPVector](operationName = operationName, uid = uid) {

  private lazy val bucketizer =
    new NumericBucketizer[N, I2](operationName = operationName, uid = uid)
      .setBuckets(splits)
      .setTrackNulls(trackNulls)
      .setInput(in2.asFeatureLike[I2])

  def transformFn: (RealNN, I2) => OPVector = (_, input) => {
    if (shouldSplit) bucketizer.transformFn(input) else OPVector.empty
  }

}

trait SmartNumericBucketizerParams extends Params {

  /**
   * Criterion used for information gain calculation (case-insensitive).
   * Supported: "entropy" and "gini".
   * (default = gini)
   *
   * @group param
   */
  final val impurity: Param[String] = new Param[String](this, "impurity", "Criterion used for" +
    " information gain calculation (case-insensitive). Supported options: entropy & gini",
    (value: String) => Impurity.values.map(_.sparkName).contains(value.toLowerCase))

  /** @group setParam */
  final def setImpurity(value: Impurity): this.type = {
    set(impurity, value.sparkName)
    this
  }

  /** @group setParam */
  final def getImpurity: String = $(impurity).toLowerCase


  /**
   * Maximum depth of the tree (>= 0).
   * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * (default = 5)
   *
   * @group param
   */
  final val maxDepth: IntParam =
    new IntParam(this, "maxDepth", "Maximum depth of the tree. (>= 0)" +
      " E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.",
      ParamValidators.gtEq(0))

  /** @group setParam */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group getParam */
  final def getMaxDepth: Int = $(maxDepth)

  // TODO : Use a ratio instead : the percentage of number unique instances
  /**
   * Maximum number of bins
   * Must be >= 2 and <= number of categories in any categorical feature.
   * (default = 32)
   *
   * @group param
   */
  final val maxBins: IntParam = new IntParam(this, "maxBins", "Max number of bins for" +
    " discretizing continuous features.  Must be >=2 and >= number of categories for any" +
    " categorical feature.", ParamValidators.gtEq(2))

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group getParam */
  final def getMaxBins: Int = $(maxBins)

  /**
   * Minimum number of instances each child must have after split.
   * If a split causes the left or right child to have fewer than minInstancesPerNode,
   * the split will be discarded as invalid.
   * Should be >= 1.
   * (default = 1)
   *
   * @group param
   */
  final val minInstancesPerNode: IntParam = new IntParam(this, "minInstancesPerNode", "Minimum" +
    " number of instances each child must have after split.  If a split causes the left or right" +
    " child to have fewer than minInstancesPerNode, the split will be discarded as invalid." +
    " Should be >= 1.", ParamValidators.gtEq(1))

  /** @group setParam */
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group getParam */
  final def getMinInstancesPerNode: Int = $(minInstancesPerNode)

  /**
   * Minimum information gain for a split to be considered at a tree node.
   * Should be >= 0.0.
   * (default = 0.0)
   *
   * @group param
   */
  final val minInfoGain: DoubleParam = new DoubleParam(this, "minInfoGain",
    "Minimum information gain for a split to be considered at a tree node.",
    ParamValidators.gtEq(0.0))

  /** @group setParam */
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group getParam */
  final def getMinInfoGain: Double = $(minInfoGain)

  setDefault(
    maxDepth -> DecisionTreeNumericBucketizer.MaxDepth,
    maxBins -> DecisionTreeNumericBucketizer.MaxBins,
    minInstancesPerNode -> DecisionTreeNumericBucketizer.MinInstancesPerNode,
    minInfoGain -> DecisionTreeNumericBucketizer.MinInfoGain,
    impurity -> DecisionTreeNumericBucketizer.Impurity.sparkName
  )

}

case object DecisionTreeNumericBucketizer {
  val Impurity = com.salesforce.op.stages.impl.classification.Impurity.Gini
  val MaxDepth: Int = 5
  val MaxBins: Int = 32
  val MinInstancesPerNode: Int = 1
  val MinInfoGain: Double = 0.1
}
