/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.classification.Impurity
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.RichNode._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}
import com.salesforce.op.utils.spark.RichMetadata._

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
  with DecisionTreeNumericBucketizerParams
  with VectorizerDefaults with TrackInvalidParam
  with TrackNullsParam with NumericBucketizerMetadata {

  // Since we bucketize based on the label information we override the response function allowing label leakage here,
  // except the cases where both input features are responses
  override protected def outputIsResponse: Boolean = getTransientFeatures().forall(_.isResponse)

  def fitFn(dataset: Dataset[(Option[Double], Option[N])]): BinaryModel[RealNN, I2, OPVector] = {
    import dataset.sparkSession.implicits._

    val (labelColumn, featureColumn) = in1.name -> in2.name
    val ds = dataset.filter(_._2.isDefined) // drop the missing feature values

    val splits: Array[Double] = {
      // Check if input data is empty - prevents from decision tree classifier to fail
      if (ds.isEmpty) {
        logWarning(s"Unable to find buckets for feature '$featureColumn' since it contains only empty values.")
        Array.empty
      }
      else {
        val data =
          ds.map { case (label, feature) => label -> Vectors.dense(Array(nev.toDouble(feature.get))) }
            .toDF(labelColumn, featureColumn)

        new DecisionTreeClassifier()
          .setImpurity(getImpurity)
          .setMaxDepth(getMaxDepth)
          .setMaxBins(getMaxBins)
          .setMinInstancesPerNode(getMinInstancesPerNode)
          .setMinInfoGain(getMinInfoGain)
          .setLabelCol(labelColumn)
          .setFeaturesCol(featureColumn)
          .fit(data)
          .rootNode.splits
      }
    }
    val theSplits = Double.NegativeInfinity +: splits :+ Double.PositiveInfinity

    // TODO: additionally add logic to inspect the gain from the trained decision tree model
    // and figure out if we need to split at all, otherwise simply produce an empty vector
    val shouldSplit = NumericBucketizer.checkSplits(theSplits)

    val (finalSplits, bucketLabels) =
      if (shouldSplit) {
        theSplits -> NumericBucketizer.splitsToBucketLabels(theSplits, DecisionTreeNumericBucketizer.Inclusion)
      }
      else Array.empty[Double] -> Array.empty[String]

    val meta = makeMetadata(shouldSplit, finalSplits, bucketLabels)
    setMetadata(meta)

    new DecisionTreeNumericBucketizerModel[N, I2](
      shouldSplit = shouldSplit,
      splits = finalSplits,
      trackNulls = $(trackNulls),
      trackInvalid = $(trackInvalid),
      operationName = operationName,
      uid = uid
    )
  }

  private def makeMetadata(shouldSplit: Boolean, splits: Array[Double], bucketLabels: Array[String]): Metadata = {
    val vectorMeta = makeVectorMetadata(
      input = in2, bucketLabels = bucketLabels,
      trackNulls = shouldSplit && $(trackNulls),
      trackInvalid = shouldSplit && $(trackInvalid)
    )
    val splitsMeta = new MetadataBuilder()
      .putBoolean(DecisionTreeNumericBucketizer.ShouldSplitKey, shouldSplit)
      .putDoubleArray(DecisionTreeNumericBucketizer.SplitsKey, splits)
      .build()
    vectorMeta.toMetadata.withSummaryMetadata(splitsMeta)
  }

}

final class DecisionTreeNumericBucketizerModel[N, I2 <: OPNumeric[N]] private[op]
(
  val shouldSplit: Boolean,
  val splits: Array[Double],
  val trackNulls: Boolean,
  val trackInvalid: Boolean,
  operationName: String,
  uid: String
)(
  implicit tti2: TypeTag[I2],
  val nev: Numeric[N]
) extends BinaryModel[RealNN, I2, OPVector](operationName = operationName, uid = uid) {

  def transformFn: (RealNN, I2) => OPVector = (_, input) =>
    if (shouldSplit) NumericBucketizer.bucketize[N, I2](
      splits = splits,
      trackNulls = trackNulls,
      trackInvalid = trackInvalid,
      splitInclusion = DecisionTreeNumericBucketizer.Inclusion,
      input
    )
    else OPVector.empty

}

trait DecisionTreeNumericBucketizerParams extends Params {

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
  val MinInfoGain: Double = 0.01
  val ShouldSplitKey = "shouldSplit"
  val SplitsKey = "splits"
  val Inclusion = com.salesforce.op.stages.impl.feature.Inclusion.Right
}
