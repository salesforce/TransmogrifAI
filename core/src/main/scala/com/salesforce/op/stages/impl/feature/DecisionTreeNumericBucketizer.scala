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
import com.salesforce.op.stages.AllowLabelAsInput
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.classification.Impurity
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.RichNode._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.Metadata

import scala.reflect.runtime.universe.TypeTag

/**
 * Smart bucketizer for numeric values based on a Decision Tree classifier.
 *
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti2          type tag for numeric feature type
 * @param ttiv2         type tag for numeric feature value type
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
  ttiv2: TypeTag[I2#Value],
  val nev: Numeric[N]
) extends BinaryEstimator[RealNN, I2, OPVector](operationName = operationName, uid = uid)
  with DecisionTreeNumericBucketizerParams
  with VectorizerDefaults with TrackInvalidParam
  with TrackNullsParam with NumericBucketizerMetadata
  with AllowLabelAsInput[OPVector] {

  def fitFn(dataset: Dataset[(Option[Double], Option[N])]): BinaryModel[RealNN, I2, OPVector] = {
    import dataset.sparkSession.implicits._

    require(!dataset.isEmpty, "Dataset is empty, buckets cannot be computed.")

    val data: Dataset[(Double, Double)] =
      dataset
        .filter(_._2.isDefined) // drop the missing feature values
        .map { case (l, v) => l.get -> nev.toDouble(v.get) }

    val Splits(shouldSplit, finalSplits, bucketLabels) = computeSplits(data, featureName = in2.name)

    val meta = makeMetadata(shouldSplit, finalSplits, bucketLabels)
    setMetadata(meta)

    new DecisionTreeNumericBucketizerModel[I2](
      shouldSplit = shouldSplit,
      splits = finalSplits,
      trackNulls = $(trackNulls),
      trackInvalid = $(trackInvalid),
      operationName = operationName,
      uid = uid
    )
  }

  private def makeMetadata(shouldSplit: Boolean, splits: Array[Double], bucketLabels: Array[String]): Metadata =
    makeVectorMetadata(
      input = in2,
      bucketLabels = bucketLabels,
      trackNulls = $(trackNulls),
      trackInvalid = shouldSplit && $(trackInvalid)
    ).toMetadata

}

final class DecisionTreeNumericBucketizerModel[I2 <: OPNumeric[_]] private[op]
(
  val shouldSplit: Boolean,
  val splits: Array[Double],
  val trackNulls: Boolean,
  val trackInvalid: Boolean,
  operationName: String,
  uid: String
)(implicit tti2: TypeTag[I2])
  extends BinaryModel[RealNN, I2, OPVector](operationName = operationName, uid = uid)
  with AllowLabelAsInput[OPVector] {

  def transformFn: (RealNN, I2) => OPVector = (_, input) =>
    NumericBucketizer.bucketize(
      shouldSplit = shouldSplit,
      splits = splits,
      trackNulls = trackNulls,
      trackInvalid = trackInvalid,
      splitInclusion = DecisionTreeNumericBucketizer.Inclusion,
      input = input.toDouble
    ).toOPVector

}


trait DecisionTreeNumericBucketizerParams {
  self: PipelineStage =>

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

  /**
   * Computed splits
   *
   * @param shouldSplit  should or not split
   * @param splits       computed split values
   * @param bucketLabels bucket labels
   */
  case class Splits(shouldSplit: Boolean, splits: Array[Double], bucketLabels: Array[String])

  /**
   * Compute splits using [[DecisionTreeClassifier]]
   *
   * @param data        input dataset of (label, feature) tuples
   * @param featureName feature name
   * @return computed [[Splits]]
   */
  protected def computeSplits(data: Dataset[(Double, Double)], featureName: String): Splits = {
    import data.sparkSession.implicits._

    val splits: Array[Double] = {
      // Check if input data is empty - prevents from decision tree classifier to fail
      if (data.isEmpty) {
        logWarning(s"Unable to find buckets for feature '$featureName' since it contains only empty values.")
        Array.empty
      }
      else {
        val ds = data.map { case (label, v) => label -> Vectors.dense(Array(v)) }
        new DecisionTreeClassifier()
          .setImpurity(getImpurity)
          .setMaxDepth(getMaxDepth)
          .setMaxBins(getMaxBins)
          .setMinInstancesPerNode(getMinInstancesPerNode)
          .setMinInfoGain(getMinInfoGain)
          .setLabelCol(ds.columns(0))
          .setFeaturesCol(ds.columns(1))
          .fit(ds)
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
      } else Array.empty[Double] -> Array.empty[String]

    Splits(shouldSplit, finalSplits, bucketLabels)
  }

}

case object DecisionTreeNumericBucketizer {
  val Impurity = com.salesforce.op.stages.impl.classification.Impurity.Gini
  val MaxDepth: Int = 5
  val MaxBins: Int = 32
  val MinInstancesPerNode: Int = 1
  val MinInfoGain: Double = 0.01
  val Inclusion = com.salesforce.op.stages.impl.feature.Inclusion.Right
}
