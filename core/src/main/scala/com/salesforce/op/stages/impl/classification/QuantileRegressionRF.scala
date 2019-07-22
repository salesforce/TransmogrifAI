package com.salesforce.op.stages.impl.classification

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types._
import org.apache.spark.ml.tree.RichNode._
import org.apache.spark.ml.tree.RichOldNode._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Encoders, Row, SparkSession}
import org.apache.spark.sql.expressions.Window

class QuantileRegressionRF(uid: String = UID[QuantileRegressionRF], operationName: String = "quantiles",
  val trees: Array[DecisionTreeRegressionModel])
  extends BinaryEstimator[RealNN, OPVector, RealMap](operationName = operationName, uid = uid) {

  val percentageLevel = new DoubleParam(
    parent = this, name = "percentageLevel",
    doc = "level of prediction interval",
    isValid = ParamValidators.inRange(lowerBound = 0.0, upperBound = 1.0)
  )

  final def setPercentageLevel(v: Double): this.type = set(percentageLevel, v)

  final def getPercentageLevel: Double = $(percentageLevel)

  setDefault(percentageLevel -> 0.95)

  implicit val encoderLeafNode: Encoder[(Option[Double], Array[(Int, Long)])] = Encoders.kryo[(Option[Double],
    Array[(Int, Long)])]
  private implicit val encoderInt: Encoder[Int] = ExpressionEncoder[Int]()
  private implicit val encoderIntLong: Encoder[(Int, Long)] = ExpressionEncoder[(Int, Long)]()


  override def fitFn(dataset: Dataset[(Option[Double], linalg.Vector)]): BinaryModel[RealNN, OPVector, RealMap] = {

    val lowerLevel = (1 - getPercentageLevel) / 2.0
    val upperLevel = (1 + getPercentageLevel) / 2.0

    val leaves = dataset.map { case (l, f) =>
      l -> trees.map { case tree =>
        val node = tree.rootNode
        val leaf = node.predictImpl(f)
        val oldNode = node.toOld(1)
        val leafID = oldNode.predictImplIdx(f)
        leafID -> leaf.stats.count
      }
    }
    val T = trees.length

    new QuantileRegressionRFModels(leaves.collect, trees, lowerLevel, upperLevel, T, operationName, uid)

  }
}


class QuantileRegressionRFModels(leaves: Seq[(Option[Double], Array[(Int, Long)])],
  trees: Array[DecisionTreeRegressionModel], lowerLevel: Double, upperLevel: Double,
  T: Int, operationName: String, uid: String)
  extends BinaryModel[RealNN, OPVector, RealMap](operationName = operationName, uid = uid) {

  private implicit val encoder: Encoder[(Double, Option[Double])] = ExpressionEncoder[(Double, Option[Double])]()


  def transformFn: (RealNN, OPVector) => RealMap = {

    (l: RealNN, f: OPVector) => {
      val pred_leaves = trees.map(_.rootNode.toOld(1).predictImplIdx(f.value))
      val weightsSeq = leaves.map { case (y, y_leaves) => y_leaves.zip(pred_leaves).zipWithIndex.map {
        case (((l1, count), l2), i) =>
          if (l1 == l2) {
            1.0 / count
          } else 0.0
      }.sum / T -> y
      }


      val cumF = weightsSeq.sortBy(_._2).scan(0.0 -> Option(0.0)) { case ((v, _), (w: Double, l: Option[Double])) =>
        v + w -> l
      }.tail

      val qLower = cumF.filter {
        _._1 >= lowerLevel
      }.head._2
      val qUpper = cumF.filter {
        _._1 >= upperLevel
      }.head._2

      new RealMap(Map("lowerQuantile" -> qLower.get, "upperQuantile" -> qUpper.get))
    }
  }


}

