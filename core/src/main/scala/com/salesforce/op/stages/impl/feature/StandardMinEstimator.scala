package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class StandardMinEstimator
(
  uid: String = UID[StandardMinEstimator]
)
  extends UnaryEstimator[Real, Real](operationName = "standardMin", uid = uid) {

  def fitFn(dataset: Dataset[Real#Value]): UnaryModel[Real, Real] = {
    val grouped = dataset.groupBy()
    val minVal = grouped.min().first().getDouble(0)

    val vecData: RDD[OldVector] = dataset.rdd.map(v => OldVectors.fromML(Vectors.dense(v.get)))
    val std = new StandardScaler().fit(vecData).std.toArray
    val stdVal = std.head

    val scalingArgs = LinearScalerArgs(1 / stdVal, - minVal / stdVal)
    val meta = ScalerMetadata(ScalingType.Linear, scalingArgs).toMetadata()
    setMetadata(meta)


    new StandardMinEstimatorModel(
      min = minVal,
      std = stdVal,
      operationName = operationName,
      uid = uid
    )
  }
}

final class StandardMinEstimatorModel private[op]
(
  val min: Double,
  val std: Double,
  operationName: String,
  uid: String
) extends UnaryModel[Real, Real](operationName = operationName, uid = uid) {
  def transformFn: Real => Real = r => r.v.map(v => (v - min) / std).toReal
}
