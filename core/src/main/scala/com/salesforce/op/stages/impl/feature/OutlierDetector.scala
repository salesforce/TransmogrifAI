package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._

/**
 * @author nguyen.tuan
 * @since 214
 */
class OutlierDetector(uid: String = UID[OutlierDetector], operationName: String = "outlier detector")
  extends UnaryEstimator[RealNN, Binary](uid = uid, operationName = operationName) {
  override def fitFn(dataset: Dataset[Option[Double]]): UnaryModel[RealNN, Binary] = {

    val Array(mean, std) = dataset.summary("mean", "stddev")
      .select("value").collect

    val realSTD = std match {
      case Row("NaN") => 1.0
      case Row(null) => 1.0
      case Row(s: String) => s.toDouble
    }
    val realMean = mean match {
      case Row("NaN") => 1.0
      case Row(null) => 1.0
      case Row(s: String) => s.toDouble
    }

    new OutlierDetectorModel(operationName = operationName, uid = uid,
      mean = realMean, std = realSTD)

  }


}

class OutlierDetectorModel(operationName: String, uid: String, mean: Double, std: Double)
  extends UnaryModel[RealNN, Binary](uid = uid, operationName = operationName) {
  // override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  // scalastyle:off
  override def transformFn: RealNN => Binary = (value: RealNN) => {
    val zScore = (value.value.get - mean)/std
    (math.abs(zScore) >= 1.96).toBinary
  }
  // scalastyle:on
}
