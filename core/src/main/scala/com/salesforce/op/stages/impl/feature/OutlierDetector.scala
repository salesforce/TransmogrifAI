package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

/**
 * @author nguyen.tuan
 * @since 214
 */
class OutlierDetector(uid: String = UID[OutlierDetector], operationName: String = "outlier detector")
  extends UnaryEstimator[RealNN, Binary](uid = uid, operationName = operationName) {
  override def fitFn(dataset: Dataset[Option[Double]]): UnaryModel[RealNN, Binary] = {
    val minElement = dataset.agg(min(col("value"))).first().toSeq.head.asInstanceOf[Double]
    new OutlierDetectorModel(operationName = operationName, uid = uid, minElement = minElement)

  }


}

class OutlierDetectorModel(operationName: String, uid: String, minElement: Double)
  extends UnaryModel[RealNN, Binary](uid = uid, operationName = operationName) {
  // override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  // scalastyle:off
  override def transformFn: RealNN => Binary = (value: RealNN) => (value.value.get > minElement).toBinary
  // scalastyle:on
}
