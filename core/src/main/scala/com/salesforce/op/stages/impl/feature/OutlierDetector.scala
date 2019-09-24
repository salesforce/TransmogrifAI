package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types.{Binary, RealNN}
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * @author nguyen.tuan
 * @since 214
 */
class OutlierDetector(uid: String = UID[OutlierDetector], operationName: String = "outlier detector")
  extends UnaryEstimator[RealNN, Binary](uid = uid, operationName = operationName) {
  override def fitFn(dataset: Dataset[Option[Double]]): UnaryModel[RealNN, Binary] = {
    new OutlierDetectorModel(operationName = operationName, uid = uid)

  }


}

class OutlierDetectorModel(operationName: String, uid: String)
  extends UnaryModel[RealNN, Binary](uid = uid, operationName = operationName) {
  override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  // scalastyle:off
  override def transformFn: RealNN => Binary = ???
  // scalastyle:on
}
