/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.{BinaryEstimator, BinaryModel}
import com.salesforce.op.stages.impl.feature.{OpIndexToStringNoFilter, SaveOthersParams}
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.sql.Dataset

import scala.util.{Failure, Success, Try}

/**
 * Estimator which takes response feature and predinction feature as inputs. It deindexes the pred by using response's
 * metadata
 *
 * Input 1 : response
 * Input 2 : pred feature
 *
 * @param uid
 */
class PredictionDeIndexer(uid: String = UID[PredictionDeIndexer])
  extends BinaryEstimator[RealNN, RealNN, Text](operationName = "idx2str", uid = uid) with SaveOthersParams {

  setDefault(unseenName, OpIndexToStringNoFilter.unseenDefault)

  /**
   * Function used to convert input to output
   */
  override def fitFn(dataset: Dataset[(Option[Double], Option[Double])]): BinaryModel[RealNN, RealNN, Text] = {
    val colSchema = getInputSchema()(in1.name)
    val labels: Array[String] = Try(Attribute.fromStructField(colSchema).asInstanceOf[NominalAttribute].values.get)
    match {
      case Success(l) => l
      case Failure(l) => throw new Error(s"The feature ${in1.name} does not contain" +
        s" any label/index mapping in its metadata")
    }


    new PredictionDeIndexerModel(labels, $(unseenName), operationName, uid)
  }
}

private final class PredictionDeIndexerModel
(
  labels: Array[String],
  unseen: String,
  operationName: String,
  uid: String
) extends BinaryModel[RealNN, RealNN, Text](operationName = operationName, uid = uid) {
  override def transformFn: (RealNN, RealNN) => Text = {
    (response: RealNN, pred: RealNN) => {
      val idx = pred.value.get.toInt
      if (0 <= idx && idx < labels.length) {
        labels(idx).toText
      } else {
        unseen.toText
      }
    }
  }
}
