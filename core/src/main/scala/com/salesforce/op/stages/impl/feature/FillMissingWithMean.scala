/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.FeatureSparkTypes
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe._


/**
 * Fill missing values with mean for any numeric feature
 */
class FillMissingWithMean[N, I <: OPNumeric[N]]
(
  uid: String = UID[FillMissingWithMean[_, _]]
)(implicit tti: TypeTag[I], ttiv: TypeTag[I#Value])
  extends UnaryEstimator[I, RealNN](operationName = "fillWithMean", uid = uid) {

  val defaultValue = new DoubleParam(this, "defaultValue", "default value to replace the missing ones")
  set(defaultValue, 0.0)

  def setDefaultValue(v: Double): this.type = set(defaultValue, v)

  private implicit val dEncoder = FeatureSparkTypes.featureTypeEncoder[Real]

  def fitFn(dataset: Dataset[Option[N]]): UnaryModel[I, RealNN] = {
    val grouped = dataset.map(v => iConvert.ftFactory.newInstance(v).toDouble).groupBy()
    val mean = grouped.mean().first().getOption[Double](0).getOrElse($(defaultValue))
    new FillMissingWithMeanModel[I](mean = mean, operationName = operationName, uid = uid)
  }

}

final class FillMissingWithMeanModel[I <: OPNumeric[_]] private[op]
(
  val mean: Double,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[I])
  extends UnaryModel[I, RealNN](operationName = operationName, uid = uid) {
  def transformFn: I => RealNN = _.toDouble.getOrElse(mean).toRealNN
}
