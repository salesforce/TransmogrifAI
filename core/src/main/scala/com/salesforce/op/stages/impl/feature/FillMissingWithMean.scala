/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
