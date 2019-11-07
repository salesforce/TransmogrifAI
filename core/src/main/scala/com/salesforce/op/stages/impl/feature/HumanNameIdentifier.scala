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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.utils.stages.NameIdentificationFun
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamValidators}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder

import scala.reflect.runtime.universe.TypeTag

/**
 * Unary estimator for identifying whether a single Text column is a name or not. If the column does appear to be a
 * name, a custom map will be returned that contains the guessed gender for each entry. If the column does not appear
 * to be a name, then the output will be an empty map.
 * @param uid           uid for instance
 * @param operationName unique name of the operation this stage performs
 * @param tti           type tag for input
 * @param ttiv          type tag for input value
 * @tparam T            the FeatureType (subtype of Text) to operate over
 */
class HumanNameIdentifier[T <: Text]
(
  uid: String = UID[HumanNameIdentifier[T]],
  operationName: String = "human name identifier"
)
(
  implicit tti: TypeTag[T],
  override val ttiv: TypeTag[T#Value]
) extends UnaryEstimator[T, NameStats](
  uid = uid,
  operationName = operationName
) with NameIdentificationFun[T] {
  override lazy val spark: SparkSession = SparkSession.builder().getOrCreate()

  // Parameters
  val defaultThreshold = new DoubleParam(
    parent = this,
    name = "defaultThreshold",
    doc = "default fraction of entries to be names before treating as name",
    isValid = (value: Double) => {
      ParamValidators.gt(0.0)(value) && ParamValidators.lt(1.0)(value)
    }
  )
  setDefault(defaultThreshold, 0.50)
  def setThreshold(value: Double): this.type = set(defaultThreshold, value)

  val countApproxTimeout = new IntParam(
    parent = this,
    name = "countApproxTimeout",
    doc = "how long to wait (in milliseconds) for result of dataset.rdd.countApprox",
    isValid = (value: Int) => { ParamValidators.gt(0)(value) }
  )
  setDefault(countApproxTimeout, 3 * 60 * 1000)
  def setCountApproxTimeout(value: Int): this.type = set(countApproxTimeout, value)

  def fitFn(dataset: Dataset[T#Value]): HumanNameIdentifierModel[T] = {
    require(dataset.schema.fieldNames.length == 1, "There is exactly one column in this dataset")

    val column = col(dataset.schema.fieldNames.head)
    val (predictedNameProb, treatAsName, bestFirstNameIndex) = unaryEstimatorFitFn(
      dataset, column, $(defaultThreshold), $(countApproxTimeout)
    )

    // modified from: https://docs.transmogrif.ai/en/stable/developer-guide/index.html#metadata
    val preExistingMetadata = getMetadata()
    val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)
    metaDataBuilder.putBoolean("treatAsName", treatAsName)
    metaDataBuilder.putLong("predictedNameProb", predictedNameProb.toLong)
    metaDataBuilder.putLong("bestFirstNameIndex", bestFirstNameIndex.getOrElse(-1).toLong)
    val updatedMetadata = metaDataBuilder.build()
    setMetadata(updatedMetadata)

    new HumanNameIdentifierModel[T](uid, treatAsName, indexFirstName = bestFirstNameIndex)
  }
}


class HumanNameIdentifierModel[T <: Text]
(
  override val uid: String,
  val treatAsName: Boolean,
  val indexFirstName: Option[Int] = None
)(implicit tti: TypeTag[T])
  extends UnaryModel[T, NameStats]("human name identifier", uid) with NameIdentificationFun[T] {
  override lazy val spark: SparkSession = SparkSession.builder().getOrCreate()
  def transformFn: T => NameStats = (input: T) => transformerFn(treatAsName, indexFirstName, input)
}
