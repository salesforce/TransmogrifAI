/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpMinMaxEstimatorReaderWriterTest extends OpPipelineStageReaderWriterTest {
  private val minMax = new MinMaxNormEstimator().setInput(weight).setMetadata(meta)

  val stage: OpPipelineStageBase = minMax.fit(passengersDataSet)

  val expected =
    Array(1.0.toReal, Real.empty, 0.10476190476190476.toReal, 0.0.toReal, 0.2761904761904762.toReal, 0.0.toReal)
}


class MinMaxNormEstimator(uid: String = UID[MinMaxNormEstimator])
  extends UnaryEstimator[Real, Real](operationName = "minMaxNorm", uid = uid) {

  def fitFn(dataset: Dataset[Real#Value]): UnaryModel[Real, Real] = {
    val grouped = dataset.groupBy()
    val maxVal = grouped.max().first().getDouble(0)
    val minVal = grouped.min().first().getDouble(0)
    new MinMaxNormEstimatorModel(
      min = minVal,
      max = maxVal,
      seq = Seq(minVal, maxVal),
      map = Map("a" -> Map("b" -> 1.0, "c" -> 2.0), "d" -> Map.empty),
      operationName = operationName,
      uid = uid
    )
  }
}

final class MinMaxNormEstimatorModel private[op]
(
  val min: Double,
  val max: Double,
  val seq: Seq[Double],
  val map: Map[String, Map[String, Double]],
  operationName: String, uid: String
) extends UnaryModel[Real, Real](operationName = operationName, uid = uid) {
  def transformFn: Real => Real = r => r.map(v => (v - min) / (max - min)).toReal
}
