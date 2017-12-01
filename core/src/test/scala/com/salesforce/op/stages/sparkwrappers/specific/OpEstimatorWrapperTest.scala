/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.test.{PrestigeData, TestFeatureBuilder, _}
import com.salesforce.op.features.types._
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpEstimatorWrapperTest extends FlatSpec with TestSparkContext with PrestigeData {

  val (ds, education, income, women, prestige) =
    TestFeatureBuilder[OPVector, OPVector, OPVector, OPVector]("education", "income", "women", "prestige",
      prestigeSeq.map(p =>
        (Vectors.dense(p.prestige).toOPVector, Vectors.dense(p.education).toOPVector,
          Vectors.dense(p.income).toOPVector, Vectors.dense(p.women).toOPVector)
      )
    )

  Spec[OpEstimatorWrapper[_, _, _, _]] should "scale variables properly with default min/max params" in {
    val baseScaler = new MinMaxScaler()
    val scalerModel: MinMaxScalerModel = fitScalerModel(baseScaler)

    (scalerModel.getMax - 1.0).abs should be < 1E-6
  }

  it should "scale variables properly with custom min/max params" in {
    val maxParam = 100
    val baseScaler = new MinMaxScaler().setMax(maxParam)
    val scalerModel: MinMaxScalerModel = fitScalerModel(baseScaler)

    (scalerModel.getMax - maxParam).abs should be < 1E-6
  }

  it should "should have the expected feature name" in {
    val wrappedEstimator =
      new OpEstimatorWrapper[OPVector, OPVector, MinMaxScaler, MinMaxScalerModel](new MinMaxScaler()).setInput(income)
    wrappedEstimator.getOutput().name shouldBe wrappedEstimator.outputName
  }

  private def fitScalerModel(baseScaler: MinMaxScaler): MinMaxScalerModel = {
    val scaler =
      new OpEstimatorWrapper[OPVector, OPVector, MinMaxScaler, MinMaxScalerModel](baseScaler).setInput(income)

    val model = scaler.fit(ds)
    val scalerModel = model.getSparkMlStage().get
    val output = scalerModel.transform(ds)
    output.show(false)
    scalerModel
  }
}
