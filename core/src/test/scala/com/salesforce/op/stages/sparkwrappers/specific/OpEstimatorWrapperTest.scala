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

package com.salesforce.op.stages.sparkwrappers.specific

import com.salesforce.op.features.types._
import com.salesforce.op.test.{PrestigeData, TestFeatureBuilder, _}
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpEstimatorWrapperTest extends FlatSpec with TestSparkContext with PrestigeData {

  val log = LoggerFactory.getLogger(this.getClass)

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
    wrappedEstimator.getOutput().name shouldBe wrappedEstimator.getOutputFeatureName
  }

  private def fitScalerModel(baseScaler: MinMaxScaler): MinMaxScalerModel = {
    val scaler =
      new OpEstimatorWrapper[OPVector, OPVector, MinMaxScaler, MinMaxScalerModel](baseScaler).setInput(income)

    val model = scaler.fit(ds)
    val scalerModel = model.getSparkMlStage().get

    if (log.isInfoEnabled) {
      val output = scalerModel.transform(ds)
      output.show(false)
    }
    scalerModel
  }
}
