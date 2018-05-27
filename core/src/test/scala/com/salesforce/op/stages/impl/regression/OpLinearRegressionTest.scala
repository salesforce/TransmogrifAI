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

package com.salesforce.op.stages.impl.regression

import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.preparators.SanityChecker
import com.salesforce.op.stages.sparkwrappers.generic._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegressionModel
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OpLinearRegressionTest extends FlatSpec with TestSparkContext {
  val stageNames = Array[String]("LinearRegression_predictionCol")

  val (ds, rawLabel, features) = TestFeatureBuilder(
    Seq[(RealNN, OPVector)](
      (10.0.toRealNN, Vectors.dense(1.0, 4.3, 1.3).toOPVector),
      (20.0.toRealNN, Vectors.dense(2.0, 0.3, 0.1).toOPVector),
      (30.0.toRealNN, Vectors.dense(3.0, 3.9, 4.3).toOPVector),
      (40.0.toRealNN, Vectors.dense(4.0, 1.3, 0.9).toOPVector),
      (50.0.toRealNN, Vectors.dense(5.0, 4.7, 1.3).toOPVector)
      )
  )
  val label = rawLabel.copy(isResponse = true)
  val linReg = new OpLinearRegression().setInput(label, features)

  Spec[OpLinearRegression] should "have output with correct origin stage" in {
    val output = linReg.getOutput()
    assert(output.originStage.isInstanceOf[SwBinaryEstimator[_, _, _, _, _]])
    the[IllegalArgumentException] thrownBy {
      linReg.setInput(label.copy(isResponse = true), features.copy(isResponse = true))
    } should have message "The feature vector should not contain any response features."
  }

  it should "return a properly formed LinearRegressionModel when fitted" in {
    val model = linReg.setSparkParams("maxIter", 10).fit(ds)
    assert(model.isInstanceOf[SwBinaryModel[RealNN, OPVector, RealNN, LinearRegressionModel]])

    val sparkStage = model.getSparkMlStage()

    sparkStage.isDefined shouldBe true
    sparkStage.get shouldBe a[LinearRegressionModel]

    val inputNames = model.getInputFeatures().map(_.name)
    inputNames should have length 2
    inputNames shouldBe Array(label.name, features.name)
  }


  it should "allow the user to set the desired spark parameters" in {
    linReg.setMaxIter(10).setRegParam(0.1)
    linReg.getMaxIter shouldBe 10
    linReg.getRegParam shouldBe 0.1

    linReg.setFitIntercept(true).setElasticNetParam(0.1).setSolver("normal")
    linReg.getFitIntercept shouldBe true
    linReg.getElasticNetParam shouldBe 0.1
    linReg.getSolver shouldBe "normal"
  }
}
