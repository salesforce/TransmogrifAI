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

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryModel}
import com.salesforce.op.test.{FeatureTypeEquality, OpEstimatorSpec, OpPipelineStageAsserts, TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{DataSources, RandomText}
import com.salesforce.op.utils.stages.GenderDetectStrategy
import com.salesforce.op.utils.stages.GenderDetectStrategy.FindHonorific
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TempTest
  extends FlatSpec
    with FeatureTypeEquality[NameStats]
    with TestSparkContext
    with OpPipelineStageAsserts {

  /**
   * Input Dataset to fit & transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq("NOTANAME").toText)

  /**
   * Estimator instance to be tested
   */
  val estimator: HumanNameDetector[Text] = new HumanNameDetector().setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult: Seq[NameStats] = Seq(NameStats(Map.empty[String, String]))

  private def identifyName(data: Seq[Text]) = {
    val (newData, newFeature) = TestFeatureBuilder(data)
    val model = estimator.setInput(newFeature).fit(newData)
    val result: DataFrame = model.transform(newData)
    (newData, newFeature, model, result)
  }

  it should "identify a Text column with multiple first name entries as Name" in {
    val names = DataSources.FirstNames.map(_.toText)
    val (_, _, model, _) = identifyName(names)
    model.asInstanceOf[HumanNameDetectorModel[Text]].treatAsName shouldBe true
  }
}
