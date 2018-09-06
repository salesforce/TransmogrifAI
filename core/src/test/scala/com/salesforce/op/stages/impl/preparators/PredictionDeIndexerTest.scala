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

package com.salesforce.op.stages.impl.preparators


import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.OpStringIndexerNoFilter
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class PredictionDeIndexerTest extends FlatSpec with TestSparkContext {

  val data = Seq(("a", 0.0), ("b", 1.0), ("c", 2.0)).map { case (txt, num) => (txt.toText, num.toRealNN) }
  val (ds, txtF, numF) = TestFeatureBuilder(data)

  val response = txtF.indexed()
  val indexedData = response.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(ds).transform(ds)

  val permutation = new UnaryLambdaTransformer[RealNN, RealNN](
    operationName = "modulo",
    transformFn = v => ((v.value.get + 1).toInt % 3).toRealNN
  ).setInput(response)
  val pred = permutation.getOutput()
  val permutedData = permutation.transform(indexedData)

  val expected = Array("b", "c", "a").map(_.toText)

  Spec[PredictionDeIndexer] should "deindexed the feature correctly" in {
    val predDeIndexer = new PredictionDeIndexer().setInput(response, pred)
    val deIndexed = predDeIndexer.getOutput()

    val results = predDeIndexer.fit(permutedData).transform(permutedData).collect(deIndexed)
    results shouldBe expected
  }


  it should "throw a nice error when there is no metadata" in {
    val predDeIndexer = new PredictionDeIndexer().setInput(numF, pred)
    the[Error] thrownBy {
      predDeIndexer.fit(permutedData).transform(permutedData)
    } should have message
      s"The feature ${numF.name} does not contain any label/index mapping in its metadata"
  }
}
