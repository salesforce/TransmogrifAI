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
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.test.{SwTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpHashingTFTest extends SwTransformerSpec[OPVector, HashingTF, OpHashingTF] {

  // scalastyle:off
  val testData = Seq(
    "Hamlet: To be or not to be - that is the question.",
    "Гамлет: Быть или не быть - вот в чём вопрос.",
    "המלט: להיות או לא להיות - זאת השאלה.",
    "Hamlet: Être ou ne pas être - telle est la question."
  ).map(_.toLowerCase.split(" ").toSeq.toTextList)
  // scalastyle:on

  val (inputData, f1): (DataFrame, Feature[TextList]) = TestFeatureBuilder(testData)

  val hashed = f1.tf(numTerms = 5)
  val transformer = hashed.originStage.asInstanceOf[OpHashingTF]

  val expectedResult: Seq[OPVector] = Seq(
    Vectors.sparse(5, Array(0, 1, 2, 3, 4), Array(4.0, 1.0, 3.0, 2.0, 2.0)),
    Vectors.sparse(5, Array(0, 1, 2, 3), Array(1.0, 5.0, 3.0, 1.0)),
    Vectors.sparse(5, Array(0, 1, 2, 3), Array(1.0, 2.0, 3.0, 2.0)),
    Vectors.sparse(5, Array(0, 2, 3, 4), Array(1.0, 4.0, 2.0, 4.0))
  ).map(_.toOPVector)

  def hash(
    s: String,
    numOfFeatures: Int = TransmogrifierDefaults.DefaultNumOfFeatures,
    binary: Boolean = false
  ): Int = {
    val hashingTF = new org.apache.spark.ml.feature.HashingTF
    hashingTF.setNumFeatures(numOfFeatures).setBinary(binary).indexOf(s)
  }

  it should "hash categorical data" in {
    val hashed = f1.tf()
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(inputData)
    val results = transformedData.select(hashed.name).collect(hashed)

    hashed.name shouldBe hashed.originStage.getOutputFeatureName

    // scalastyle:off
    results.forall(_.value.size == TransmogrifierDefaults.DefaultNumOfFeatures) shouldBe true
    results(0).value(hash("be")) shouldBe 2.0
    results(0).value(hash("that")) shouldBe 1.0
    results(1).value(hash("быть")) shouldBe 2.0
    results(2).value(hash("להיות")) shouldBe 2.0
    results(3).value(hash("être")) shouldBe 2.0
    // scalastyle:on
  }

  it should "hash categorical data with custom numFeatures" in {
    val numFeatures = 100

    val hashed = f1.tf(numTerms = numFeatures)
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(inputData)
    val results = transformedData.select(hashed.name).collect(hashed)

    // scalastyle:off
    results.forall(_.value.size == numFeatures) shouldBe true
    results(0).value(hash("be", numOfFeatures = numFeatures)) shouldBe 2.0
    results(1).value(hash("быть", numOfFeatures = numFeatures)) shouldBe 2.0
    results(2).value(hash("question", numOfFeatures = numFeatures)) shouldBe 0.0
    // scalastyle:on
  }

  it should "hash categorical data when binary = true" in {
    val binary = true

    val hashed = f1.tf(binary = binary)
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(inputData)
    val results = transformedData.select(hashed.name).collect(hashed)

    // scalastyle:off
    val values = Set(0.0, 1.0)
    results.forall(_.value.toArray.forall(values contains _)) shouldBe true
    results(0).value(hash("be", binary = binary)) shouldBe 1.0
    results(1).value(hash("быть", binary = binary)) shouldBe 1.0
    results(2).value(hash("question", binary = binary)) shouldBe 0.0
    // scalastyle:on
  }
}
