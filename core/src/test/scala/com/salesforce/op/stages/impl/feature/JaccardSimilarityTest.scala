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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class JaccardSimilarityTest extends OpTransformerSpec[RealNN, JaccardSimilarity] {

  val (inputData, f1, f2) = TestFeatureBuilder(
    Seq(
      (Seq("Red", "Green"), Seq("Red")),
      (Seq("Red", "Green"), Seq("Yellow, Blue")),
      (Seq("Red", "Yellow"), Seq("Red", "Yellow"))
    ).map(v => v._1.toMultiPickList -> v._2.toMultiPickList)
  )

  val transformer = new JaccardSimilarity().setInput(f1, f2)

  val expectedResult: Seq[RealNN] = Seq(0.5, 0.0, 1.0).toRealNN

  it should "have a shortcut" in {
    val jaccard = f1.jaccardSimilarity(f2)

    jaccard.name shouldBe jaccard.originStage.getOutputFeatureName
    jaccard.parents shouldBe Array(f1, f2)
    jaccard.originStage shouldBe a[JaccardSimilarity]
  }
  it should "return 1 when both vectors are empty" in {
    val set1 = Seq.empty[String].toMultiPickList
    val set2 = Seq.empty[String].toMultiPickList
    transformer.transformFn(set1, set2) shouldBe 1.0.toRealNN
  }
  it should "return 1 when both vectors are the same" in {
    val set1 = Seq("Red", "Blue", "Green").toMultiPickList
    val set2 = Seq("Red", "Blue", "Green").toMultiPickList
    transformer.transformFn(set1, set2) shouldBe 1.0.toRealNN
  }

  it should "calculate similarity correctly when vectors are different" in {
    val set1 = Seq("Red", "Green", "Blue").toMultiPickList
    val set2 = Seq("Red", "Blue").toMultiPickList
    transformer.transformFn(set1, set2) shouldBe (2.0 / 3.0).toRealNN

    val set3 = Seq("Red").toMultiPickList
    val set4 = Seq("Blue").toMultiPickList
    transformer.transformFn(set3, set4) shouldBe 0.0.toRealNN

    val set5 = Seq("Red", "Yellow", "Green").toMultiPickList
    val set6 = Seq("Pink", "Green", "Blue").toMultiPickList
    transformer.transformFn(set5, set6) shouldBe (1.0 / 5.0).toRealNN
  }
}
