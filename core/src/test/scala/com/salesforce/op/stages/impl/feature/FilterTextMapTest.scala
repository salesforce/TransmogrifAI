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
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class FilterTextMapTest extends OpTransformerSpec[TextMap, FilterMap[TextMap]] {
  val (inputData, f1) = TestFeatureBuilder[TextMap](
    Seq(
      TextMap(Map("Arthur" -> "King", "Lancelot" -> "Brave", "Galahad" -> "Pure")),
      TextMap(Map("Lancelot" -> "Brave", "Galahad" -> "Pure", "Bedevere" -> "Wise")),
      TextMap(Map("Knight" -> "Ni"))
    )
  )

  val transformer = new FilterMap[TextMap]().setInput(f1)

  val expectedResult: Seq[TextMap] = Array(
    TextMap(Map("Arthur" -> "King", "Lancelot" -> "Brave", "Galahad" -> "Pure")),
    TextMap(Map("Lancelot" -> "Brave", "Galahad" -> "Pure", "Bedevere" -> "Wise")),
    TextMap(Map("Knight" -> "Ni"))
  )

  it should "filter whitelisted keys" in {
    transformer.setWhiteListKeys(Array("Arthur", "Knight"))

    val filtered = transformer.transform(inputData).collect(transformer.getOutput)
    val dataExpected = Array(
      TextMap(Map("Arthur" -> "King")),
      TextMap.empty,
      TextMap(Map("Knight" -> "Ni"))
    )

    filtered should contain theSameElementsAs dataExpected
  }

  it should "filter blacklisted keys" in {
    transformer.setInput(f1)
      .setWhiteListKeys(Array[String]())
      .setBlackListKeys(Array("Arthur", "Knight"))
    val filtered = transformer.transform(inputData).collect(transformer.getOutput)

    val dataExpected = Array(
      TextMap(Map("Lancelot" -> "Brave", "Galahad" -> "Pure")),
      TextMap(Map("Lancelot" -> "Brave", "Galahad" -> "Pure", "Bedevere" -> "Wise")),
      TextMap.empty
    )

    filtered should contain theSameElementsAs dataExpected
  }

  it should "set cleanMapFlag correctly" in {
    transformer.setCleanText(false)
    transformer.get[Boolean](transformer.cleanText).get shouldBe false
    transformer.setCleanKeys(false)
    transformer.get[Boolean](transformer.cleanKeys).get shouldBe false
  }

  it should "filter correctly when using shortcut" in {
    val filtered = f1.filter(whiteList = Seq("Arthur", "Knight"), blackList = Seq())

    filtered.name shouldBe filtered.originStage.getOutputFeatureName
    filtered.originStage shouldBe a[FilterMap[_]]
    filtered.parents shouldBe Array(f1)
  }
}
