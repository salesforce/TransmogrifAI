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
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class FilterMultiPickListMapTest extends OpTransformerSpec[MultiPickListMap, FilterMap[MultiPickListMap]] {
  val (inputData, f1Cat) = TestFeatureBuilder[MultiPickListMap](
    Seq(
      MultiPickListMap(Map("Arthur" -> Set("King", "Briton"),
        "Lancelot" -> Set("Brave", "Knight"),
        "Galahad" -> Set("Pure", "Knight"))),
      MultiPickListMap(Map("Lancelot" -> Set("Brave", "Knight"),
        "Galahad" -> Set("Pure", "Knight"),
        "Bedevere" -> Set("Wise", "Knight"))),
      MultiPickListMap(Map("Knight" -> Set("Ni", "Ekke Ekke Ekke Ekke Ptang Zoo Boing")))
    )
  )
  val transformer = new FilterMap[MultiPickListMap]().setInput(f1Cat)

  val expectedResult = Seq(
    MultiPickListMap(Map("Arthur" -> Set("King", "Briton"),
      "Lancelot" -> Set("Brave", "Knight"),
      "Galahad" -> Set("Pure", "Knight"))),
    MultiPickListMap(Map("Lancelot" -> Set("Brave", "Knight"),
      "Galahad" -> Set("Pure", "Knight"),
      "Bedevere" -> Set("Wise", "Knight"))),
    MultiPickListMap(Map("Knight" -> Set("Ni", "EkkeEkkeEkkeEkkePtangZooBoing")))
  )

  it should "filter allowlisted keys" in {
    transformer.setAllowListKeys(Array("Arthur", "Knight"))
    val filtered = transformer.transform(inputData).collect(transformer.getOutput())

    val dataExpected = Array(
      MultiPickListMap(Map("Arthur" -> Set("King", "Briton"))),
      MultiPickListMap.empty,
      MultiPickListMap(Map("Knight" -> Set("Ni", "EkkeEkkeEkkeEkkePtangZooBoing")))
    )

    filtered should contain theSameElementsAs dataExpected
  }

  it should "filter blocklisted keys" in {
    transformer
      .setAllowListKeys(Array[String]())
      .setDenyListKeys(Array("Arthur", "Knight"))

    val filtered = transformer.transform(inputData).collect(transformer.getOutput())

    val dataExpected = Array(
      MultiPickListMap(Map("Lancelot" -> Set("Brave", "Knight"),
        "Galahad" -> Set("Pure", "Knight"))),
      MultiPickListMap(Map("Lancelot" -> Set("Brave", "Knight"),
        "Galahad" -> Set("Pure", "Knight"),
        "Bedevere" -> Set("Wise", "Knight"))),
      MultiPickListMap.empty
    )

    filtered should contain theSameElementsAs dataExpected
  }

  it should "not clean map when flag set to false" in {
    transformer
      .setCleanText(false)
      .setCleanKeys(false)
      .setAllowListKeys(Array("Arthur", "Knight"))
      .setDenyListKeys(Array())
    val filtered = transformer.transform(inputData).collect(transformer.getOutput())

    val dataExpected = Array(
      MultiPickListMap(Map("Arthur" -> Set("King", "Briton"))),
      MultiPickListMap.empty,
      MultiPickListMap(Map("Knight" -> Set("Ni", "Ekke Ekke Ekke Ekke Ptang Zoo Boing")))
    )
    filtered should contain theSameElementsAs dataExpected
  }

  it should "clean map when flag set to true" in {
    transformer
      .setCleanKeys(true)
      .setCleanText(true)
      .setAllowListKeys(Array("Arthur", "Knight"))
      .setDenyListKeys(Array())
    val filtered = transformer.transform(inputData).collect(transformer.getOutput())

    val dataExpected = Array(
      MultiPickListMap(Map("Arthur" -> Set("King", "Briton"))),
      MultiPickListMap.empty,
      MultiPickListMap(Map("Knight" -> Set("Ni", "EkkeEkkeEkkeEkkePtangZooBoing")))
    )
    filtered should contain theSameElementsAs dataExpected
  }

}
