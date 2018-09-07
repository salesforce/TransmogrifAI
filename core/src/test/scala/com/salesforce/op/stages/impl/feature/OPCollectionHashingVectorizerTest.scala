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
import com.salesforce.op.test.TestOpVectorColumnType.{IndCol, PivotColNoInd}
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichMetadata._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class OPCollectionHashingVectorizerTest extends FlatSpec with TestSparkContext {

  val (catData, top, bot) = TestFeatureBuilder("top", "bot",
    Seq[(MultiPickList, MultiPickList)](
      (Seq("a", "b").toMultiPickList, Seq("x", "x").toMultiPickList),
      (Seq("a").toMultiPickList, Seq("z", "y", "z", "z", "y").toMultiPickList),
      (Seq("c", "x").toMultiPickList, Seq("x", "y").toMultiPickList),
      (Seq("C ", "A.").toMultiPickList, Seq("Z", "Z", "Z").toMultiPickList)
    )
  )
  val (textListData, textList1, textList2) = TestFeatureBuilder("textList1", "textList2",
    Seq[(TextList, TextList)](
      (Seq("a", "b").toTextList, Seq("x", "x").toTextList),
      (Seq("a").toTextList, Seq("z", "y", "z", "z", "y").toTextList),
      (Seq("c", "x").toTextList, Seq("x", "y").toTextList),
      (Seq("C ", "A.").toTextList, Seq("Z", "Z", "Z").toTextList),
      (TextList.empty, TextList.empty)
    )
  )
  val (realMapData, rm1, rm2) = TestFeatureBuilder("rm1", "rm2",
    Seq[(RealMap, RealMap)](
      (Map("a" -> 1.0, "b" -> 5.0).toRealMap, Map("z" -> 10.0).toRealMap),
      (Map("c" -> 11.0).toRealMap, Map("y" -> 3.0, "x" -> 0.0).toRealMap),
      (Map.empty[String, Double].toRealMap, Map.empty[String, Double].toRealMap)
    )
  )

  Spec[OPCollectionHashingVectorizer[_]] should "have the correct number of features" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
    vectorizer.setNumFeatures(128).getNumFeatures() shouldBe 128
  }

  it should "take an array of features as input and return a single vector feature" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot).setNumFeatures(128)
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
    val vectorMetadata = vectorizer.getMetadata()

    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      top -> (0 to 127).map(i => PivotColNoInd(i.toString)).toList,
      bot -> (0 to 127).map(i => PivotColNoInd(i.toString)).toList
    )
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "validate the params correctly" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList]()

    assertThrows[IllegalArgumentException](vectorizer.setNumFeatures(-1))
    assertThrows[IllegalArgumentException](vectorizer.setNumFeatures(0))
    assertThrows[IllegalArgumentException](vectorizer.setNumFeatures(Integer.MAX_VALUE))
    assertThrows[IllegalArgumentException](vectorizer.setNumFeatures(Integer.MIN_VALUE))
  }

  it should "be able to vectorize several columns of MultiPickList features" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(catData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe false

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    val topNameHash = hasher.indexOf(top.name)
    val botNameHash = hasher.indexOf(bot.name)
    result(0).value(hasher.indexOf(topNameHash + "_" + "a")) shouldBe 1.0
    result(0).value(hasher.indexOf(topNameHash + "_" + "c")) shouldBe 0.0
    result(0).value(hasher.indexOf(botNameHash + "_" + "x") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(0).value(hasher.indexOf(botNameHash + "_" + "y") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(1).value(hasher.indexOf(topNameHash + "_" + "a")) shouldBe 1.0
    result(1).value(hasher.indexOf(topNameHash + "_" + "b")) shouldBe 0.0
    result(1).value(hasher.indexOf(botNameHash + "_" + "x") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(1).value(hasher.indexOf(botNameHash + "_" + "z") + vectorizer.getNumFeatures()) shouldBe 1.0

    result(2).value(hasher.indexOf(topNameHash + "_" + "b")) shouldBe 0.0
    result(2).value(hasher.indexOf(topNameHash + "_" + "c")) shouldBe 1.0
    result(2).value(hasher.indexOf(botNameHash + "_" + "y") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(2).value(hasher.indexOf(botNameHash + "_" + "z") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(3).value(hasher.indexOf(topNameHash + "_" + "a")) shouldBe 0.0
    result(3).value(hasher.indexOf(botNameHash + "_" + "z") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(3).value(hasher.indexOf(botNameHash + "_" + "Z") + vectorizer.getNumFeatures()) shouldBe 1.0
  }

  it should "be able to vectorize several columns of MultiPickList data using separate hash spaces" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
        .setNumFeatures(256).setPrependFeatureName(false)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(catData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe false

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    result(0).value(hasher.indexOf("a")) shouldBe 1.0
    result(0).value(hasher.indexOf("c")) shouldBe 0.0
    result(0).value(hasher.indexOf("x") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(0).value(hasher.indexOf("y") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(1).value(hasher.indexOf("a")) shouldBe 1.0
    result(1).value(hasher.indexOf("b")) shouldBe 0.0
    result(1).value(hasher.indexOf("x") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(1).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 1.0

    result(2).value(hasher.indexOf("b")) shouldBe 0.0
    result(2).value(hasher.indexOf("c")) shouldBe 1.0
    result(2).value(hasher.indexOf("y") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(2).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(3).value(hasher.indexOf("a")) shouldBe 0.0
    result(3).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(3).value(hasher.indexOf("Z") + vectorizer.getNumFeatures()) shouldBe 1.0
  }

  /**
   * Hashing a TextList column will hash the indices with the values, eg. the first element of the list [1,2,3]
   * will be hashed as the string (1,0)
   */
  it should "be able to vectorize several columns of TextList data using separate hash spaces" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[TextList].setInput(textList1, textList2)
        .setNumFeatures(256).setPrependFeatureName(false).setHashWithIndex(true)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(textListData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe false

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    result(0).value(hasher.indexOf("(a,0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(x,1)") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(0).value(hasher.indexOf("(z,0)") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(1).value(hasher.indexOf("(a,0)")) shouldBe 1.0
    result(1).value(hasher.indexOf("(b,1)")) shouldBe 0.0
    result(1).value(hasher.indexOf("(y,4)") + vectorizer.getNumFeatures()) shouldBe 1.0

    result(2).value(hasher.indexOf("(c,0)")) shouldBe 1.0
    result(2).value(hasher.indexOf("(y,1)") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(2).value(hasher.indexOf("(z,0)") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(3).value(hasher.indexOf("(b,1)")) shouldBe 0.0
    result(3).value(hasher.indexOf("(z,0)") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(3).value(hasher.indexOf("(Z,2)") + vectorizer.getNumFeatures()) shouldBe 1.0
  }

  it should "be able to vectorize several columns of RealMap data using separate hash spaces" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[RealMap].setInput(rm1, rm2)
        .setNumFeatures(256).setPrependFeatureName(false).setHashWithIndex(true)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(realMapData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(false))

    vectorizer.isSharedHashSpace shouldBe false

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    result(0).value(hasher.indexOf("(a,1.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(b,5.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(z,10.0)") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(0).value(hasher.indexOf("(c,10.0)") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(1).value(hasher.indexOf("(c,11.0)")) shouldBe 1.0
    result(1).value(hasher.indexOf("(b,1.0)")) shouldBe 0.0
    result(1).value(hasher.indexOf("(x,0.0)") + vectorizer.getNumFeatures()) shouldBe 1.0

    result(2).value(hasher.indexOf("(c,0)")) shouldBe 0.0
    result(2).value(hasher.indexOf("(z,0)") + vectorizer.getNumFeatures()) shouldBe 0.0
  }

  it should "be able to vectorize several columns of MultiPickList data using a shared hash space" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
      .setNumFeatures((TransmogrifierDefaults.MaxNumOfFeatures / 2) + 1).setPrependFeatureName(false)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(catData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe true

    // When checking result in a shared hash space, no offset is needed
    result(0).value(hasher.indexOf("a")) shouldBe 1.0
    result(0).value(hasher.indexOf("c")) shouldBe 0.0
    result(0).value(hasher.indexOf("x")) shouldBe 1.0
    result(0).value(hasher.indexOf("y")) shouldBe 0.0

    result(1).value(hasher.indexOf("a")) shouldBe 1.0
    result(1).value(hasher.indexOf("b")) shouldBe 0.0
    result(1).value(hasher.indexOf("x")) shouldBe 0.0
    result(1).value(hasher.indexOf("z")) shouldBe 1.0

    result(2).value(hasher.indexOf("b")) shouldBe 0.0
    result(2).value(hasher.indexOf("c")) shouldBe 1.0
    result(2).value(hasher.indexOf("y")) shouldBe 1.0
    result(2).value(hasher.indexOf("z")) shouldBe 0.0

    result(3).value(hasher.indexOf("a")) shouldBe 0.0
    result(3).value(hasher.indexOf("z")) shouldBe 0.0
    result(3).value(hasher.indexOf("Z")) shouldBe 1.0
  }

  it should "be able to vectorize several columns of TextList data using a shared hash space" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[TextList].setInput(textList1, textList2)
        .setNumFeatures((TransmogrifierDefaults.MaxNumOfFeatures / 2) + 1)
        .setPrependFeatureName(false).setHashWithIndex(true)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(textListData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe true

    // When checking result in a shared hash space, no offset is needed
    result(0).value(hasher.indexOf("(a,0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(x,1)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(z,0)")) shouldBe 0.0

    result(1).value(hasher.indexOf("(a,0)")) shouldBe 1.0
    result(1).value(hasher.indexOf("(b,1)")) shouldBe 0.0
    result(1).value(hasher.indexOf("(y,4)")) shouldBe 1.0

    result(2).value(hasher.indexOf("(c,0)")) shouldBe 1.0
    result(2).value(hasher.indexOf("(y,1)")) shouldBe 1.0
    result(2).value(hasher.indexOf("(z,0)")) shouldBe 0.0

    result(3).value(hasher.indexOf("(b,1)")) shouldBe 0.0
    result(3).value(hasher.indexOf("(z,0)")) shouldBe 0.0
    result(3).value(hasher.indexOf("(Z,2)")) shouldBe 1.0
  }

  it should "be able to vectorize several columns of RealMap data using a shared hash spaces" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[RealMap].setInput(rm1, rm2)
        .setNumFeatures((TransmogrifierDefaults.MaxNumOfFeatures / 2) + 1)
        .setPrependFeatureName(false).setHashWithIndex(true)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(realMapData)
    val result = transformed.collect(vector)
    // TODO : Find a way to recognize hashed RealMap has Categoricals

    vectorizer.isSharedHashSpace shouldBe true

    // When checking result in a shared hash space, no offset is needed
    result(0).value(hasher.indexOf("(a,1.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(b,5.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(z,10.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(c,10.0)")) shouldBe 0.0

    result(1).value(hasher.indexOf("(c,11.0)")) shouldBe 1.0
    result(1).value(hasher.indexOf("(b,1.0)")) shouldBe 0.0
    result(1).value(hasher.indexOf("(x,0.0)")) shouldBe 1.0

    result(2).value(hasher.indexOf("(c,0)")) shouldBe 0.0
    result(2).value(hasher.indexOf("(z,0)")) shouldBe 0.0
  }

  it should "force a shared hash space when requested" in {
    val vectorizer = new OPCollectionHashingVectorizer[RealMap].setInput(rm1, rm2)
      .setNumFeatures(256).setHashSpaceStrategy(HashSpaceStrategy.Shared)
      .setPrependFeatureName(false).setHashWithIndex(true)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(realMapData)
    val result = transformed.collect(vector)
    // TODO : Find a way to recognize hashed RealMap has Categoricals

    vectorizer.isSharedHashSpace shouldBe true

    // When checking result in a shared hash space, no offset is needed
    result(0).value(hasher.indexOf("(a,1.0)")) shouldBe 1.0
    result(0).value(hasher.indexOf("(c,10.0)")) shouldBe 0.0

    result(1).value(hasher.indexOf("(b,1.0)")) shouldBe 0.0
    result(1).value(hasher.indexOf("(x,0.0)")) shouldBe 1.0

    result(2).value(hasher.indexOf("(c,0)")) shouldBe 0.0
    result(2).value(hasher.indexOf("(z,0)")) shouldBe 0.0
  }

  it should "hash lists without indices when requested" in {
    val vectorizer =
      new OPCollectionHashingVectorizer[TextList].setInput(textList1, textList2)
        .setNumFeatures(256).setHashWithIndex(false).setPrependFeatureName(false)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(textListData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe false

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    result(0).value(hasher.indexOf("a")) shouldBe 1.0
    result(0).value(hasher.indexOf("x") + vectorizer.getNumFeatures()) shouldBe 2.0
    result(0).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(1).value(hasher.indexOf("b")) shouldBe 0.0
    result(1).value(hasher.indexOf("y") + vectorizer.getNumFeatures()) shouldBe 2.0
    result(1).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 3.0

    result(2).value(hasher.indexOf("c")) shouldBe 1.0
    result(2).value(hasher.indexOf("y") + vectorizer.getNumFeatures()) shouldBe 1.0
    result(2).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 0.0

    result(3).value(hasher.indexOf("b")) shouldBe 0.0
    result(3).value(hasher.indexOf("z") + vectorizer.getNumFeatures()) shouldBe 0.0
    result(3).value(hasher.indexOf("Z") + vectorizer.getNumFeatures()) shouldBe 3.0
  }

  it should "preserve binarity (if requested), even in shared hash spaces" in {
    val vectorizer = new OPCollectionHashingVectorizer[TextList].setInput(textList1, textList2)
      .setNumFeatures(256).setHashWithIndex(false).setHashSpaceStrategy(HashSpaceStrategy.Shared)
      .setBinaryFreq(true).setPrependFeatureName(false)
    val hasher = vectorizer.hashingTF()
    val vector = vectorizer.getOutput()
    val transformed = vectorizer.transform(textListData)
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))

    vectorizer.isSharedHashSpace shouldBe true

    // When checking result in separate hash spaces, make sure to add in the vectorizer.getNumFeatures() offset
    // to get to the hash space of the second column
    result(0).value(hasher.indexOf("a")) shouldBe 1.0
    result(0).value(hasher.indexOf("x")) shouldBe 1.0
    result(0).value(hasher.indexOf("z")) shouldBe 0.0

    result(1).value(hasher.indexOf("b")) shouldBe 0.0
    result(1).value(hasher.indexOf("y")) shouldBe 1.0
    result(1).value(hasher.indexOf("z")) shouldBe 1.0

    result(2).value(hasher.indexOf("x")) shouldBe 1.0
    result(2).value(hasher.indexOf("y")) shouldBe 1.0
    result(2).value(hasher.indexOf("z")) shouldBe 0.0

    result(3).value(hasher.indexOf("b")) shouldBe 0.0
    result(3).value(hasher.indexOf("z")) shouldBe 0.0
    result(3).value(hasher.indexOf("Z")) shouldBe 1.0
  }

  it should "make the correct metadata with separate hash spaces" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
      .setNumFeatures(10).setHashSpaceStrategy(HashSpaceStrategy.Separate)
    val feature = vectorizer.getOutput()
    val transformed = vectorizer.transform(catData)
    val vector = vectorizer.getOutput()
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))
    val meta = OpVectorMetadata(transformed.schema(feature.name))
    meta.history.keys shouldBe Set(top.name, bot.name)
    meta.columns.length shouldBe 20
    meta.columns.foreach(
      c => c.parentFeatureName == Seq(top.name) || c.parentFeatureName == Seq(bot.name) shouldBe true
    )
    meta.getColumnHistory().length shouldBe 20
  }

  it should "make the correct metadata with shared hash spaces" in {
    val vectorizer = new OPCollectionHashingVectorizer[MultiPickList].setInput(top, bot)
      .setNumFeatures(10).setHashSpaceStrategy(HashSpaceStrategy.Shared)
    val feature = vectorizer.getOutput()
    val transformed = vectorizer.transform(catData)
    val vector = vectorizer.getOutput()
    val field = transformed.schema(vector.name)
    AttributeTestUtils.assertNominal(field, Array.fill(transformed.collect(vector).head.value.size)(true))
    val meta = OpVectorMetadata(transformed.schema(feature.name))
    meta.history.keys shouldBe Set(top.name, bot.name)
    meta.columns.length shouldBe 10
    meta.columns.foreach(
      c => c.parentFeatureName == Seq(top.name, bot.name) shouldBe true
    )
    meta.getColumnHistory().length shouldBe 10
  }

}
