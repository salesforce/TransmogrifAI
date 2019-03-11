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
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.TestOpVectorColumnType.IndCol
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestOpWorkflowBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Estimator, Transformer}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpSetVectorizerTest extends FlatSpec with TestSparkContext with AttributeAsserts {

  val log = LoggerFactory.getLogger(this.getClass)

  val data = Seq(
    (Seq("a", "b"), Seq("x", "x")),
    (Seq("a", "b"), Seq("x", "x")),
    (Seq("a"), Seq("z", "y", "z", "z", "y")),
    (Seq("c"), Seq("x", "y")),
    (Seq("C ", "A."), Seq("Z", "Z", "Z"))
  )
  val expectedData = Array(
    Vectors.dense(1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    Vectors.dense(1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    Vectors.dense(1.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    Vectors.dense(0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
    Vectors.dense(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)
  ).map(_.toOPVector)

  val (dataSet, top, bot) = TestFeatureBuilder("top", "bot", data.map(v =>
    v._1.toMultiPickList -> v._2.toMultiPickList))
  val (dataSetEmpty, _, _) = TestFeatureBuilder(top.name, bot.name,
    Seq[(MultiPickList, MultiPickList)](
      (Seq("a", "b").toMultiPickList, MultiPickList.empty),
      (Seq("a", "b").toMultiPickList, MultiPickList.empty),
      (Seq("a").toMultiPickList, MultiPickList.empty),
      (MultiPickList.empty, MultiPickList.empty)
    )
  )

  val (dataSetAllEmpty, _) =
    TestFeatureBuilder(top.name, Seq[MultiPickList](MultiPickList.empty, MultiPickList.empty, MultiPickList.empty))

  val vectorizer = new OpSetVectorizer[MultiPickList]().setInput(top, bot).setMinSupport(0).setTopK(10)


  Spec[OpSetVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
    vector.originStage shouldBe vectorizer
    vector.parents should contain theSameElementsAs Array(top, bot)
  }

  it should "return the a fitted vectorizer with the correct parameters" in {
    val fitted = vectorizer.fit(dataSet)
    fitted.isInstanceOf[SequenceModel[_, _]]
    val vectorMetadata = fitted.getMetadata()
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      top -> List(IndCol(Some("A")), IndCol(Some("B")), IndCol(Some("C")), IndCol(Some("OTHER")),
        IndCol(Some(TransmogrifierDefaults.NullString))),
      bot -> List(IndCol(Some("X")), IndCol(Some("Y")), IndCol(Some("Z")), IndCol(Some("OTHER")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
    fitted.getInputFeatures() shouldBe Array(top, bot)
    fitted.parent shouldBe vectorizer
  }

  it should "return the expected vector with the default param settings" in {
    val fitted = vectorizer.fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(10, Array(0, 1, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 1, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 6, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(2, 5, 6), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 2, 7), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "not clean the variable names when clean text is set to false" in {
    val fitted = vectorizer.setCleanText(false).fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(13, Array(0, 1, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 1, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 8, 10), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(4, 7, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(2, 3, 9), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    val vectorMetadata = fitted.getMetadata()
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      top -> List(IndCol(Some("a")), IndCol(Some("b")), IndCol(Some("A.")), IndCol(Some("C ")), IndCol(Some("c")),
        IndCol(Some("OTHER")), IndCol(Some(TransmogrifierDefaults.NullString))),
      bot -> List(IndCol(Some("x")), IndCol(Some("y")), IndCol(Some("Z")), IndCol(Some("z")), IndCol(Some("OTHER")),
        IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "throw an error if you try to set the topK to 0 or a negative number" in {
    intercept[java.lang.IllegalArgumentException](vectorizer.setTopK(0))
    intercept[java.lang.IllegalArgumentException](vectorizer.setTopK(-1))
  }

  it should "return only the specified number of elements when top K is set" in {
    val fitted = vectorizer.setCleanText(true).setTopK(1).fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expectedData
    vectorizer.setTopK(10)
  }

  it should "return only elements that exceed the min support value" in {
    val fitted = vectorizer.setCleanText(true).setMinSupport(4).fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vector = vectorizer.getOutput()
    val result = transformed.collect(vector)
    transformed.collect(vector) shouldBe Array(
      Vectors.dense(1.0, 1.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 1.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 2.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 2.0, 0.0),
      Vectors.dense(1.0, 1.0, 0.0, 1.0, 0.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
  }

  it should "return a vector with elements only in the other & null columns and not throw errors when passed data" +
    " it was not trained with" in {
    val fitted = vectorizer.setMinSupport(0).setTopK(10).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.dense(1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(1.0, 1.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    val vectorMetadata = fitted.getMetadata()
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      top -> List(
        IndCol(Some("A")), IndCol(Some("B")), IndCol(Some("OTHER")), IndCol(Some(TransmogrifierDefaults.NullString))
      ),
      bot -> List(IndCol(Some("OTHER")), IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    println(OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata))
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
    val expected2 = Array(
      Vectors.dense(1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 2.0, 0.0),
      Vectors.dense(0.0, 0.0, 1.0, 0.0, 2.0, 0.0),
      Vectors.dense(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    ).map(_.toOPVector)
    val transformed2 = fitted.transform(dataSet)
    transformed2.collect(vector) shouldBe expected2
  }

  it should "return a vector with elements only in the other & null columns and not throw errors when passed data" +
    " it was not trained with, even when null tracking is disabled" in {
    val localVectorizer = new OpSetVectorizer[MultiPickList]().setInput(top, bot).setTopK(10).setTrackNulls(false)
      .setMinSupport(0)
    val fitted = localVectorizer.fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.dense(1.0, 1.0, 0.0, 0.0),
      Vectors.dense(1.0, 1.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    val vectorMetadata = fitted.getMetadata()
    val expectedMeta = TestOpVectorMetadataBuilder(
      localVectorizer,
      top -> List(IndCol(Some("A")), IndCol(Some("B")), IndCol(Some("OTHER"))),
      bot -> List(IndCol(Some("OTHER")))
    )
    OpVectorMetadata(localVectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
    val expected2 = Array(
      Vectors.dense(1.0, 1.0, 0.0, 1.0),
      Vectors.dense(1.0, 1.0, 0.0, 1.0),
      Vectors.dense(1.0, 0.0, 0.0, 2.0),
      Vectors.dense(0.0, 0.0, 1.0, 2.0),
      Vectors.dense(1.0, 0.0, 1.0, 1.0)
    ).map(_.toOPVector)
    val transformed2 = fitted.transform(dataSet)
    transformed2.collect(vector) shouldBe expected2
  }

  it should "work even if all features passed in are empty" in {
    val fitted = vectorizer.setInput(top).setTopK(10).fit(dataSetAllEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetAllEmpty)
    val expected = Array(Vectors.dense(0.0, 1.0), Vectors.dense(0.0, 1.0), Vectors.dense(0.0, 1.0)).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    val vectorMetadata = fitted.getMetadata()
    val expectedMeta = TestOpVectorMetadataBuilder(
      vectorizer,
      top -> List(IndCol(Some("OTHER")), IndCol(Some(TransmogrifierDefaults.NullString)))
    )
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual expectedMeta
  }

  it should "be implemented as 'pivot' shortcut" in {
    val result = top.pivot(others = Array(bot), topK = 1, cleanText = true, minSupport = 0, trackNulls = true)
    val df = result.originStage
      .asInstanceOf[Estimator[_]].fit(dataSet)
      .asInstanceOf[Transformer].transform(dataSet)

    result.originStage shouldBe a[OpSetVectorizer[_]]
    val actual = df.collect(result)
    actual shouldBe expectedData
    val field = df.schema(result.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, actual)
  }

  it should "expand number of columns for picklist features by two (one for other & one for null)" in {
    val (df, feature) = TestFeatureBuilder(List(
      "a", "b", "b", "a", "a", "b", "b", "c", "a"
    ).map(v => Set(v).toMultiPickList))

    val vectorized = feature.vectorize(topK = TransmogrifierDefaults.TopK, trackNulls = true, minSupport = 0,
      cleanText = TransmogrifierDefaults.CleanText)

    val untypedVectorizedStage = vectorized.originStage
    untypedVectorizedStage shouldBe a[OpSetVectorizer[_]]
    val inputDF = TestOpWorkflowBuilder(df, vectorized).computeDataUpTo(vectorized)
    val featArray = inputDF.collect(vectorized)
    featArray.foreach { opVec => opVec.value.size shouldBe 5 }
  }

  it should "process a single column of MultiPickLists" in {
    val localData = Seq(
      Seq("Alice", "Bob"),
      Seq("Bob"),
      Seq("Garth"),
      Seq.empty[String],
      Seq("Alice"),
      Seq("Alice"),
      Seq.empty[String],
      Seq("James"),
      Seq("Garth", "James")
    )
    val (localDataSet, f1) = TestFeatureBuilder("f1", localData.map(_.toMultiPickList))
    val localVectorizer = new OpSetVectorizer[MultiPickList]().setInput(f1).setTopK(3)

    val fitted = localVectorizer.fit(localDataSet)
    val transformed = fitted.transform(localDataSet)
    val vector = localVectorizer.getOutput()
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
  }

  it should "process multiple columns of PickList using the vectorize shortcut" in {
    val localData = Seq(
      (Some("A"), Some("Alice")),
      (Some("B"), Some("Bob")),
      (None, Some("Garth")),
      (None, None),
      (Some("A"), Some("Alice")),
      (Some("C"), Some("Alice")),
      (Some("C"), None),
      (Some("A"), Some("James")),
      (None, Some("Garth"))
    )
    val (localDF, f1, f2) = TestFeatureBuilder(localData.map(v => v._1.toPickList -> v._2.toPickList))
    val vectorized = Seq(f1, f2).transmogrify()

    val transformed = new OpWorkflow().setResultFeatures(vectorized).transform(localDF)
    val field = transformed.schema(vectorized.name)
    val result = transformed.collect(vectorized)
    assertNominal(field, Array.fill(result.head.value.toArray.length)(true), result)

    val metaMap = transformed.metadata(vectorized)
    log.info(metaMap.toString)
  }

  it should "process multiple columns of PickList and MultiPickLists using the vectorize shortcut" in {
    val localData = Seq(
      (Some("A"), Seq("Alice", "Bob")),
      (Some("B"), Seq("Bob")),
      (None, Seq("Garth")),
      (None, Seq.empty[String]),
      (Some("A"), Seq("Alice")),
      (Some("C"), Seq("Alice")),
      (Some("C"), Seq.empty[String]),
      (Some("A"), Seq("James")),
      (None, Seq("Garth", "James"))
    )
    val (localDF, f1, f2) = TestFeatureBuilder(localData.map(v => v._1.toPickList -> v._2.toMultiPickList))
    val vectorized = Seq(f1, f2).transmogrify()

    val transformed = new OpWorkflow().setResultFeatures(vectorized).transform(localDF)
    val field = transformed.schema(vectorized.name)
    val result = transformed.collect(vectorized)
    val expect = OpVectorMetadata("", field.metadata).columns
      .map(c => !(c.isOtherIndicator && c.parentFeatureType.head == FeatureType.typeName[MultiPickList]))
    assertNominal(field, expect, result)

    val metaMap = transformed.metadata(vectorized)
    log.info(metaMap.toString)
  }

  it should "process multiple columns of PickList and MultiPickLists using transformWith" in {
    val localData = Seq(
      (Some("A"), Seq("Alice", "Bob")),
      (Some("B"), Seq("Bob")),
      (None, Seq("Garth")),
      (None, Seq.empty[String]),
      (Some("A"), Seq("Alice")),
      (Some("C"), Seq("Alice")),
      (Some("C"), Seq.empty[String]),
      (Some("A"), Seq("James")),
      (None, Seq("Garth", "James"))
    )
    val (localDF, f1, f2) = TestFeatureBuilder(localData.map(v => v._1.toPickList -> v._2.toMultiPickList))

    val oPSetVectorizer = new OpSetVectorizer[MultiPickList]().setMinSupport(0)
    val res = f2.transformWith[OPVector](stage = oPSetVectorizer.setTopK(3), Array.empty[FeatureLike[MultiPickList]])

    val transformed = new OpWorkflow().setResultFeatures(res).transform(localDF)
    val field = transformed.schema(res.name)
    val result = transformed.collect(res)
    val expect = OpVectorMetadata("", field.metadata).columns
      .map(c => !(c.isOtherIndicator && c.parentFeatureType.head == FeatureType.typeName[MultiPickList]))
    assertNominal(field, expect, result)
  }

  it should "process multiple columns of numerics, PickLists, and MultiPickLists using the vectorize shortcut" in {
    val localData = Seq[(Option[Double], Option[String], Seq[String])](
      (None, Some("A"), Seq("Alice", "Bob")),
      (Some(41.0), Some("B"), Seq("Bob")),
      (Some(12.0), None, Seq("Garth")),
      (None, None, Seq.empty[String]),
      (Some(32.2), Some("A"), Seq("Alice")),
      (Some(87.6), Some("C"), Seq("Alice")),
      (Some(55.1), Some("C"), Seq.empty[String]),
      (None, Some("A"), Seq("James")),
      (Some(12.9), None, Seq("Garth", "James"))
    )
    val (localDF, f1, f2, f3) = TestFeatureBuilder(localData.map(v => (v._1.toReal, v._2.toPickList,
      v._3.toMultiPickList)))
    val vectorized = Seq(f1, f2, f3).transmogrify()

    val transformed = new OpWorkflow().setResultFeatures(vectorized).transform(localDF)
    val field = transformed.schema(vectorized.name)
    assertNominal(field, Array(false, true, true, true, false, true), transformed.collect(vectorized))

    val metaMap = transformed.metadata(vectorized)
    log.info(metaMap.toString)
  }
}
