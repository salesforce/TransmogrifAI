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
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.TestOpVectorColumnType.IndColWithGroup
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class MultiPickListMapVectorizerTest
  extends OpEstimatorSpec[OPVector,
    SequenceModel[MultiPickListMap, OPVector],
    MultiPickListMapVectorizer[MultiPickListMap]]
  with AttributeAsserts {

  val expectedResult = Seq(
    Vectors.sparse(20, Array(2, 3, 7, 10, 15, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    Vectors.sparse(20, Array(2, 4, 9, 12, 13, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    Vectors.sparse(20, Array(0, 6, 9, 10, 13, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    Vectors.sparse(20, Array(0, 3, 9, 12, 15, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
  ).map(_.toOPVector)

  val log = LoggerFactory.getLogger(this.getClass)

  lazy val (inputData, top, bot) = TestFeatureBuilder("top", "bot",
    Seq(
      (Map("a" -> Set("d"), "b" -> Set("d")), Map("x" -> Set("W"))),
      (Map("a" -> Set("e")), Map("z" -> Set("w"), "y" -> Set("v"))),
      (Map("c" -> Set("D")), Map("x" -> Set("w"), "y" -> Set("V"))),
      (Map("c" -> Set("d"), "a" -> Set("d")), Map("z" -> Set("v")))
    ).map(v => v._1.toMultiPickListMap -> v._2.toMultiPickListMap)
  )

  lazy val (dataSetEmpty, _, _) = TestFeatureBuilder(top.name, bot.name,
    Seq(
      (Map("a" -> Set("d"), "b" -> Set("d")), Map[String, Set[String]]()),
      (Map("a" -> Set("e")), Map[String, Set[String]]()),
      (Map[String, Set[String]](), Map[String, Set[String]]())
    ).map(v => v._1.toMultiPickListMap -> v._2.toMultiPickListMap)
  )

  lazy val (dataSetAllEmpty, _) = TestFeatureBuilder(top.name,
    Seq(MultiPickListMap.empty, MultiPickListMap.empty, MultiPickListMap.empty))

  val estimator: MultiPickListMapVectorizer[MultiPickListMap] = new MultiPickListMapVectorizer()
    .setCleanKeys(true).setMinSupport(0).setTopK(10).setInput(top, bot)

  lazy val (ds, tech, cnty) = TestFeatureBuilder(
    Seq(
      (Map("tech" -> Set("Spark", "Scala")), Map("cnty" -> Set("Canada", "US"))),
      (Map("tech" -> Set("        sPaRk   ", "Python", "Torch   ")), Map("cnty" -> Set("France ", "UK           "))),
      (Map("tech" -> Set("R", "Hive")), Map("cnty" -> Set("Germany"))),
      (Map("tech" -> Set("python", "TenSoRflow", " TorcH ")), Map("cnty" -> Set("france", "UK", "US")))
    ).map(v => v._1.toMultiPickListMap -> v._2.toMultiPickListMap)
  )

  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)

  it should "take an array of features as input and return a single vector feature" in {
    val vector = estimator.getOutput()
    vector.name shouldBe estimator.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return the a fitted vectorizer with the correct default parameters" in {
    val fitted = estimator.setTrackNulls(false).fit(inputData)
    fitted shouldBe a[SequenceModel[_, _]]
    val transformed = fitted.transform(inputData)
    val vector = estimator.getOutput()
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    val vectorMetadata = fitted.getMetadata()
    val meta = OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata)
    val expect = meta.columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    meta shouldEqual
      TestOpVectorMetadataBuilder(estimator,
        top -> List(
          IndColWithGroup(Some("D"), "C"), IndColWithGroup(Some("OTHER"), "C"), IndColWithGroup(Some("D"), "A"),
          IndColWithGroup(Some("E"), "A"), IndColWithGroup(Some("OTHER"), "A"),
          IndColWithGroup(Some("D"), "B"), IndColWithGroup(Some("OTHER"), "B")
        ),
        bot -> List(
          IndColWithGroup(Some("W"), "X"), IndColWithGroup(Some("OTHER"), "X"), IndColWithGroup(Some("V"), "Y"),
          IndColWithGroup(Some("OTHER"), "Y"), IndColWithGroup(Some("V"), "Z"),
          IndColWithGroup(Some("W"), "Z"), IndColWithGroup(Some("OTHER"), "Z")
        )
      )
    fitted.getInputFeatures() shouldBe Array(top, bot)
    fitted.parent shouldBe estimator
  }

  it should "track nulls with the correct default parameters" in {
    val fitted = estimator.setTrackNulls(true).fit(inputData)
    fitted shouldBe a[SequenceModel[_, _]]
    val transformed = fitted.transform(inputData)
    val vector = estimator.getOutput()
    val result = transformed.collect(vector)
    val field = transformed.schema(vector.name)
    val vectorMetadata = fitted.getMetadata()
    val meta = OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata)
    val expect = meta.columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    meta shouldEqual
      TestOpVectorMetadataBuilder(estimator,
        top -> List(
          IndColWithGroup(Some("D"), "C"), IndColWithGroup(Some("OTHER"), "C"),
          IndColWithGroup(nullIndicatorValue, "C"),
          IndColWithGroup(Some("D"), "A"), IndColWithGroup(Some("E"), "A"), IndColWithGroup(Some("OTHER"), "A"),
          IndColWithGroup(nullIndicatorValue, "A"),
          IndColWithGroup(Some("D"), "B"), IndColWithGroup(Some("OTHER"), "B"),
          IndColWithGroup(nullIndicatorValue, "B")
        ),
        bot -> List(
          IndColWithGroup(Some("W"), "X"), IndColWithGroup(Some("OTHER"), "X"),
          IndColWithGroup(nullIndicatorValue, "X"),
          IndColWithGroup(Some("V"), "Y"), IndColWithGroup(Some("OTHER"), "Y"),
          IndColWithGroup(nullIndicatorValue, "Y"),
          IndColWithGroup(Some("V"), "Z"), IndColWithGroup(Some("W"), "Z"), IndColWithGroup(Some("OTHER"), "Z"),
          IndColWithGroup(nullIndicatorValue, "Z")
        )
      )
    fitted.getInputFeatures() shouldBe Array(top, bot)
    fitted.parent shouldBe estimator
  }

  it should "return the expected vector with the default param settings" in {
    val fitted = estimator.setTrackNulls(false).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val result = transformed.collect(vector)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Seq(
      Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "track nulls with the default param settings" in {
    val fitted = estimator.setTrackNulls(true).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val result = transformed.collect(vector)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expectedResult
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "not clean the variable names when clean text is set to false" in {
    val fitted = estimator.setTrackNulls(false).setCleanText(false).setCleanKeys(false).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val result = transformed.collect(vector)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(17, Array(3, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(4, 12, 15), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(0, 9, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(1, 3, 14), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(estimator,
        top -> List(
          IndColWithGroup(Some("D"), "c"), IndColWithGroup(Some("d"), "c"), IndColWithGroup(Some("OTHER"), "c"),
          IndColWithGroup(Some("d"), "a"), IndColWithGroup(Some("e"), "a"),
          IndColWithGroup(Some("OTHER"), "a"), IndColWithGroup(Some("d"), "b"), IndColWithGroup(Some("OTHER"), "b")
        ),
        bot -> List(
          IndColWithGroup(Some("W"), "x"), IndColWithGroup(Some("w"), "x"), IndColWithGroup(Some("OTHER"), "x"),
          IndColWithGroup(Some("V"), "y"), IndColWithGroup(Some("v"), "y"),
          IndColWithGroup(Some("OTHER"), "y"), IndColWithGroup(Some("v"), "z"), IndColWithGroup(Some("w"), "z"),
          IndColWithGroup(Some("OTHER"), "z")
        )
      )
  }

  it should "track nulls when clean text is set to false" in {
    val fitted = estimator.setTrackNulls(true).setCleanText(false).setCleanKeys(false).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(23, Array(3, 4, 8, 11, 18, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(3, 5, 10, 14, 16, 20), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(0, 7, 10, 12, 15, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(1, 4, 10, 14, 18, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
    OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(estimator,
        top -> List(
          IndColWithGroup(Some("D"), "c"), IndColWithGroup(Some("d"), "c"), IndColWithGroup(Some("OTHER"), "c"),
          IndColWithGroup(nullIndicatorValue, "c"), IndColWithGroup(Some("d"), "a"), IndColWithGroup(Some("e"), "a"),
          IndColWithGroup(Some("OTHER"), "a"), IndColWithGroup(nullIndicatorValue, "a"),
          IndColWithGroup(Some("d"), "b"), IndColWithGroup(Some("OTHER"), "b"), IndColWithGroup(nullIndicatorValue, "b")
        ),
        bot -> List(
          IndColWithGroup(Some("W"), "x"), IndColWithGroup(Some("w"), "x"), IndColWithGroup(Some("OTHER"), "x"),
          IndColWithGroup(nullIndicatorValue, "x"), IndColWithGroup(Some("V"), "y"), IndColWithGroup(Some("v"), "y"),
          IndColWithGroup(Some("OTHER"), "y"), IndColWithGroup(nullIndicatorValue, "y"),
          IndColWithGroup(Some("v"), "z"), IndColWithGroup(Some("w"), "z"),
          IndColWithGroup(Some("OTHER"), "z"), IndColWithGroup(nullIndicatorValue, "z")
        )
      )
  }

  it should "return only the specified number of elements when top K is set" in {
    val fitted = estimator.setTrackNulls(false).setCleanText(true).setTopK(1).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(12, Array(2, 4, 6), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 8, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 2, 10), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "track nulls when top K is set" in {
    val fitted = estimator.setTrackNulls(true).setCleanText(true).setTopK(1).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(18, Array(2, 3, 6, 9, 14, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(2, 4, 8, 11, 12, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 5, 8, 9, 12, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 3, 8, 11, 14, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "return only the elements that exceed the minimum support requirement when minSupport is set" in {
    val fitted = estimator.setTrackNulls(false).setCleanText(true).setTopK(10).setMinSupport(2).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(10, Array(2, 4, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(3, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 2, 9), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "track nulls when minSupport is set" in {
    val fitted = estimator.setTrackNulls(true).setCleanText(true).setTopK(10).setMinSupport(2).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.sparse(16, Array(2, 3, 6, 8, 13, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(2, 4, 7, 10, 11, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 5, 7, 8, 11, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 3, 7, 10, 13, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "behave correctly when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = estimator.setTrackNulls(false).setCleanText(true).setMinSupport(0).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)

    result shouldBe expected

    val transformed2 = fitted.transform(inputData)
    val result2 = transformed2.collect(vector)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field2 = transformed2.schema(vector.name)
    val expect2 = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field2, expect2, result2)
    result2 shouldBe expected2
  }

  it should "track nulls when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = estimator.setTrackNulls(true).setCleanText(true).setMinSupport(0).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)

    result shouldBe expected

    val transformed2 = fitted.transform(inputData)
    val result2 = transformed2.collect(vector)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)
    val field2 = transformed2.schema(vector.name)
    val expect2 = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field2, expect2, result2)
    result2 shouldBe expected2
  }

  it should "behave correctly when passed only empty maps" in {
    val fitted = estimator.setInput(top).setTrackNulls(false).setCleanText(true).setTopK(10).fit(dataSetAllEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetAllEmpty)
    val result = transformed.collect(vector)
    val expected = Array(
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double])
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "correctly allowlist keys" in {
    val fitted = estimator.setTrackNulls(false).setInput(top, bot).setTopK(10).setAllowListKeys(Array("a", "x"))
      .fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(5, Array(0, 3), Array(1.0, 1.0)),
      Vectors.sparse(5, Array(1), Array(1.0)),
      Vectors.sparse(5, Array(3), Array(1.0)),
      Vectors.sparse(5, Array(0), Array(1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "track nulls with allowlist keys" in {
    val fitted = estimator.setTrackNulls(true).setInput(top, bot).setTopK(10).setAllowListKeys(Array("a", "x"))
      .fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(7, Array(0, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(3, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(0, 6), Array(1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "correctly blocklist keys" in {
    val fitted = estimator.setAllowListKeys(Array()).setTrackNulls(false)
      .setBlockListKeys(Array("a", "x")).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(9, Array(2), Array(1.0)),
      Vectors.sparse(9, Array(5, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 4), Array(1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "track nulls with blocklist keys" in {
    val fitted = estimator.setAllowListKeys(Array()).setTrackNulls(true)
      .setBlockListKeys(Array("a", "x")).fit(inputData)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)
    printRes(transformed, vectorMetadata, estimator.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(13, Array(2, 3, 9, 12), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(2, 5, 7, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 9, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 6, 12), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "correctly handle MultiPickList with multiple elements (top 3)" in {
    val kcVectorizer = new MultiPickListMapVectorizer().setMinSupport(0).setCleanKeys(true).setTrackNulls(false)
      .setInput(tech, cnty).setTopK(3)
    val fitted = kcVectorizer.fit(ds)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(ds)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)

    printRes(transformed, vectorMetadata, kcVectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(8, Array(1, 3, 6, 7), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(0, 1, 2, 4, 5), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(3, 7), Array(2.0, 1.0)),
      Vectors.sparse(8, Array(0, 2, 3, 4, 5, 6), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "track nulls with multiple elements (top 3)" in {
    val kcVectorizer = new MultiPickListMapVectorizer().setMinSupport(0).setCleanKeys(true).setTrackNulls(true)
      .setInput(tech, cnty).setTopK(3)
    val fitted = kcVectorizer.fit(ds)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(ds)
    val vectorMetadata = fitted.getMetadata()
    val result = transformed.collect(vector)

    printRes(transformed, vectorMetadata, kcVectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(10, Array(1, 3, 7, 8), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 1, 2, 5, 6), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(3, 8), Array(2.0, 1.0)),
      Vectors.dense(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val expect = OpVectorMetadata("", field.metadata).columns.map(c => !c.isOtherIndicator)
    assertNominal(field, expect, result)
    result shouldBe expected
  }

  it should "drop features with max cardinality" in {
    val fitted = estimator.setMaxPctCardinality(0.01)
      .fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      OPVector.empty,
      OPVector.empty,
      OPVector.empty,
      OPVector.empty
    )
    val vector = estimator.getOutput()
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  private def printRes(df: DataFrame, meta: Metadata, outName: String): Unit = {
    if (log.isInfoEnabled) df.show(false)
    log.info("Metadata: {}", OpVectorMetadata(outName, meta).toString)
  }


}
