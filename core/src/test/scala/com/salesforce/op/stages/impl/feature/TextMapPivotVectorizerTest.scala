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
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class TextMapPivotVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[TextMap, OPVector], TextMapPivotVectorizer[TextMap]]
  with AttributeAsserts {

  val log = LoggerFactory.getLogger(classOf[TextMapPivotVectorizerTest])

  lazy val (inputData, top, bot) = TestFeatureBuilder("top", "bot",
    Seq(
      (Map("a" -> "d", "b" -> "d"), Map("x" -> "W")),
      (Map("a" -> "e"), Map("z" -> "w", "y" -> "v")),
      (Map("c" -> "D"), Map("x" -> "w", "y" -> "V")),
      (Map("c" -> "d", "a" -> "d"), Map("z" -> "v"))
    ).map(v => v._1.toTextMap -> v._2.toTextMap)
  )

  lazy val (dataSetEmpty, _, _) = TestFeatureBuilder(top.name, bot.name,
    Seq(
      (Map("a" -> "d", "b" -> "d"), Map[String, String]()),
      (Map("a" -> "e"), Map[String, String]()),
      (Map[String, String](), Map[String, String]())
    ).map(v => v._1.toTextMap -> v._2.toTextMap)
  )

  lazy val (dataSetAllEmpty, _, _) = TestFeatureBuilder(top.name, bot.name, Seq(
    (Map[String, String](), Map[String, String]()),
    (Map[String, String](), Map[String, String]()),
    (Map[String, String](), Map[String, String]())
  ).map(v => v._1.toTextMap -> v._2.toTextMap)
  )

  val estimator = new TextMapPivotVectorizer[TextMap]().setCleanKeys(true).setMinSupport(0).setTopK(10)
    .setTrackNulls(false).setInput(top, bot)

  val expectedResult = Seq(
    Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
    Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
  ).map(_.toOPVector)

  val vector = estimator.getOutput()

  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)


  Spec[TextMapPivotVectorizer[_]] should "take an array of features as input and return a single vector feature" in {
    val vector = estimator.getOutput()
    vector.name shouldBe estimator.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return the a fitted vectorizer with the correct parameters" in {
    val fitted = estimator.fit(inputData)
    fitted.isInstanceOf[SequenceModel[_, _]]
    val vectorMetadata = fitted.getMetadata()
    OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata) shouldEqual
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

  it should "return the expected vector with the default param settings" in {
    val fitted = estimator.fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expectedResult.head.value.size)(true), result)
    result shouldBe expectedResult
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "track nulls" in {
    val fitted = estimator.setTrackNulls(true).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(20, Array(2, 3, 7, 10, 15, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(2, 4, 9, 12, 13, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(0, 6, 9, 10, 13, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(0, 3, 9, 12, 15, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "not clean the variable names when clean text is set to false" in {
    val fitted = estimator.setCleanText(false).setCleanKeys(false).setTrackNulls(false).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(17, Array(3, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(4, 12, 15), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(0, 9, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(1, 3, 14), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
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
    val fitted = estimator.setCleanText(false).setCleanKeys(false).setTrackNulls(true).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(23, Array(3, 4, 8, 11, 18, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(3, 5, 10, 14, 16, 20), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(0, 7, 10, 12, 15, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(1, 4, 10, 14, 18, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
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
    val fitted = estimator.setCleanText(true).setTrackNulls(false).setTopK(1).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(12, Array(2, 4, 6), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 8, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 2, 10), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "track nulls the specified number of elements when top K is set" in {
    val fitted = estimator.setCleanText(true).setTrackNulls(true).setTopK(1).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(18, Array(2, 3, 6, 9, 14, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(2, 4, 8, 11, 12, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 5, 8, 9, 12, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 3, 8, 11, 14, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "return only the elements that exceed the minimum support requirement when minSupport is set" in {
    val fitted = estimator.setCleanText(true).setTopK(10).setTrackNulls(false).setMinSupport(2).fit(inputData)
    val transformed = fitted.transform(inputData)
    val expected = Array(
      Vectors.sparse(10, Array(2, 4, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(3, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 2, 9), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "track nulls the elements that exceed the minimum support requirement when minSupport is set" in {
    val fitted = estimator.setCleanText(true).setTopK(10).setTrackNulls(true).setMinSupport(2).fit(inputData)
    val transformed = fitted.transform(inputData)
    val expected = Array(
      Vectors.sparse(16, Array(2, 3, 6, 8, 13, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(2, 4, 7, 10, 11, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 5, 7, 8, 11, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 3, 7, 10, 13, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "behave correctly when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = estimator.setCleanText(true).setTrackNulls(false).setMinSupport(0).fit(dataSetEmpty)
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(vector)
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected

    val transformed2 = fitted.transform(inputData)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    val field2 = transformed2.schema(vector.name)
    val result2 = transformed2.collect(fitted.getOutput())
    assertNominal(field2, Array.fill(expected.head.value.size)(true), result2)
    result2 shouldBe expected2
  }

  it should "track nulls when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = estimator.setCleanText(true).setTrackNulls(true).setMinSupport(0).fit(dataSetEmpty)
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected

    val transformed2 = fitted.transform(inputData)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)
    val field2 = transformed2.schema(vector.name)
    val result2 = transformed2.collect(fitted.getOutput())
    assertNominal(field2, Array.fill(expected.head.value.size)(true), result2)
    result2 shouldBe expected2
  }


  it should "behave correctly when passed only empty maps" in {
    val fitted = estimator.setCleanText(true).setTrackNulls(false).setTopK(10).fit(dataSetAllEmpty)
    val transformed = fitted.transform(dataSetAllEmpty)
    val expected = Array(
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double])
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "correctly whitelist keys" in {
    val fitted = estimator.setTopK(10).setTrackNulls(false).setWhiteListKeys(Array("a", "x")).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(5, Array(0, 3), Array(1.0, 1.0)),
      Vectors.sparse(5, Array(1), Array(1.0)),
      Vectors.sparse(5, Array(3), Array(1.0)),
      Vectors.sparse(5, Array(0), Array(1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "track nulls with whitelist keys" in {
    val fitted = estimator.setTopK(10).setTrackNulls(true).setWhiteListKeys(Array("a", "x")).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(7, Array(0, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(3, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(0, 6), Array(1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "correctly blacklist keys" in {
    val fitted = estimator.setTrackNulls(false).setWhiteListKeys(Array()).setBlackListKeys(Array("a", "x"))
      .fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(9, Array(2), Array(1.0)),
      Vectors.sparse(9, Array(5, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 4), Array(1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "track nulls with blacklist keys" in {
    val fitted = estimator.setTrackNulls(true).setWhiteListKeys(Array()).setBlackListKeys(Array("a", "x"))
      .fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(13, Array(2, 3, 9, 12), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(2, 5, 7, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 9, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 6, 12), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }

  it should "drop features with max cardinality" in {
    val fitted = estimator.setMaxPctCardinality(0.2).fit(inputData)
    val transformed = fitted.transform(inputData)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(estimator.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array.fill(4)(OPVector.empty)
    val field = transformed.schema(vector.name)
    val result = transformed.collect(fitted.getOutput())
    assertNominal(field, Array.fill(expected.head.value.size)(true), result)
    result shouldBe expected
  }
}
