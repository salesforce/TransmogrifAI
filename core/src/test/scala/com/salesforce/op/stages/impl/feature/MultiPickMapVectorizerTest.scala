/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.TestOpVectorColumnType.IndColWithGroup
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.log4j.Level
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.Metadata
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class MultiPickListMapVectorizerTest extends FlatSpec with TestSparkContext {

  val log = LoggerFactory.getLogger(this.getClass)

  lazy val (dataSet, top, bot) = TestFeatureBuilder("top", "bot",
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

  val vectorizer = new MultiPickListMapVectorizer().setCleanKeys(true).setMinSupport(0).setTopK(10).setInput(top, bot)

  lazy val (ds, tech, cnty) = TestFeatureBuilder(
    Seq(
      (Map("tech" -> Set("Spark", "Scala")), Map("cnty" -> Set("Canada", "US"))),
      (Map("tech" -> Set("        sPaRk   ", "Python", "Torch   ")), Map("cnty" -> Set("France ", "UK           "))),
      (Map("tech" -> Set("R", "Hive")), Map("cnty" -> Set("Germany"))),
      (Map("tech" -> Set("python", "TenSoRflow", " TorcH ")), Map("cnty" -> Set("france", "UK", "US")))
    ).map(v => v._1.toMultiPickListMap -> v._2.toMultiPickListMap)
  )

  val nullIndicatorValue = Some(OpVectorColumnMetadata.NullString)

  Spec[MultiPickListMapVectorizer[_]] should
    "take an array of features as input and return a single vector feature" in {
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return the a fitted vectorizer with the correct default parameters" in {
    val fitted = vectorizer.setTrackNulls(false).fit(dataSet)
    fitted shouldBe a[SequenceModel[_, _]]
    val vectorMetadata = fitted.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(vectorizer,
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
    fitted.parent shouldBe vectorizer
  }

  it should "track nulls with the correct default parameters" in {
    val fitted = vectorizer.setTrackNulls(true).fit(dataSet)
    fitted shouldBe a[SequenceModel[_, _]]
    val vectorMetadata = fitted.getMetadata()
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(vectorizer,
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
    fitted.parent shouldBe vectorizer
  }

  it should "return the expected vector with the default param settings" in {
    val fitted = vectorizer.setTrackNulls(false).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "track nulls with the default param settings" in {
    val fitted = vectorizer.setTrackNulls(true).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(20, Array(2, 3, 7, 10, 15, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(2, 4, 9, 12, 13, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(0, 6, 9, 10, 13, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(20, Array(0, 3, 9, 12, 15, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "not clean the variable names when clean text is set to false" in {
    val fitted = vectorizer.setTrackNulls(false).setCleanText(false).setCleanKeys(false).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(17, Array(3, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(4, 12, 15), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(0, 9, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(1, 3, 14), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(vectorizer,
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
    val fitted = vectorizer.setTrackNulls(true).setCleanText(false).setCleanKeys(false).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(23, Array(3, 4, 8, 11, 18, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(3, 5, 10, 14, 16, 20), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(0, 7, 10, 12, 15, 22), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(23, Array(1, 4, 10, 14, 18, 19), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata) shouldEqual
      TestOpVectorMetadataBuilder(vectorizer,
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
    val fitted = vectorizer.setTrackNulls(false).setCleanText(true).setTopK(1).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(12, Array(2, 4, 6), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 8, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 2, 10), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "track nulls when top K is set" in {
    val fitted = vectorizer.setTrackNulls(true).setCleanText(true).setTopK(1).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.sparse(18, Array(2, 3, 6, 9, 14, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(2, 4, 8, 11, 12, 16), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 5, 8, 9, 12, 17), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(18, Array(0, 3, 8, 11, 14, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "return only the elements that exceed the minimum support requirement when minSupport is set" in {
    val fitted = vectorizer.setTrackNulls(false).setCleanText(true).setTopK(10).setMinSupport(2).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val expected = Array(
      Vectors.sparse(10, Array(2, 4, 5), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(3, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 2, 9), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "track nulls when minSupport is set" in {
    val fitted = vectorizer.setTrackNulls(true).setCleanText(true).setTopK(10).setMinSupport(2).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val expected = Array(
      Vectors.sparse(16, Array(2, 3, 6, 8, 13, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(2, 4, 7, 10, 11, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 5, 7, 8, 11, 15), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(16, Array(0, 3, 7, 10, 13, 14), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "behave correctly when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = vectorizer.setTrackNulls(false).setCleanText(true).setMinSupport(0).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected

    val transformed2 = fitted.transform(dataSet)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0)
    ).map(_.toOPVector)
    transformed2.collect(vector) shouldBe expected2
  }

  it should "track nulls when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = vectorizer.setTrackNulls(true).setCleanText(true).setMinSupport(0).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)
    val expected = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected

    val transformed2 = fitted.transform(dataSet)
    val expected2 = Array(
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
      Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
      Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ).map(_.toOPVector)
    transformed2.collect(vector) shouldBe expected2
  }

  it should "behave correctly when passed only empty maps" in {
    val fitted = vectorizer.setInput(top).setTrackNulls(false).setCleanText(true).setTopK(10).fit(dataSetAllEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetAllEmpty)
    val expected = Array(
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double]),
      Vectors.dense(Array.empty[Double])
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "correctly whitelist keys" in {
    val fitted = vectorizer.setTrackNulls(false).setInput(top, bot).setTopK(10).setWhiteListKeys(Array("a", "x"))
      .fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(5, Array(0, 3), Array(1.0, 1.0)),
      Vectors.sparse(5, Array(1), Array(1.0)),
      Vectors.sparse(5, Array(3), Array(1.0)),
      Vectors.sparse(5, Array(0), Array(1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "track nulls with whitelist keys" in {
    val fitted = vectorizer.setTrackNulls(true).setInput(top, bot).setTopK(10).setWhiteListKeys(Array("a", "x"))
      .fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(7, Array(0, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(3, 4), Array(1.0, 1.0)),
      Vectors.sparse(7, Array(0, 6), Array(1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "correctly blacklist keys" in {
    val fitted = vectorizer.setWhiteListKeys(Array()).setTrackNulls(false)
      .setBlackListKeys(Array("a", "x")).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(9, Array(2), Array(1.0)),
      Vectors.sparse(9, Array(5, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 4), Array(1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "track nulls with blacklist keys" in {
    val fitted = vectorizer.setWhiteListKeys(Array()).setTrackNulls(true)
      .setBlackListKeys(Array("a", "x")).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(13, Array(2, 3, 9, 12), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(2, 5, 7, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 9, 10), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(13, Array(0, 5, 6, 12), Array(1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "correctly handle MultiPickList with multiple elements (top 3)" in {
    val kcVectorizer = new MultiPickListMapVectorizer().setMinSupport(0).setCleanKeys(true).setTrackNulls(false)
      .setInput(tech, cnty).setTopK(3)
    val fitted = kcVectorizer.fit(ds)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(ds)
    val vectorMetadata = fitted.getMetadata()

    printRes(transformed, vectorMetadata, kcVectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(8, Array(1, 3, 6, 7), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(0, 1, 2, 4, 5), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(3, 7), Array(2.0, 1.0)),
      Vectors.sparse(8, Array(0, 2, 3, 4, 5, 6), Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "track nulls with multiple elements (top 3)" in {
    val kcVectorizer = new MultiPickListMapVectorizer().setMinSupport(0).setCleanKeys(true).setTrackNulls(true)
      .setInput(tech, cnty).setTopK(3)
    val fitted = kcVectorizer.fit(ds)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(ds)
    val vectorMetadata = fitted.getMetadata()

    printRes(transformed, vectorMetadata, kcVectorizer.getOutputFeatureName)

    val expected = Array(
      Vectors.sparse(10, Array(1, 3, 7, 8), Array(1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(0, 1, 2, 5, 6), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(10, Array(3, 8), Array(2.0, 1.0)),
      Vectors.dense(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  private def printRes(df: DataFrame, meta: Metadata, outName: String): Unit = {
    if (log.isInfoEnabled) df.show(false)
    log.info("Metadata: {}", OpVectorMetadata(outName, meta).toString)
  }
}
