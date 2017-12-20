/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.TestOpVectorColumnType.IndColWithGroup
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
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

  Spec[MultiPickListMapVectorizer[_]] should
    "take an array of features as input and return a single vector feature" in {
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.outputName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "return the a fitted vectorizer with the correct default parameters" in {
    val fitted = vectorizer.fit(dataSet)
    fitted shouldBe a[SequenceModel[_, _]]
    val vectorMetadata = fitted.getMetadata()
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(vectorizer,
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

  it should "return the expected vector with the default param settings" in {
    val fitted = vectorizer.fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)
    val expected = Array(
      Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  it should "not clean the variable names when clean text is set to false" in {
    val fitted = vectorizer.setCleanText(false).setCleanKeys(false).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)
    val expected = Array(
      Vectors.sparse(17, Array(3, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(4, 12, 15), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(0, 9, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(17, Array(1, 3, 14), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
    OpVectorMetadata(vectorizer.outputName, vectorMetadata) shouldEqual TestOpVectorMetadataBuilder(vectorizer,
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


  it should "return only the specified number of elements when top K is set" in {
    val fitted = vectorizer.setCleanText(true).setTopK(1).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)
    val expected = Array(
      Vectors.sparse(12, Array(2, 4, 6), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(3, 8, 11), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 6, 8), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(12, Array(0, 2, 10), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "return only the elements that exceed the minimum support requirement when minSupport is set" in {
    val fitted = vectorizer.setCleanText(true).setTopK(10).setMinSupport(2).fit(dataSet)
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

  it should "behave correctly when passed empty maps and not throw errors when passed data it was not trained with" in {
    val fitted = vectorizer.setCleanText(true).setMinSupport(0).fit(dataSetEmpty)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSetEmpty)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)
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

  it should "behave correctly when passed only empty maps" in {
    val fitted = vectorizer.setInput(top).setCleanText(true).setTopK(10).fit(dataSetAllEmpty)
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
    val fitted = vectorizer.setInput(top, bot).setTopK(10).setWhiteListKeys(Array("a", "x")).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)

    val expected = Array(
      Vectors.sparse(5, Array(0, 3), Array(1.0, 1.0)),
      Vectors.sparse(5, Array(1), Array(1.0)),
      Vectors.sparse(5, Array(3), Array(1.0)),
      Vectors.sparse(5, Array(0), Array(1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "correctly blacklist keys" in {
    val fitted = vectorizer.setWhiteListKeys(Array())
      .setBlackListKeys(Array("a", "x")).fit(dataSet)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    printRes(transformed, vectorMetadata, vectorizer.outputName)

    val expected = Array(
      Vectors.sparse(9, Array(2), Array(1.0)),
      Vectors.sparse(9, Array(5, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 7), Array(1.0, 1.0)),
      Vectors.sparse(9, Array(0, 4), Array(1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  it should "correctly handle MultiPickList with multiple elements (top 3)" in {
    val kcVectorizer = new MultiPickListMapVectorizer().setMinSupport(0).setCleanKeys(true)
      .setInput(tech, cnty).setTopK(3)
    val fitted = kcVectorizer.fit(ds)
    val vector = fitted.getOutput()
    val transformed = fitted.transform(ds)
    val vectorMetadata = fitted.getMetadata()

    printRes(transformed, vectorMetadata, kcVectorizer.outputName)

    val expected = Array(
      Vectors.sparse(8, Array(1, 6), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 1, 2, 4, 5), Array(1.0, 1.0, 1.0, 1.0, 1.0)),
      Vectors.sparse(8, Array(3, 7), Array(1.0, 1.0)),
      Vectors.sparse(8, Array(0, 2, 4, 5, 6), Array(1.0, 1.0, 1.0, 1.0, 1.0))
    ).map(_.toOPVector)
    transformed.collect(vector) shouldBe expected
  }

  private def printRes(df: DataFrame, meta: Metadata, outName: String): Unit = {
    if (log.isInfoEnabled) df.show(false)
    log.info("Metadata: {}", OpVectorMetadata(outName, meta).toString)
  }
}
