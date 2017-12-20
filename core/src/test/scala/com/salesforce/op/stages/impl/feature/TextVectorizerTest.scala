/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextVectorizerTest extends FlatSpec with TestSparkContext {
  // scalastyle:off
  lazy val (data, f1, f2) = TestFeatureBuilder(
    Seq[(Text, Text)](
      (Text("Hamlet: To be or not to be - that is the question."), Text("Enter Hamlet")),
      (Text("Гамлет: Быть или не быть - вот в чём вопрос."), Text("Входит Гамлет")),
      (Text("המלט: להיות או לא להיות - זאת השאלה."), Text("נככס המלט"))
    )
  )
  // scalastyle:on

  "TextVectorizer" should "work correctly out of the box" in {
    val vectorized = f1.vectorize(numHashes = TransmogrifierDefaults.DefaultNumOfFeatures,
      autoDetectLanguage = TextTokenizer.AutoDetectLanguage,
      minTokenLength = TextTokenizer.MinTokenLength,
      toLowercase = TextTokenizer.ToLowercase
    )
    vectorized.originStage shouldBe a[OPCollectionHashingVectorizer[_]]
    val hasher = vectorized.originStage.asInstanceOf[OPCollectionHashingVectorizer[_]].hashingTF()
    val transformed = new OpWorkflow().setResultFeatures(vectorized).transform(data)
    val result = transformed.collect(vectorized)
    val f1NameHash = hasher.indexOf(vectorized.originStage.getInputFeatures().head.name)

    // scalastyle:off
    result(0).value(hasher.indexOf(s"${f1NameHash}_" + "hamlet")) should be >= 1.0
    result(0).value(hasher.indexOf(s"${f1NameHash}_" + "question")) should be >= 1.0
    result(1).value(hasher.indexOf(s"${f1NameHash}_" + "гамлет")) should be >= 1.0
    result(1).value(hasher.indexOf(s"${f1NameHash}_" + "вопрос")) should be >= 1.0
    result(1).value(hasher.indexOf(s"${f1NameHash}_" + "быть")) should be >= 2.0
    result(2).value(hasher.indexOf(s"${f1NameHash}_" + "המלט")) should be >= 1.0
    result(2).value(hasher.indexOf(s"${f1NameHash}_" + "להיות")) should be >= 2.0
    // scalastyle:on
  }
  it should "allow forcing hashing into a shared hash space" in {
    val vectorized = f1.vectorize(numHashes = TransmogrifierDefaults.DefaultNumOfFeatures,
      autoDetectLanguage = TextTokenizer.AutoDetectLanguage,
      minTokenLength = TextTokenizer.MinTokenLength,
      toLowercase = TextTokenizer.ToLowercase,
      binaryFreq = true,
      others = Array(f2))
    val hasher = vectorized.originStage.asInstanceOf[OPCollectionHashingVectorizer[_]].hashingTF()
    val transformed = new OpWorkflow().setResultFeatures(vectorized).transform(data)
    val result = transformed.collect(vectorized)
    val f1NameHash = hasher.indexOf(vectorized.originStage.getInputFeatures().head.name)

    // scalastyle:off
    result(0).value(hasher.indexOf(s"${f1NameHash}_" + "hamlet")) shouldBe 1.0
    result(0).value(hasher.indexOf(s"${f1NameHash}_" + "hamlet")) shouldBe 1.0
    result(1).value(hasher.indexOf(s"${f1NameHash}_" + "гамлет")) shouldBe 1.0
    result(1).value(hasher.indexOf(s"${f1NameHash}_" + "гамлет")) shouldBe 1.0
    result(2).value(hasher.indexOf(s"${f1NameHash}_" + "המלט")) shouldBe 1.0
    result(2).value(hasher.indexOf(s"${f1NameHash}_" + "המלט")) shouldBe 1.0
    // scalastyle:on
  }
}
