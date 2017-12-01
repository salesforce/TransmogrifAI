/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.features.Feature
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class HashingTFTest extends FlatSpec with TestSparkContext {

  // scalastyle:off
  val testData = Seq(
    "Hamlet: To be or not to be - that is the question.",
    "Гамлет: Быть или не быть - вот в чём вопрос.",
    "המלט: להיות או לא להיות - זאת השאלה.",
    "Hamlet: Être ou ne pas être - telle est la question."
  ).map(_.toLowerCase.split(" ").toSeq.toTextList)
  // scalastyle:on

  lazy val (ds, f1): (DataFrame, Feature[TextList]) = TestFeatureBuilder(testData)

  def hash(s: String, numOfFeatures: Int = Transmogrifier.DefaultNumOfFeatures, binary: Boolean = false): Int = {
    new HashingTF(numOfFeatures).setBinary(binary).indexOf(s)
  }

  Spec[HashingTF] should "hash categorical data" in {
    val hashed = f1.tf()
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.select(hashed.name).collect(hashed)

    hashed.name shouldBe hashed.originStage.outputName

    // scalastyle:off
    results.forall(_.value.size == Transmogrifier.DefaultNumOfFeatures) shouldBe true
    results(0).value(hash("be")) shouldBe 2.0
    results(0).value(hash("that")) shouldBe 1.0
    results(1).value(hash("быть")) shouldBe 2.0
    results(2).value(hash("להיות")) shouldBe 2.0
    results(3).value(hash("être")) shouldBe 2.0
    // scalastyle:on
  }

  it should "hash categorical data with custom numFeatures" in {
    val numFeatures = 100

    val hashed = f1.tf(numTerms = numFeatures)
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.select(hashed.name).collect(hashed)

    // scalastyle:off
    results.forall(_.value.size == numFeatures) shouldBe true
    results(0).value(hash("be", numOfFeatures = numFeatures)) shouldBe 2.0
    results(1).value(hash("быть", numOfFeatures = numFeatures)) shouldBe 2.0
    results(2).value(hash("question", numOfFeatures = numFeatures)) shouldBe 0.0
    // scalastyle:on
  }

  it should "hash categorical data when binary = true" in {
    val binary = true

    val hashed = f1.tf(binary = binary)
    val transformedData = hashed.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.select(hashed.name).collect(hashed)

    // scalastyle:off
    val values = Set(0.0, 1.0)
    results.forall(_.value.toArray.forall(values contains _)) shouldBe true
    results(0).value(hash("be", binary = binary)) shouldBe 1.0
    results(1).value(hash("быть", binary = binary)) shouldBe 1.0
    results(2).value(hash("question", binary = binary)) shouldBe 0.0
    // scalastyle:on
  }
}
