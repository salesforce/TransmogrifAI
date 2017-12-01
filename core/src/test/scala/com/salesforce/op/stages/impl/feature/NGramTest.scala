/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.NGram
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class NGramTest extends FlatSpec with TestSparkContext {

  val data = Seq("a b c d e f g").map(_.split(" ").toSeq.toTextList)
  lazy val (ds, f1) = TestFeatureBuilder(data)

  Spec[NGram] should "generate bigrams by default" in {
    val bigrams = f1.ngram()
    val transformedData = bigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(bigrams)

    bigrams.name shouldBe bigrams.originStage.outputName
    results(0) shouldBe Seq("a b", "b c", "c d", "d e", "e f", "f g").toTextList
  }

  it should "generate unigrams" in {
    val bigrams = f1.ngram(n = 1)
    val transformedData = bigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(bigrams)

    results(0) shouldBe data.head
  }

  it should "generate trigrams" in {
    val trigrams = f1.ngram(n = 3)
    val transformedData = trigrams.originStage.asInstanceOf[Transformer].transform(ds)
    val results = transformedData.collect(trigrams)

    results(0) shouldBe Seq("a b c", "b c d", "c d e", "d e f", "e f g").toTextList
  }

  it should "not allow n < 1" in {
    the[IllegalArgumentException] thrownBy f1.ngram(n = 0)
    the[IllegalArgumentException] thrownBy f1.ngram(n = -1)
  }

}
