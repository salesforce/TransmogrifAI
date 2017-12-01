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
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class NGramSimilarityTest extends FlatSpec with TestSparkContext {

  val (dsCat, f1Cat, f2Cat) = TestFeatureBuilder(
    Seq(
      (Seq("Red", "Green"), Seq("Red")),
      (Seq("Red", "Green"), Seq("Yellow, Blue")),
      (Seq("Red", "Yellow"), Seq("Red", "Yellow")),
      (Seq[String](), Seq("Red", "Yellow")),
      (Seq[String](), Seq[String]()),
      (Seq[String](""), Seq[String]("asdf")),
      (Seq[String](""), Seq[String]("")),
      (Seq[String]("", ""), Seq[String]("", ""))
    ).map(v => v._1.toMultiPickList -> v._2.toMultiPickList)
  )

  val(dsText, f1Text, f2Text) = TestFeatureBuilder(
    Seq[(Text, Text)](
      (Text("Hamlet: To be or not to be - that is the question."), Text("I like like Hamlet")),
      (Text("that is the question"), Text("There is no question")),
      (Text("Just some random text"), Text("I like like Hamlet")),
      (Text("Adobe CreativeSuite 5 Master Collection from cheap 4zp"),
        Text("Adobe CreativeSuite 5 Master Collection from cheap d1x")),
      (Text.empty, Text.empty),
      (Text(""), Text("")),
      (Text(""), Text.empty),
      (Text("asdf"), Text.empty),
      (Text.empty, Text("asdf"))
    )
  )

  Spec[SetNGramSimilarity] should "correctly compute char-n-gram similarity" in {
    val catNGramSimilarity = f1Cat.toNGramSimilarity(f2Cat)
    val transformedDs = catNGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsCat)
    val actualOutput = transformedDs.collect(catNGramSimilarity)

    actualOutput shouldBe Seq(0.3333333134651184, 0.09722214937210083, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[SetNGramSimilarity] should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val catNGramSimilarity = f1Cat.toNGramSimilarity(f2Cat, 5)
    val transformedDs = catNGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsCat)
    val actualOutput = transformedDs.collect(catNGramSimilarity)

    actualOutput shouldBe Seq(0.3333333432674408, 0.12361115217208862, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[TextNGramSimilarity[_]] should "correctly compute char-n-gram similarity" in {
    val nGramSimilarity = f1Text.toNGramSimilarity(f2Text)
    val transformedDs = nGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsText)
    val actualOutput = transformedDs.collect(nGramSimilarity)

    actualOutput shouldBe Seq(0.12666672468185425, 0.6083333492279053, 0.15873020887374878,
      0.9629629850387573, 0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }

  Spec[TextNGramSimilarity[_]] should "correctly compute char-n-gram similarity with nondefault ngram param" in {
    val nGramSimilarity = f1Text.toNGramSimilarity(f2Text, 4)
    val transformedDs = nGramSimilarity.originStage.asInstanceOf[Transformer].transform(dsText)
    val actualOutput = transformedDs.collect(nGramSimilarity)

    actualOutput shouldBe Seq(0.11500000953674316, 0.5666666626930237, 0.1547619104385376, 0.9722222089767456,
      0.0, 0.0, 0.0, 0.0, 0.0).toRealNN
  }
}
