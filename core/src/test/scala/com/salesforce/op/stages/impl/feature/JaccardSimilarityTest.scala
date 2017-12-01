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
class JaccardSimilarityTest extends FlatSpec with TestSparkContext {

  val (ds, f1, f2) = TestFeatureBuilder(
    Seq(
      (Seq("Red", "Green"), Seq("Red")),
      (Seq("Red", "Green"), Seq("Yellow, Blue")),
      (Seq("Red", "Yellow"), Seq("Red", "Yellow"))
    ).map(v => v._1.toMultiPickList -> v._2.toMultiPickList)
  )

  val jacSimTrans = new JaccardSimilarity().setInput(f1, f2)

  classOf[JaccardSimilarity].getSimpleName should "return single properly formed feature" in {
    val jaccard = jacSimTrans.getOutput()

    jaccard.name shouldBe jacSimTrans.outputName
    jaccard.parents shouldBe Array(f1, f2)
    jaccard.originStage shouldBe jacSimTrans
  }
  it should "have a shortcut" in {
    val jaccard = f1.jaccardSimilarity(f2)

    jaccard.name shouldBe jaccard.originStage.outputName
    jaccard.parents shouldBe Array(f1, f2)
    jaccard.originStage shouldBe a[JaccardSimilarity]
  }
  it should "return 1 when both vectors are empty" in {
    val set1 = Seq.empty[String].toMultiPickList
    val set2 = Seq.empty[String].toMultiPickList
    jacSimTrans.transformFn(set1, set2) shouldBe 1.0.toRealNN
  }
  it should "return 1 when both vectors are the same" in {
    val set1 = Seq("Red", "Blue", "Green").toMultiPickList
    val set2 = Seq("Red", "Blue", "Green").toMultiPickList
    jacSimTrans.transformFn(set1, set2) shouldBe 1.0.toRealNN
  }

  it should "calculate similarity correctly when vectors are different" in {
    val set1 = Seq("Red", "Green", "Blue").toMultiPickList
    val set2 = Seq("Red", "Blue").toMultiPickList
    jacSimTrans.transformFn(set1, set2) shouldBe (2.0 / 3.0).toRealNN

    val set3 = Seq("Red").toMultiPickList
    val set4 = Seq("Blue").toMultiPickList
    jacSimTrans.transformFn(set3, set4) shouldBe 0.0.toRealNN

    val set5 = Seq("Red", "Yellow", "Green").toMultiPickList
    val set6 = Seq("Pink", "Green", "Blue").toMultiPickList
    jacSimTrans.transformFn(set5, set6) shouldBe (1.0 / 5.0).toRealNN
  }

  it should "calculate similarity correctly on a dataset" in {
    val transformed = jacSimTrans.transform(ds)
    val output = jacSimTrans.getOutput()
    val actualOutput = transformed.collect(output)
    actualOutput shouldBe Seq(0.5, 0.0, 1.0).toRealNN
  }
}
