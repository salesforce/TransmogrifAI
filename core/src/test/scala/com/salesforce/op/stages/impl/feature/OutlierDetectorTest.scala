package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryEstimator
import com.salesforce.op.test._
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec
import com.salesforce.op.utils.spark.RichDataset._


class OutlierDetectorTest extends FlatSpec with TestSparkContext{

  val detector: UnaryEstimator[RealNN, Binary] = new OutlierDetector()
  lazy val output = detector.getOutput()
  it should "return an empty dataset when actually checking an empty dataset" in {
    val emptySeq = Seq.empty[Double].map(_.toRealNN)
    val (inputData, f1) = TestFeatureBuilder(emptySeq)
    val result = detector.setInput(f1).fit(inputData).transform(inputData).collect(output)
    println(result.toSeq)
    result shouldBe Array.empty[Boolean]

  }
  it should "return the same dataset when the data contains only one element" in {
    val oneElementSeq = Seq(42.0).map(_.toRealNN)
    val (inputData, f1) = TestFeatureBuilder(oneElementSeq)
    val result = detector.setInput(f1).fit(inputData).transform(inputData).collect(output)
    result shouldBe Array(false.toBinary)

  }

  it should "not detect any outlier when the data contains the same repeating element" in {
    val multipleElementSeq = Seq.fill(100)(42.0).map(_.toRealNN)
    val (inputData, f1) = TestFeatureBuilder(multipleElementSeq)
    val result = detector.setInput(f1).fit(inputData).transform(inputData).collect(output)
    result shouldBe Array.fill(100)(false.toBinary)

  }


  it should "detect an outlier if all the elements are the same but one is much larger" in {
    val element = 42.0
    val multipleElementSeq = Seq.fill(100)(element).map(_.toRealNN)
    val seqWithOutlier = multipleElementSeq :+ ((element * 1e9).toRealNN)
    val (inputData, f1) = TestFeatureBuilder(seqWithOutlier)
    val result = detector.setInput(f1).fit(inputData).transform(inputData).collect(output)
    result shouldBe Array.fill(100)(false.toBinary) :+ true.toBinary

  }
}

