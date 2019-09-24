package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryEstimator
import com.salesforce.op.test._
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec


class OutlierDetectorTest extends FlatSpec with TestSparkContext{

  val detector: UnaryEstimator[RealNN, Binary] = new OutlierDetector()
  it should "return an empty dataset when actually checking an empty dataset" in {
    val emptySeq = Seq.empty[Double].map(_.toRealNN)
    val (inputData, f1) = TestFeatureBuilder(emptySeq)
    val result = detector.setInput(f1).fit(inputData).transform(inputData).collect()
    result shouldBe Array.empty[Row]

  }
}

