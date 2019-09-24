package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryEstimator
import com.salesforce.op.test._
import org.apache.spark.sql.Row
import org.scalatest.FlatSpec


class OutlierDetectorTest extends FlatSpec with TestSparkContext{

  val detector: UnaryEstimator[RealNN, Binary] = new OutlierDetector()
  it should "return an empty dataset when actually checking an empty dataset" in {

    import spark.sqlContext.implicits._
    val data = spark.sparkContext.parallelize(Seq.empty[Double]).toDS()
    val result = detector.fit(data).transform(data).collect()

    result shouldBe Array.empty[Row]

  }
}
