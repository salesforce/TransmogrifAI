package com.salesforce.op.evaluators

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.mllib.evaluation.RichBinaryClassificationMetrics
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RichBinaryClassificationMetricsTest extends FlatSpec with TestSparkContext {
  val numRecs = 400 // Number or records to use in threshold metrics tests
  val numBins = 100 // Number of bins for threshold metrics
  val scores = Seq.fill(numRecs)(math.random)
  val labels = Seq.fill(numRecs)(math.round(math.random).toDouble)

  val synthRDD = spark.sparkContext.parallelize(scores.zip(labels))

  Spec[RichBinaryClassificationMetrics] should "produce deterministic metrics" in {
    val numComparisons = 5
    val sparkMLMetrics = new RichBinaryClassificationMetrics(scoreAndLabels = synthRDD, numBins = numBins)

    for {_ <- 1 to numComparisons} {
      sparkMLMetrics.confusionMatrixByThreshold() shouldBe sparkMLMetrics.confusionMatrixByThreshold()
    }
  }

}
