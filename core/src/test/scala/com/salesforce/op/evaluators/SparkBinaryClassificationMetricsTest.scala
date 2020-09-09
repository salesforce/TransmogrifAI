package com.salesforce.op.evaluators

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.mllib.evaluation.SparkBinaryClassificationMetrics
import org.junit.runner.RunWith
import org.scalatest.{Assertions, FlatSpec}
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class SparkBinaryClassificationMetricsTest extends FlatSpec with TestSparkContext {
  val numRecs = 400 // Number or records to use in threshold metrics tests
  val numBins = 100 // Number of bins for threshold metrics
  val scores = Seq.fill(numRecs)(math.random)
  val labels = Seq.fill(numRecs)(math.round(math.random).toDouble)

  val synthRDD = spark.sparkContext.parallelize(scores.zip(labels))

  Spec[SparkBinaryClassificationMetrics] should "produce deterministic threshold metrics" in {
    val numComparisons = 5

    for {_ <- 1 to numComparisons} {
      val sparkMLMetrics = new SparkBinaryClassificationMetrics(scoreAndLabels = synthRDD, numBins = numBins)
      sparkMLMetrics.confusionMatrixByThreshold().map(_._1).collect() should contain theSameElementsInOrderAs
        sparkMLMetrics.thresholds().collect()
    }
  }

}
