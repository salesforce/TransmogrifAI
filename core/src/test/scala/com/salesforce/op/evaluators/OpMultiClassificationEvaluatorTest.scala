/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.evaluators

import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomIntegral, RandomReal, RandomText, RandomVector}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpMultiClassificationEvaluatorTest extends FlatSpec with TestSparkContext {

  // loggingLevel(Level.INFO)

  val numRows = 1000L
  val defaultThresholds = (0 to 100).map(_ / 100.0).toArray
  val defaultTopNs = Array(1, 3)
  val defaultTopKs = Array(5, 10, 20, 50, 100)

  val (dsMulti, labelRawMulti, predictionMulti) =
    TestFeatureBuilder[RealNN, Prediction](Seq.fill(numRows.toInt)(
      (RealNN(1.0), Prediction(0.0, Vectors.dense(10.0, 5.0, 1.0, 0.0, 0.0), Vectors.dense(0.70, 0.25, 0.05, 0.0, 0.0)))
    ))
  val labelMulti = labelRawMulti.copy(isResponse = true)

  // Predictions should never be correct for top1 (since correct class has 2nd highest probability).
  // For top3, it should be correct up to a threshold of 0.25
  val expectedCorrects = Map(
    1 -> Seq.fill(defaultThresholds.length)(0L),
    3 -> (Seq.fill(26)(numRows) ++ Seq.fill(defaultThresholds.length - 26)(0L))
  )
  // For top1, prediction is incorrect up to a threshold of 0.7, and then no prediction
  // For top3, prediction is incorrect in the threshold range (0.25, 0.7], then no prediction
  val expectedIncorrects = Map(
    1 -> (Seq.fill(71)(numRows) ++ Seq.fill(defaultThresholds.length - 71)(0L)),
    3 -> (Seq.fill(26)(0L) ++ Seq.fill(71 - 26)(numRows) ++ Seq.fill(defaultThresholds.length - 71)(0L))
  )
  val expectedNoPredictons = Map(
    1 -> (Seq.fill(71)(0L) ++ Seq.fill(defaultThresholds.length - 71)(numRows)),
    3 -> (Seq.fill(26)(0L) ++ Seq.fill(71 - 26)(0L) ++ Seq.fill(defaultThresholds.length - 71)(numRows))
  )

  val expectedTopKZero = Seq.fill(defaultTopKs.length)(0.0)
  val expectedTopKOne = Seq.fill(defaultTopKs.length)(1.0)

  Spec[OpMultiClassificationEvaluator] should
    "determine incorrect/correct counts from the thresholds with one prediciton input" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti)
      .setPredictionCol(predictionMulti)

    val metricsMulti = evaluatorMulti.evaluateAll(dsMulti)

    metricsMulti.ThresholdMetrics shouldEqual MulticlassThresholdMetrics(
      topNs = defaultTopNs,
      thresholds = defaultThresholds,
      correctCounts = expectedCorrects,
      incorrectCounts = expectedIncorrects,
      noPredictionCounts = expectedNoPredictons
    )

    metricsMulti.TopKMetrics shouldEqual MultiClassificationMetricsTopK(
      topKs = defaultTopKs,
      Precision = expectedTopKZero,
      Recall = expectedTopKZero,
      F1 = expectedTopKZero,
      Error = expectedTopKOne
    )
  }

  it should "have settable thresholds, topNs, and topKs" in {
    val thresholds = Array(0.1, 0.2, 0.5, 0.8, 0.9, 1.0)
    val topNs = Array(1, 4, 12)
    val topKs = Array(1, 5, 15, 30)

    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti)
      .setPredictionCol(predictionMulti)
      .setThresholds(thresholds)
      .setTopNs(topNs)
      .setTopKs(topKs)

    val metricsMulti = evaluatorMulti.evaluateAll(dsMulti)

    // Predictions should never be correct for top1 (since correct class has 2nd highest probability).
    // For top4 & top12, it should be correct up to a threshold of 0.25
    val expectedCorrects = Map(
      1 -> Seq(0L, 0L, 0L, 0L, 0L, 0L),
      4 -> Seq(numRows, numRows, 0L, 0L, 0L, 0L),
      12 -> Seq(numRows, numRows, 0L, 0L, 0L, 0L)
    )
    // For top1, prediction is incorrect up to a threshold of 0.7, and then no prediction
    // For top4 & top 12, prediction is incorrect in the threshold range (0.25, 0.7], then no prediction
    val expectedIncorrects = Map(
      1 -> Seq(numRows, numRows, numRows, 0L, 0L, 0L),
      4 -> Seq(0L, 0L, numRows, 0L, 0L, 0L),
      12 -> Seq(0L, 0L, numRows, 0L, 0L, 0L)
    )
    val expectedNoPredictons = Map(
      1 -> Seq(0L, 0L, 0L, numRows, numRows, numRows),
      4 -> Seq(0L, 0L, 0L, numRows, numRows, numRows),
      12 -> Seq(0L, 0L, 0L, numRows, numRows, numRows)
    )

    val expectedTopKZero = Seq.fill(topKs.length)(0.0)
    val expectedTopKOne = Seq.fill(topKs.length)(1.0)

    metricsMulti.ThresholdMetrics shouldEqual MulticlassThresholdMetrics(
      topNs = topNs,
      thresholds = thresholds,
      correctCounts = expectedCorrects,
      incorrectCounts = expectedIncorrects,
      noPredictionCounts = expectedNoPredictons
    )

    metricsMulti.TopKMetrics shouldEqual MultiClassificationMetricsTopK(
      topKs = topKs,
      Precision = expectedTopKZero,
      Recall = expectedTopKZero,
      F1 = expectedTopKZero,
      Error = expectedTopKOne
    )
  }

  it should "work on randomly generated probabilities" in {
    val numClasses = 100

    val vectors = RandomVector.dense(RandomReal.uniform[Real](0.0, 1.0), numClasses).limit(numRows.toInt)
    val probVectors = vectors.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x / denom).toOPVector
    })
    val predictions = vectors.zip(probVectors).map { case (raw, prob) =>
      Prediction(prediction = prob.v.argmax.toDouble, rawPrediction = raw.v.toArray, probability = prob.v.toArray)
    }

    val labels = RandomIntegral.integrals(from = 0, to = numClasses).limit(numRows.toInt)
      .map(x => x.value.get.toDouble.toRealNN)

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)

    val topNs = Array(1, 3)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)
    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    // The accuracy at threshold zero should be the same as what Spark calculates (error = 1 - accuracy)
    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0) / numRows
    accuracyAtZero + metricsMulti.Error shouldBe 1.0

    // Each row should correspond to either a correct, incorrect, or no prediction
    metricsMulti.ThresholdMetrics.correctCounts(1)
      .zip(metricsMulti.ThresholdMetrics.incorrectCounts(1))
      .zip(metricsMulti.ThresholdMetrics.noPredictionCounts(1)).foreach {
      case ((c, i), n) => c + i + n shouldBe numRows
    }

    // The no prediction count should always be 0 when the threshold is 0
    metricsMulti.ThresholdMetrics.noPredictionCounts.foreach {
      case (_, v) => v.head shouldBe 0L
    }

    // Metrics for topK 100 should always equal the regular metrics if the label cardinality is less than or equal
    metricsMulti.TopKMetrics.Precision(4) shouldBe metricsMulti.Precision
    metricsMulti.TopKMetrics.Recall(4) shouldBe metricsMulti.Recall
    metricsMulti.TopKMetrics.F1(4) shouldBe metricsMulti.F1
    metricsMulti.TopKMetrics.Error(4) shouldBe metricsMulti.Error

    // Metrics should improve or be equal as topK increases
    (metricsMulti.TopKMetrics.Precision, metricsMulti.TopKMetrics.Precision.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.Recall, metricsMulti.TopKMetrics.Recall.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.F1, metricsMulti.TopKMetrics.F1.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.Error, metricsMulti.TopKMetrics.Error.drop(1)).zipped.foreach {
      case (a, b) => a should be >= b
    }
  }

  it should "work on probability vectors where there are many ties (low unique score cardinality)" in {
    val numClasses = 200
    val correctProb = 0.3

    // Try and make the score one large number for a random class, and equal & small probabilities for all other ones
    val truePredIndex = math.floor(math.random * numClasses).toInt
    val vectors = Seq.fill[OPVector](numRows.toInt) {
      val truePredIndex = math.floor(math.random * numClasses).toInt
      val myVector = Array.fill(numClasses)(1e-10)
      myVector.update(truePredIndex, 4.0)
      myVector.toOPVector
    }

    val probVectors = vectors.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x / denom).toOPVector
    })
    val predictions = vectors.zip(probVectors).map { case (raw, prob) =>
      Prediction(prediction = prob.v.argmax, rawPrediction = raw.v.toArray, probability = prob.v.toArray)
    }
    val labels = predictions.map(x => if (math.random < correctProb) x.prediction.toRealNN else RealNN(1.0))

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)

    val topNs = Array(1, 3, 5, 10)
    val topKs = Array(5, 10, 20, 50, 100, 200)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)
      .setTopKs(topKs)
    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    // The accuracy at threshold zero should be the same as what Spark calculates (error = 1 - accuracy)
    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0) / numRows
    accuracyAtZero + metricsMulti.Error shouldBe 1.0

    // Each row should correspond to either a correct, incorrect, or no prediction
    metricsMulti.ThresholdMetrics.correctCounts(1)
      .zip(metricsMulti.ThresholdMetrics.incorrectCounts(1))
      .zip(metricsMulti.ThresholdMetrics.noPredictionCounts(1)).foreach {
      case ((c, i), n) => c + i + n shouldBe numRows
    }

    // The no prediction count should always be 0 when the threshold is 0
    metricsMulti.ThresholdMetrics.noPredictionCounts.foreach {
      case (_, v) => v.head shouldBe 0L
    }

    // Metrics for topK 200 should always equal the regular metrics if the label cardinality is less than or equal
    metricsMulti.TopKMetrics.Precision(5) shouldBe metricsMulti.Precision
    metricsMulti.TopKMetrics.Recall(5) shouldBe metricsMulti.Recall
    metricsMulti.TopKMetrics.F1(5) shouldBe metricsMulti.F1
    metricsMulti.TopKMetrics.Error(5) shouldBe metricsMulti.Error

    // Metrics should improve or be equal as topK increases
    (metricsMulti.TopKMetrics.Precision, metricsMulti.TopKMetrics.Precision.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.Recall, metricsMulti.TopKMetrics.Recall.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.F1, metricsMulti.TopKMetrics.F1.drop(1)).zipped.foreach {
      case (a, b) => a should be <= b
    }
    (metricsMulti.TopKMetrics.Error, metricsMulti.TopKMetrics.Error.drop(1)).zipped.foreach {
      case (a, b) => a should be >= b
    }
  }

  // TODO: Add a way to robustly handle scores that tie (either many-way or complete ties)
  ignore should "work on probability vectors where there are ties in probabilities" in {
    val numClasses = 5
    val numRows = 10

    val vectors = Seq.fill[OPVector](numRows)(Array.fill(numClasses)(4.2).toOPVector)
    val probVectors = vectors.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x / denom).toOPVector
    })
    val predictions = vectors.zip(probVectors).map { case (raw, prob) =>
      Prediction(prediction = math.floor(math.random * numClasses),
        rawPrediction = raw.v.toArray, probability = prob.v.toArray)
    }
    val labels = Seq.fill[RealNN](numRows)(RealNN(1.0))

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)

    val topNs = Array(1, 3)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)
    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0) / numRows
    assert(accuracyAtZero + metricsMulti.Error == 1.0)
  }


  it should "not allow topNs to be negative or 0" in {
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setTopNs(Array(0, 1, 3)))
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setTopNs(Array(1, -4, 3)))
  }

  it should "not allow thresholds to be out of the range [0.0, 1.0]" in {
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setThresholds(Array(-0.1, 0.4)))
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setThresholds(Array(1.1, 0.4)))
  }

  it should "not allow topKs to be negative or 0" in {
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setTopKs(Array(0, 1, 5)))
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().setTopKs(Array(-3, 3, 10)))
  }

  it should "not allow topKs to have a length greater than 10" in {
    intercept[java.lang.IllegalArgumentException](new OpMultiClassificationEvaluator().
      setTopKs(Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)))
  }

  it should "find the right confidence bin via search" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()
    val testThreshold = (1 to 9).map(_ / 10.0)

    // test for when the target value is outside of range of the array
    evaluatorMulti.SearchHelper.findThreshold(testThreshold, 0.99) shouldEqual 0.9
    evaluatorMulti.SearchHelper.findThreshold(testThreshold, 0.05) shouldEqual 0.0
    // test for when the target value equals to an element in the array
    testThreshold.foreach( element => {
      evaluatorMulti.SearchHelper.findThreshold(testThreshold, element) shouldEqual element
    })
    // test for when the target value is within the range of the array, but not one of its elements
    testThreshold.foreach( element => {
      evaluatorMulti.SearchHelper.findThreshold(testThreshold, element + 0.05) shouldEqual element
    })
  }

  it should "calculate confusion matrix counts correctly given a RDD" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()

    val testLabels = Array(1.0, 2.0, 3.0, 4.0)
    var labelPredCount = for {
      label <- testLabels
      prediction <- testLabels
    } yield {
      ((label, prediction, 0.5), (label + prediction).toLong)
    }

    var expectedCmValues = Seq(
      2L, 3L, 4L, 5L,
      3L, 4L, 5L, 6L,
      4L, 5L, 6L, 7L,
      5L, 6L, 7L, 8L)

    // test that the returned Confusion Matrix has correct counts
    evaluatorMulti.constructConfusionMatrix(sc.parallelize(labelPredCount), testLabels) shouldEqual expectedCmValues

    // test that the returned Confusion Matrix fills in the zeros
    labelPredCount = testLabels.map(l => ((l, l, 0.5), 1L))
    expectedCmValues = Seq(
      1L, 0L, 0L, 0L,
      0L, 1L, 0L, 0L,
      0L, 0L, 1L, 0L,
      0L, 0L, 0L, 1L)

    evaluatorMulti.constructConfusionMatrix(sc.parallelize(labelPredCount), testLabels) shouldEqual expectedCmValues
  }

  it should "calculate confusion matrices by threshold correctly" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()
    val testConfMatrixNumClasses = 2
    evaluatorMulti.setConfMatrixNumClasses(testConfMatrixNumClasses)

    val testThresholds = Array(0.4, 0.7)
    evaluatorMulti.setConfMatrixThresholds(testThresholds)

    // create a test 2D array where 1st dimension is the label and 2nd dimension is the prediction,
    // and the # of (label, prediction) equals to the value of the label
    // _| 1  2  3
    // 1| 1L 1L 1L
    // 2| 2L 2L 2L
    // 3| 3L 3L 3L
    val testLabels = Array(1.0, 2.0, 3.0)
    val labelAndPrediction = testLabels.flatMap(label => {
      testLabels.flatMap(pred => Seq.fill(label.toInt)((label, pred)))
    })

    val data = Array(0.5, 0.8).flatMap(topProb => {
      labelAndPrediction.map {
        case (label, prediction) => (label, prediction, Array(topProb, 0.99-topProb, 0.01))
      }})
    val outputMetrics = evaluatorMulti.calculateConfMatrixMetricsByThreshold(sc.parallelize(data))

    outputMetrics.confMatrixClassIndices shouldEqual Array(3.0, 2.0)
    outputMetrics.confMatrixNumClasses shouldEqual testConfMatrixNumClasses
    outputMetrics.confMatrixThresholds shouldEqual testThresholds
    outputMetrics.confMatrices.length shouldEqual testThresholds.length
    // topK confusion matrix for p >= 0.4
    outputMetrics.confMatrices(0) shouldEqual
    Seq(
      6L, 6L,
      4L, 4L)
    // topK confusion matrix for p >= 0.7
    outputMetrics.confMatrices(1).toArray shouldEqual
    Seq(
      3L, 3L,
      2L, 2L)
  }

  it should "calculate mis-classifications correctly" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()

    val testMinSupport = 2
    evaluatorMulti.setConfMatrixMinSupport(testMinSupport)

    // create a test 2D array with the count of each label & prediction combination as:
    // row is label and column is prediction
    // _| 1  2  3
    // 1| 2L 3L 4L
    // 2| 3L 4L 5L
    // 3| 4L 5L 6L
    val testLabels = List(1.0, 2.0, 3.0)
    val labelAndPrediction = testLabels.flatMap(label => {
      testLabels.flatMap(pred => Seq.fill(label.toInt + pred.toInt)((label, pred)))
    })

    val outputMetrics = evaluatorMulti.calculateMisClassificationMetrics(sc.parallelize(labelAndPrediction))

    outputMetrics.ConfMatrixMinSupport shouldEqual testMinSupport
    outputMetrics.MisClassificationsByLabel shouldEqual
      Seq(
        MisClassificationsPerCategory(category = 3.0, totalCount = 15L, correctCount = 6L,
          misClassifications = Map(2.0 -> 5L, 1.0 -> 4L)),
        MisClassificationsPerCategory(category = 2.0, totalCount = 12L, correctCount = 4L,
          misClassifications = Map(3.0 -> 5L, 1.0 -> 3L)),
        MisClassificationsPerCategory(category = 1.0, totalCount = 9L, correctCount = 2L,
          misClassifications = Map(3.0 -> 4L, 2.0 -> 3L))
      )

      outputMetrics.MisClassificationsByPrediction shouldEqual
        Seq(
          MisClassificationsPerCategory(category = 3.0, totalCount = 15L, correctCount = 6L,
            misClassifications = Map(2.0 -> 5L, 1.0 -> 4L)),
          MisClassificationsPerCategory(category = 2.0, totalCount = 12L, correctCount = 4L,
            misClassifications = Map(3.0 -> 5L, 1.0 -> 3L)),
          MisClassificationsPerCategory(category = 1.0, totalCount = 9L, correctCount = 2L,
            misClassifications = Map(3.0 -> 4L, 2.0 -> 3L))
        )
  }
}

