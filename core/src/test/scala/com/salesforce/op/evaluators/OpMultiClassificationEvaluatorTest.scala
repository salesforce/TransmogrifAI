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


  val (dsMulti, labelRawMulti, predictionMulti) =
    TestFeatureBuilder[RealNN, Prediction](Seq.fill(numRows.toInt)(
      (RealNN(1.0), Prediction(0.0, Vectors.dense(10.0, 5.0, 1.0, 0.0, 0.0), Vectors.dense(0.70, 0.25, 0.05, 0.0, 0.0)))
    ))
  val labelMulti = labelRawMulti.copy(isResponse = true)

  val (dsMulti2, labelRawMulti2, predictionMulti2) =
    TestFeatureBuilder[RealNN, Prediction](Seq(
      (RealNN(1.0), Prediction(1.0, Vectors.dense(10.0, 60.0, 30.0), Vectors.dense(0.10, 0.60, 0.30))),
      (RealNN(0.0), Prediction(1.0, Vectors.dense(5.0, 65.0, 30.0), Vectors.dense(0.05, 0.65, 0.30))),
      (RealNN(2.0), Prediction(0.0, Vectors.dense(50.0, 15.0, 35.0), Vectors.dense(0.50, 0.15, 0.35)))
    ))
  val labelMulti2 = labelRawMulti2.copy(isResponse = true)

  val (dsMulti3, labelRawMulti3, predictionMulti3) =
    TestFeatureBuilder[RealNN, Prediction](Seq(
      (RealNN(1.0), Prediction(2.0, Vectors.dense(1e-20, 1e-20, 0.98), Vectors.dense(1e-20, 1e-20, 1.0 - 2e-20))),
      (RealNN(0.0), Prediction(2.0, Vectors.dense(1e-20, 1e-20, 0.98), Vectors.dense(1e-20, 1e-20, 1.0 - 2e-20))),
      (RealNN(2.0), Prediction(2.0, Vectors.dense(1e-20, 1e-20, 0.98), Vectors.dense(1e-20, 1e-20, 1.0 - 2e-20)))
    ))
  val labelMulti3 = labelRawMulti3.copy(isResponse = true)

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

  Spec[OpMultiClassificationEvaluator] should
    "determine incorrect/correct counts from the thresholds with one prediciton input" in {
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti)
      .setPredictionCol(predictionMulti)

    val metricsMulti = evaluatorMulti.evaluateAll(dsMulti)

    metricsMulti.ThresholdMetrics shouldEqual ThresholdMetrics(
      topNs = defaultTopNs,
      thresholds = defaultThresholds,
      correctCounts = expectedCorrects,
      incorrectCounts = expectedIncorrects,
      noPredictionCounts = expectedNoPredictons
    )

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")
  }

  it should "have settable thresholds and topNs" in {
    val thresholds = Array(0.1, 0.2, 0.5, 0.8, 0.9, 1.0)
    val topNs = Array(1, 4, 12)

    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti)
      .setPredictionCol(predictionMulti)
      .setThresholds(thresholds)
      .setTopNs(topNs)

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

    metricsMulti.ThresholdMetrics shouldEqual ThresholdMetrics(
      topNs = topNs,
      thresholds = thresholds,
      correctCounts = expectedCorrects,
      incorrectCounts = expectedIncorrects,
      noPredictionCounts = expectedNoPredictons
    )
  }

  it should "work on the CC dataset" in {
    val topN = Array(1)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti2)
      .setPredictionCol(predictionMulti2)
      .setTopNs(topN)

    val metricsMulti = evaluatorMulti.evaluateAll(dsMulti2)

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(topN(0)).head * 1.0)/dsMulti2.count()
    println(s"Accuracy at threshold zero: $accuracyAtZero")
    assert(accuracyAtZero + metricsMulti.Error == 1.0)
  }

  it should "work on a sim of another CC dataset" in {
    val topN = Array(1)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(labelMulti3)
      .setPredictionCol(predictionMulti3)
      .setTopNs(topN)

    val metricsMulti = evaluatorMulti.evaluateAll(dsMulti3)

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(topN(0)).head * 1.0)/dsMulti3.count()
    println(s"Accuracy at threshold zero: $accuracyAtZero")
    assert(accuracyAtZero + metricsMulti.Error == 1.0)
  }

  it should "work on randomly generated probabilities" in {
    val numClasses = 100
    val numRows = 10000
    val vectors = RandomVector.dense(RandomReal.uniform[Real](0.0, 1.0), numClasses).limit(numRows)
    println(s"vectors: $vectors")
    val probVectors = vectors.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x/denom).toOPVector
    })
    println(s"probVectors: $probVectors")
    val predictions = vectors.zip(probVectors).map{ case (raw, prob) =>
      Prediction(prediction = prob.v.argmax.toDouble, rawPrediction = raw.v.toArray, probability = prob.v.toArray)
    }

    val labels = RandomIntegral.integrals(from = 0, to = numClasses).limit(numRows)
      .map(x => x.value.get.toDouble.toRealNN)
    println(s"labels: $labels")

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)
    // rawDF.show(truncate = false)

    val topNs = Array(1, 3)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)

    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0)/numRows
    println(s"Accuracy at threshold zero: $accuracyAtZero")
    assert(math.abs(accuracyAtZero + metricsMulti.Error - 1.0) < 1e-2)

    // Should also have that the noPredictionCounts at threshold 0 are always 0
    assert(metricsMulti.ThresholdMetrics.noPredictionCounts.forall{
      case(_, v) => v.head == 0L
    })
  }

  // TODO: Check if it's a tie-breaking issue
  ignore should "work on probability vectors where there are complete ties for all classes" in {
    val numClasses = 5
    val numRows = 10
    val vectors = Seq.fill[OPVector](numRows)(Array.fill(numClasses)(4.2).toOPVector)
    println(s"vectors: $vectors")
    val probVectors = vectors.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x/denom).toOPVector
    })
    println(s"probVectors: $probVectors")
    val predictions = vectors.zip(probVectors).map{ case (raw, prob) =>
      Prediction(prediction = math.floor(math.random * numClasses),
        rawPrediction = raw.v.toArray, probability = prob.v.toArray)
    }
    println(s"Prediction: $predictions")

    // val labels = RandomIntegral.integrals(from = 0, to = numClasses).limit(numRows)
    //  .map(x => x.value.get.toDouble.toRealNN)
    val labels = Seq.fill[RealNN](numRows)(RealNN(1.0))
    println(s"labels: $labels")

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)
    // rawDF.show(truncate = false)

    val topNs = Array(1, 3)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)

    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0)/rawDF.count()
    println(s"Accuracy at threshold zero: $accuracyAtZero")
    assert(accuracyAtZero + metricsMulti.Error == 1.0)
  }

  // Complete ties may be triggering the problems we're seeing, but need more checks to make sure. We can also
  // generate scores from a finite set of values - try that to see if partial ties can account for this
  ignore should "work on probability vectors where there are lots of ties (low unique score cardinality)" in {
    val numClasses = 30
    val numRows = 100
    val correctProb = 0.3

    // Try and make the score one large number for a random class, and equal & small probabilities for all other ones
    val truePredIndex = math.floor(math.random * numClasses).toInt
    val myVector = Array.fill(numClasses)(1e-10)
    myVector.update(truePredIndex, 4.0)
    val myOpVector = myVector.toOPVector

    val scoreChoices = RandomText.pickLists(domain = List("0.00001", "0.00002", "3")).limit(numClasses)
      .map(x => x.value.get.toDouble)
    println(s"example scoreChoices: $scoreChoices")
    val vectors = Seq.fill[OPVector](numRows)( RandomText.pickLists(domain = List("0.00001", "0.00001", "3"))
      .limit(numClasses).map(x => x.value.get.toDouble).toOPVector)
    println(s"vectors: $vectors")

    val vectors2 = Seq.fill[OPVector](numRows){
      val truePredIndex = math.floor(math.random * numClasses).toInt
      val myVector = Array.fill(numClasses)(1e-10)
      myVector.update(truePredIndex, 4.0)
      myVector.toOPVector
    }
    println(s"vectors2: $vectors2")

    val probVectors = vectors2.map(v => {
      val expArray = v.value.toArray.map(math.exp)
      val denom = expArray.sum
      expArray.map(x => x/denom).toOPVector
    })

    println(s"probVectors: $probVectors")
    val predictions = vectors2.zip(probVectors).map{ case (raw, prob) =>
      // println(s"raw: $raw")
      // println(s"prob: $prob")
      val res = Prediction(prediction = prob.v.argmax, rawPrediction = raw.v.toArray, probability = prob.v.toArray)
      // println(s"Making prediction: $res")
      res
    }
    println(s"Prediction: $predictions")

    // val labels = RandomIntegral.integrals(from = 0, to = numClasses).limit(numRows)
    //  .map(x => x.value.get.toDouble.toRealNN)
    val labels = predictions.map(x => {
      val rando = math.random
      if (rando < correctProb) x.prediction.toRealNN else RealNN(1.0)
    })
    println(s"labels: $labels")

    val generatedData: Seq[(RealNN, Prediction)] = labels.zip(predictions)
    println(s"generatedData: ${generatedData.toList}")
    val (rawDF, rawLabel, rawPred) = TestFeatureBuilder(generatedData)
    // rawDF.show(truncate = false)

    val topNs = Array(1, 3)
    val evaluatorMulti = new OpMultiClassificationEvaluator()
      .setLabelCol(rawLabel)
      .setPredictionCol(rawPred)
      .setTopNs(topNs)

    val metricsMulti = evaluatorMulti.evaluateAll(rawDF)

    println(s"Error: ${metricsMulti.Error}")
    println(s"F1: ${metricsMulti.F1}")
    println(s"Precision: ${metricsMulti.Precision}")
    println(s"Recall: ${metricsMulti.Recall}")
    println(s"Threshold metrics: ${metricsMulti.ThresholdMetrics}")

    val accuracyAtZero = (metricsMulti.ThresholdMetrics.correctCounts(1).head * 1.0)/rawDF.count()
    println(s"Accuracy at threshold zero: $accuracyAtZero")
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

}
