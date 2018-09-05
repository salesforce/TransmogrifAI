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

package com.salesforce.op.filters

import scala.math.round

import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.test.TestSparkContext
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class PreparedFeaturesTest extends FlatSpec with TestSparkContext {

  val responseKey1: FeatureKey = "Response1" -> None
  val responseKey2: FeatureKey = "Response2" -> None
  val predictorKey1: FeatureKey = "Predictor1" -> None
  val predictorKey2A: FeatureKey = "Predictor2" -> Option("A")
  val predictorKey2B: FeatureKey = "Predictor2" -> Option("B")

  val preparedFeatures1 = PreparedFeatures(
    responses = Map(responseKey1 -> Right(Seq(1.0)), responseKey2 -> Right(Seq(0.5))),
    predictors = Map(
      predictorKey1 -> Right(Seq(0.0, 0.0)),
      predictorKey2A -> Left(Seq("i", "ii")),
      predictorKey2B -> Left(Seq("iii"))))
  val preparedFeatures2 = PreparedFeatures(
    responses = Map(responseKey1 -> Right(Seq(0.0))),
    predictors = Map(predictorKey1 -> Right(Seq(0.4, 0.5))))
  val preparedFeatures3 = PreparedFeatures(
    responses = Map(responseKey2 -> Right(Seq(-0.5))),
    predictors = Map(predictorKey2A -> Left(Seq("iv"))))
  val allPreparedFeatures = Seq(preparedFeatures1, preparedFeatures2, preparedFeatures3)
  val (allResponseSummaries, allPredictorSummaries) = allPreparedFeatures.map(_.summaries).reduce(_ + _)

  val allResponseKeys1 = Array(responseKey1, responseKey2)
  val allResponseKeys2 = Array(responseKey1)
  val allPredictorKeys1 = Array(predictorKey1, predictorKey2A, predictorKey2B)
  val allPredictorKeys2 = Array(predictorKey1)

  Spec[PreparedFeatures] should "produce correct summaries" in {
    val (responseSummaries1, predictorSummaries1) = preparedFeatures1.summaries
    val (responseSummaries2, predictorSummaries2) = preparedFeatures2.summaries
    val (responseSummaries3, predictorSummaries3) = preparedFeatures3.summaries

    responseSummaries1 should contain theSameElementsAs
      Seq(responseKey1 -> Summary(1.0, 1.0, 1.0, 1), responseKey2 -> Summary(0.5, 0.5, 0.5, 1))
    predictorSummaries1 should contain theSameElementsAs
      Seq(predictorKey1 -> Summary(0.0, 0.0, 0.0, 2), predictorKey2A -> Summary(2.0, 2.0, 2.0, 1),
        predictorKey2B -> Summary(1.0, 1.0, 1.0, 1))
    responseSummaries2 should contain theSameElementsAs
      Seq(responseKey1 -> Summary(0.0, 0.0, 0.0, 1))
    predictorSummaries2 should contain theSameElementsAs
      Seq(predictorKey1 -> Summary(0.4, 0.5, 0.9, 2))
    responseSummaries3 should contain theSameElementsAs
      Seq(responseKey2 -> Summary(-0.5, -0.5, -0.5, 1))
    predictorSummaries3 should contain theSameElementsAs
      Seq(predictorKey2A -> Summary(1.0, 1.0, 1.0, 1))
    allResponseSummaries should contain theSameElementsAs
      Seq(responseKey1 -> Summary(0.0, 1.0, 1.0, 2), responseKey2 -> Summary(-0.5, 0.5, 0.0, 2))
    allPredictorSummaries should contain theSameElementsAs
      Seq(predictorKey1 -> Summary(0.0, 0.5, 0.9, 4), predictorKey2A -> Summary(1.0, 2.0, 3.0, 2),
        predictorKey2B -> Summary(1.0, 1.0, 1.0, 1))
  }

  it should "produce correct null-label leakage vector with single response" in {
    preparedFeatures1.getNullLabelLeakageVector(allResponseKeys2, allPredictorKeys1).toArray shouldEqual
      Array(1.0, 0.0, 0.0, 0.0)

    preparedFeatures2.getNullLabelLeakageVector(allResponseKeys2, allPredictorKeys1).toArray shouldEqual
      Array(0.0, 0.0, 1.0, 1.0)

    preparedFeatures3.getNullLabelLeakageVector(allResponseKeys2, allPredictorKeys1).toArray shouldEqual
      Array(0.0, 1.0, 0.0, 1.0)
  }

  it should "produce correct null-label leakage vector with multiple responses" in {
    preparedFeatures1.getNullLabelLeakageVector(allResponseKeys1, allPredictorKeys1).toArray shouldEqual
      Array(1.0, 0.5, 0.0, 0.0, 0.0)

    preparedFeatures2.getNullLabelLeakageVector(allResponseKeys1, allPredictorKeys1).toArray shouldEqual
      Array(0.0, 0.0, 0.0, 1.0, 1.0)

    preparedFeatures3.getNullLabelLeakageVector(allResponseKeys1, allPredictorKeys1).toArray shouldEqual
      Array(0.0, -0.5, 1.0, 0.0, 1.0)
  }

  it should "produce correct null-label leakage Pearson correlation matrix with multiple responses" in {
    val expected = Seq(
      Array(1.0, 0.87, -0.5, -0.5, -1.0),
      Array(1.0, -0.87, 0.0, -0.87),
      Array(1.0, -0.5, 0.5),
      Array(1.0, 0.5),
      Array(1.0))
    testCorrMatrix(allResponseKeys1, CorrelationType.Pearson, expected)
  }

  it should "produce correct null-label leakage Spearman correlation matrix with multiple responses" in {
    val expected = Seq(
      Array(1.0, 0.87, -0.5, -0.5, -1.0),
      Array(1.0, -0.87, 0.0, -0.87),
      Array(1.0, -0.5, 0.5),
      Array(1.0, 0.5),
      Array(1.0))
    testCorrMatrix(allResponseKeys1, CorrelationType.Spearman, expected)
  }

  it should "produce correct null-label leakage Pearson correlation matrix with single response" in {
    val expected = Seq(
      Array(1.0, -0.5, -0.5, -1.0),
      Array(1.0, -0.5, 0.5),
      Array(1.0, 0.5),
      Array(1.0))
    testCorrMatrix(allResponseKeys2, CorrelationType.Pearson, expected)
  }

  it should "produce correct null-label leakage Spearman correlation matrix with single response" in {
    val expected = Seq(
      Array(1.0, -0.5, -0.5, -1.0),
      Array(1.0, -0.5, 0.5),
      Array(1.0, 0.5),
      Array(1.0))
    testCorrMatrix(allResponseKeys2, CorrelationType.Spearman, expected)
  }

  def testCorrMatrix(
    responseKeys: Array[FeatureKey],
    correlationType: CorrelationType,
    expectedResult: Seq[Array[Double]]
  ): Unit = {
    val corrRDD =
      sc.parallelize(allPreparedFeatures.map(_.getNullLabelLeakageVector(responseKeys, allPredictorKeys1)))
    val corrMatrix = Statistics.corr(corrRDD, correlationType.sparkName)

    corrMatrix.colIter.zipWithIndex.map { case(vec, idx) =>
      // It's symmetric, so can drop based on index
      vec.toArray.drop(idx).map(BigDecimal(_).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble)
    }.toSeq should contain theSameElementsInOrderAs expectedResult
  }
}
