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

import com.salesforce.op.features.types._
import com.salesforce.op.features.{FeatureBuilder, OPFeature, TransientFeature}
import com.salesforce.op.filters.Summary._
import com.salesforce.op.stages.impl.feature.TimePeriod
import com.salesforce.op.stages.impl.preparators.CorrelationType
import com.salesforce.op.test.{Passenger, PassengerSparkFixtureTest}
import com.twitter.algebird.Operators._
import com.twitter.algebird.{HyperLogLogMonoid, Max, SparseHLL, Tuple2Semigroup}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success, Try}

@RunWith(classOf[JUnitRunner])
class PreparedFeaturesTest extends FlatSpec with PassengerSparkFixtureTest {

  import PreparedFeaturesTestData._

  def compareSummary(s1: Summary, s2: Summary): Boolean = {
    (s1.hll.estimatedSize.toInt == s2.hll.estimatedSize.toInt) &&
      (s1.count == s2.count) && (s1.max == s2.max) && (s1.min == s2.min) &&
      (s1.sum == s2.sum)
  }

  Spec[PreparedFeatures] should "produce correct summaries" in {
    val (responseSummaries1, predictorSummaries1) = preparedFeatures1.summaries
    val (responseSummaries2, predictorSummaries2) = preparedFeatures2.summaries
    val (responseSummaries3, predictorSummaries3) = preparedFeatures3.summaries
    val hllMonoid = new HyperLogLogMonoid(RawFeatureFilter.hllbits)

    compareSummary(
      responseSummaries1.get(responseKey1).get, Summary(1.0, 1.0, 1.0, 1, SparseHLL(12, Map(2273 -> Max(2))))
    ) shouldBe true

    compareSummary(
      responseSummaries1.get(responseKey2).get, Summary(0.5, 0.5, 0.5, 1, SparseHLL(12, Map(2273 -> Max(2))))
    ) shouldBe true

    compareSummary(
      predictorSummaries1.get(predictorKey1).get, Summary(0.0, 0.0, 0.0, 2, SparseHLL(12, Map(2273 -> Max(2))))
    ) shouldBe true

    compareSummary(
      predictorSummaries1.get(predictorKey2A).get, Summary(2.0, 2.0, 2.0, 1, SparseHLL(12, Map(2273 -> Max(2))))
    ) shouldBe true

    compareSummary(
      predictorSummaries1.get(predictorKey2B).get, Summary(1.0, 1.0, 1.0, 1, SparseHLL(12, Map(2273 -> Max(1))))
    ) shouldBe true

    compareSummary(
      responseSummaries2.get(responseKey1).get, Summary(0.0, 0.0, 0.0, 1, SparseHLL(12, Map(2273 -> Max(2))))
    ) shouldBe true

    compareSummary(
      predictorSummaries2.get(predictorKey1).get, Summary(0.4, 0.5, 0.9, 2, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      responseSummaries3.get(responseKey2).get, Summary(-0.5, -0.5, -0.5, 1, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      predictorSummaries3.get(predictorKey2A).get, Summary(1.0, 1.0, 1.0, 1, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      allResponseSummaries.get(responseKey1).get,  Summary(0.0, 1.0, 1.0, 2, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      allResponseSummaries.get(responseKey2).get, Summary(-0.5, 0.5, 0.0, 2, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      allPredictorSummaries.get(predictorKey1).get, Summary(0.0, 0.5, 0.9, 4, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      allPredictorSummaries.get(predictorKey2A).get, Summary(1.0, 2.0, 3.0, 2, SparseHLL(12, Map(2273 -> Max(2))))
    )

    compareSummary(
      allPredictorSummaries.get(predictorKey2B).get, Summary(1.0, 1.0, 1.0, 1, SparseHLL(12, Map(2273 -> Max(2))))
    )
  }

  it should "produce summaries that are serializable" in {
    Try(spark.sparkContext.makeRDD(allPreparedFeatures).map(_.summaries).reduce(_ + _)) match {
      case Failure(error) => fail(error)
      case Success((responses, predictors)) =>
        responses shouldBe allResponseSummaries
        predictors shouldBe allPredictorSummaries
    }
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

  it should "transform dates for each period" in {
    val expectedBins = Map(
      TimePeriod.DayOfMonth -> 17.0,
      TimePeriod.DayOfWeek -> 6.0,
      TimePeriod.DayOfYear -> 17.0,
      TimePeriod.HourOfDay -> 0.0,
      TimePeriod.MonthOfYear -> 0.0,
      TimePeriod.WeekOfMonth -> 2.0,
      TimePeriod.WeekOfYear -> 2.0
    )
    expectedBins.keys should contain theSameElementsAs TimePeriod.values

    val dateMap = FeatureBuilder.DateMap[Passenger]
      .extract(p => Map("DTMap" -> p.getBoarded.toLong).toDateMap).asPredictor

    val dateFeatures: Array[OPFeature] = Array(boarded, boardedTime, boardedTimeAsDateTime, dateMap)
    val dateDataFrame: DataFrame = dataReader.generateDataFrame(dateFeatures).persist()

    for {
      (period, expectedBin) <- expectedBins
    } {
      def createExpectedDateMap(d: Double, aggregates: Int): Map[FeatureKey, ProcessedSeq] = Map(
        (boarded.name, None) -> Right((0 until aggregates).map(_ => d).toList),
        (boardedTime.name, None) -> Right(List(d)),
        (boardedTimeAsDateTime.name, None) -> Right(List(d)),
        (dateMap.name, Option("DTMap")) -> Right(List(d)))

      val res = dateDataFrame.rdd
        .map(PreparedFeatures(_, Array.empty, dateFeatures.map(TransientFeature(_)), Option(period)))
        .map(_.predictors.mapValues(_.right.map(_.toList)))
        .collect()

      val expectedResults: Seq[Map[FeatureKey, ProcessedSeq]] =
      // The first observation is expected to be aggregated twice
        Seq(createExpectedDateMap(expectedBin, 2)) ++
          Seq.fill(4)(expectedBin).map(createExpectedDateMap(_, 1)) ++
          Seq(Map[FeatureKey, ProcessedSeq]())

      withClue(s"Computed bin for $period period does not match:\n") {
        res should contain theSameElementsAs expectedResults
      }
    }
  }

  def testCorrMatrix(
    responseKeys: Array[FeatureKey],
    correlationType: CorrelationType,
    expectedResult: Seq[Array[Double]]
  ): Unit = {
    val corrRDD = sc.parallelize(allPreparedFeatures.map(_.getNullLabelLeakageVector(responseKeys, allPredictorKeys1)))
    val corrMatrix = Statistics.corr(corrRDD, correlationType.sparkName)

    corrMatrix.colIter.zipWithIndex.map { case(vec, idx) =>
      // It's symmetric, so can drop based on index
      vec.toArray.drop(idx).map(BigDecimal(_).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble)
    }.toSeq should contain theSameElementsInOrderAs expectedResult
  }
}

object PreparedFeaturesTestData {

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
      predictorKey2B -> Left(Seq("iii")))
  )

  val preparedFeatures2 = PreparedFeatures(
    responses = Map(responseKey1 -> Right(Seq(0.0))),
    predictors = Map(predictorKey1 -> Right(Seq(0.4, 0.5)))
  )

  val preparedFeatures3 = PreparedFeatures(
    responses = Map(responseKey2 -> Right(Seq(-0.5))),
    predictors = Map(predictorKey2A -> Left(Seq("iv")))
  )

  val allPreparedFeatures = Seq(preparedFeatures1, preparedFeatures2, preparedFeatures3)
  implicit val sgTuple2 = new Tuple2Semigroup[Map[FeatureKey, Summary], Map[FeatureKey, Summary]]()
  val (allResponseSummaries, allPredictorSummaries) = allPreparedFeatures.map(_.summaries).reduce(_ + _)

  val allResponseKeys1 = Array(responseKey1, responseKey2)
  val allResponseKeys2 = Array(responseKey1)
  val allPredictorKeys1 = Array(predictorKey1, predictorKey2A, predictorKey2B)
  val allPredictorKeys2 = Array(predictorKey1)

}
