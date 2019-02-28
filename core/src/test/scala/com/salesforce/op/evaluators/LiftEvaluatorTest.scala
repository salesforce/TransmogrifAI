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

import com.salesforce.op.test.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class LiftEvaluatorTest extends FlatSpec with TestSparkContext {

  lazy val labelSeq = for {
    i <- 0 until 10
    j <- 0 until 10
  } yield {
    if (j < i) 1.0
    else 0.0
  }
  lazy val scoreSeq = (0.01 to 1.0 by 0.01)

  lazy val scores = sc.parallelize(scoreSeq)

  lazy val scoresAndLabels = sc.parallelize(scoreSeq.zip(labelSeq))

  "LiftEvaluator.getDefaultScoreBands" should "give proper default bands" in {
    val bands = LiftEvaluator.getDefaultScoreBands(scores)
    bands.head shouldBe(0.0, 0.1, "0-10")
    bands.last shouldBe(0.9, 1.0, "90-100")
    bands.size shouldBe 10
  }

  "LiftEvaluator.categorizeScoreIntoBand" should "categorize scores into correct bands" in {
    val bands = Seq((0.0, 0.5, "A"), (0.5, 0.9, "B"))
    LiftEvaluator.categorizeScoreIntoBand(0.3, bands) shouldBe Some("A")
    LiftEvaluator.categorizeScoreIntoBand(0.95, bands) shouldBe None
    a[MatchError] should be thrownBy {
      LiftEvaluator.categorizeScoreIntoBand(-0.1, bands)
    }
    a[MatchError] should be thrownBy {
      LiftEvaluator.categorizeScoreIntoBand(1.1, bands)
    }
  }

  "LiftEvaluator.aggregateBandedLabels" should "correctly count records within score bands" in {
    val bandedLabels = sc.parallelize(
      Seq(("A", 1.0), ("A", 0.0), ("B", 0.0), ("B", 0.0))
    )
    val perBandCounts = LiftEvaluator.aggregateBandedLabels(bandedLabels)
    val (numTotalA, _) = perBandCounts("A")
    val (_, numPositivesB) = perBandCounts("B")

    numTotalA shouldBe 2L
    numPositivesB shouldBe 0L
  }

  "LiftEvaluator.overallLiftRate" should "calculate an overall rate" in {
    val perBandCountsFilled = Map("A" -> (4L, 2L), "B" -> (1L, 0L))
    val perBandCountsEmpty = Map[String, (Long, Long)]()
    val overallRateFilled = LiftEvaluator.overallLiftRate(perBandCountsFilled)
    val overallRateEmpty = LiftEvaluator.overallLiftRate(perBandCountsEmpty)
    overallRateFilled shouldBe 0.4
    overallRateEmpty.isNaN shouldBe true
  }

  "LiftEvaluator.formatLiftMetricBand" should "format a LiftMetricBand as required" in {
    val perBandCounts = Map("A" -> (4L, 2L))
    val metricBandA = LiftEvaluator.formatLiftMetricBand(0.0, 0.1, "A", perBandCounts, 0.5)
    metricBandA.group shouldBe "A"
    metricBandA.lowerBound shouldBe 0.0
    metricBandA.upperBound shouldBe 0.1
    metricBandA.rate shouldBe 0.5
    metricBandA.average shouldBe 0.5
    metricBandA.totalCount shouldBe 4L
    metricBandA.yesCount shouldBe 2L
    metricBandA.noCount shouldBe 2L

    val metricBandB = LiftEvaluator.formatLiftMetricBand(0.1, 0.2, "B", perBandCounts, 0.5)
    metricBandB.group shouldBe "B"
    metricBandB.lowerBound shouldBe 0.1
    metricBandB.upperBound shouldBe 0.2
    metricBandB.rate.isNaN shouldBe true
    metricBandB.average shouldBe 0.5
    metricBandB.totalCount shouldBe 0L
    metricBandB.yesCount shouldBe 0L
    metricBandB.noCount shouldBe 0L
  }

  "LiftEvaluator.liftMetricBands" should "correctly calculate a Seq[LiftMetricBand]" in {
    val liftSeq = LiftEvaluator.liftMetricBands(scoresAndLabels, LiftEvaluator.getDefaultScoreBands)
    val band010 = liftSeq.find(_.group == "0-10").get
    val band90100 = liftSeq.find(_.group == "90-100").get

    band010.rate shouldBe 0.0
    band010.lowerBound shouldBe 0.0
    band010.upperBound shouldBe 0.1
    band010.average shouldBe 0.45
    band90100.rate shouldBe 0.9
  }

  "LiftEvaluator.liftMetricBands" should "correctly give defaults with empty RDD" in {
    val liftSeq = LiftEvaluator.liftMetricBands(
      sc.parallelize(Seq[(Double, Double)]()),
      LiftEvaluator.getDefaultScoreBands
    )
    val band010 = liftSeq.find(_.group == "0-10").get
    val band90100 = liftSeq.find(_.group == "90-100").get

    band010.rate.isNaN shouldBe true
    band010.lowerBound shouldBe 0.0
    band010.upperBound shouldBe 0.1
    band010.average.isNaN shouldBe true
    band90100.rate.isNaN shouldBe true
  }

  "LiftEvaluator.apply" should "correctly calculate a Seq[LiftMetricBand]" in {
    val liftSeq = LiftEvaluator.apply(scoresAndLabels)
    val band010 = liftSeq.find(_.group == "0-10").get
    val band90100 = liftSeq.find(_.group == "90-100").get

    band010.rate shouldBe 0.0
    band010.lowerBound shouldBe 0.0
    band010.upperBound shouldBe 0.1
    band010.average shouldBe 0.45
    band90100.rate shouldBe 0.9
  }

}
