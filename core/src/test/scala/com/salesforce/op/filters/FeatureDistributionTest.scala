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

import com.salesforce.op.features.TransientFeature
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.testkit.RandomText
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class FeatureDistributionTest extends FlatSpec with PassengerSparkFixtureTest with FiltersTestData {

  Spec[FeatureDistribution] should "be correctly created for features" in {
    val features = Array(survived, age, gender, height, weight).map(TransientFeature.apply)
    val values: Array[(Boolean, ProcessedSeq)] = Array(
      (false, Right(Seq(1.0))), (true, Right(Seq.empty[Double])), (false, Left(Seq("male", "female"))),
      (true, Left(Seq.empty[String])), (false, Right(Seq(1.0, 3.0, 5.0)))
    )
    val summary =
      Array(Summary(0.0, 1.0, 6.0, 10), Summary(-1.6, 10.6, 3.0, 10),
        Summary(0.0, 3.0, 7.0, 10), Summary(0.0, 0.0, 5.0, 10), Summary(1.0, 5.0, 10.0, 10))
    val bins = 10

    val featureKeys: Array[FeatureKey] = features.map(f => (f.name, None))
    val processedSeqs: Array[Option[ProcessedSeq]] = values.map { case (isEmpty, processed) =>
      if (isEmpty) None else Option(processed)
    }
    val distribs = featureKeys.zip(summary).zip(processedSeqs).map { case ((key, summ), seq) =>
      FeatureDistribution(key, summ, seq, bins)
    }
    distribs.foreach{ d =>
      d.key shouldBe None
      d.count shouldBe 1
      d.distribution.length shouldBe bins
    }
    distribs(0).nulls shouldBe 0
    distribs(1).nulls shouldBe 1
    distribs(1).distribution.sum shouldBe 0
    distribs(2).distribution.sum shouldBe 2
    distribs(2).summaryInfo should contain theSameElementsAs Array(0.0, 3.0, 7.0, 10.0)
    distribs(3).distribution.sum shouldBe 0
    distribs(4).distribution.sum shouldBe 3
    distribs(4).summaryInfo.length shouldBe bins
  }

  it should "be correctly created for text features" in {
    val features = Array(description, gender)
    val values: Array[(Boolean, ProcessedSeq)] = Array(
      (false, Left(RandomText.strings(1, 10).take(10000).toSeq.map(_.value.get)))
    )
    val summary = Array(Summary(1000.0, 50000.0, 70000.0, 10))
    val bins = 100
    val featureKeys: Array[FeatureKey] = features.map(f => (f.name, None))
    val processedSeqs: Array[Option[ProcessedSeq]] = values.map { case (isEmpty, processed) =>
      if (isEmpty) None else Option(processed)
    }
    val distribs = featureKeys.zip(summary).zip(processedSeqs).map { case ((key, summ), seq) =>
      FeatureDistribution(key, summ, seq, bins)
    }

    distribs(0).distribution.length shouldBe 100
    distribs(0).distribution.sum shouldBe 10000

  }

  it should "be correctly created for map features" in {
    val features = Array(stringMap, numericMap, booleanMap).map(TransientFeature.apply)
    val values: Array[Map[String, ProcessedSeq]] = Array(
      Map("A" -> Left(Seq("male", "female"))),
      Map("A" -> Right(Seq(1.0)), "B" -> Right(Seq(1.0))),
      Map("B" -> Right(Seq(0.0))))
    val summary = Array(
      Map("A" -> Summary(0.0, 2.0, 100.0, 10), "B" -> Summary(0.0, 5.0, 10.0, 10)),
      Map("A" -> Summary(-1.6, 10.6, 30.0, 10), "B" -> Summary(0.0, 3.0, 11.0, 10)),
      Map("B" -> Summary(0.0, 0.0, 0.0, 10)))
    val bins = 10
    val distribs = features.map(_.name).zip(summary).zip(values).flatMap { case ((name, summaryMaps), valueMaps) =>
      summaryMaps.map { case (key, summary) =>
        val featureKey = (name, Option(key))
        FeatureDistribution(featureKey, summary, valueMaps.get(key), bins)
      }
    }

    distribs.length shouldBe 5
    distribs.foreach{ d =>
      d.key.contains("A") || d.key.contains("B") shouldBe true
      d.count shouldBe 1
      if (d.name != "booleanMap") d.distribution.length shouldBe bins
      else d.distribution.length shouldBe 2
    }
    distribs(0).nulls shouldBe 0
    distribs(0).summaryInfo should contain theSameElementsAs Array(0.0, 2.0, 100.0, 10.0)
    distribs(1).nulls shouldBe 1
    distribs(0).distribution.sum shouldBe 2
    distribs(1).distribution.sum shouldBe 0
    distribs(2).summaryInfo.length shouldBe bins
    distribs(2).distribution.sum shouldBe 1
    distribs(4).distribution(0) shouldBe 1
    distribs(4).distribution(1) shouldBe 0
    distribs(4).summaryInfo.length shouldBe 4
  }

  it should "correctly compare fill rates" in {
    val fd1 = FeatureDistribution("A", None, 10, 1, Array.empty, Array.empty)
    val fd2 = FeatureDistribution("A", None, 20, 20, Array.empty, Array.empty)
    fd1.relativeFillRate(fd2) shouldBe 0.9
  }

  it should "correctly compare relative fill rates" in {
    val fd1 = FeatureDistribution("A", None, 10, 1, Array.empty, Array.empty)
    val fd2 = FeatureDistribution("A", None, 20, 19, Array.empty, Array.empty)
    trainSummaries(0).relativeFillRatio(scoreSummaries(0)) shouldBe 4.5
    trainSummaries(2).relativeFillRatio(scoreSummaries(2)) shouldBe 1.0
    fd1.relativeFillRatio(fd2) shouldBe 18.0
  }

  it should "correctly compute the DS divergence" in {
    val fd1 = FeatureDistribution("A", None, 10, 1, Array(1, 4, 0, 0, 6), Array.empty)
    val fd2 = FeatureDistribution("A", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty)
    fd1.jsDivergence(fd2) should be < eps

    val fd3 = FeatureDistribution("A", None, 10, 1, Array(0, 0, 1000, 1000, 0), Array.empty)
    fd3.jsDivergence(fd3) should be < eps
    val fd4 = FeatureDistribution("A", None, 20, 20, Array(200, 800, 0, 0, 1200), Array.empty)
    (fd3.jsDivergence(fd4) - 1.0) should be < eps
  }
}
