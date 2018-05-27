/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.filters

import com.salesforce.op.OpParams
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.stages.impl.feature.HashAlgorithm
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.mllib.feature.HashingTF
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RawFeatureFilterTest extends FlatSpec with PassengerSparkFixtureTest {

  private val eps = 1E-2

  private val trainSummaries = Seq(
    FeatureDistrib("A", None, 10, 1, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistrib("B", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistrib("C", Some("1"), 10, 1, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistrib("C", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistrib("D", Some("1"), 10, 9, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistrib("D", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty)
  )

  private val scoreSummaries = Seq(
    FeatureDistrib("A", None, 10, 8, Array(1, 4, 0, 0, 6), Array.empty),
    FeatureDistrib("B", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistrib("C", Some("1"), 10, 1, Array(0, 0, 10, 10, 0), Array.empty),
    FeatureDistrib("C", Some("2"), 20, 19, Array(2, 8, 0, 0, 12), Array.empty),
    FeatureDistrib("D", Some("1"), 0, 0, Array(0, 0, 0, 0, 0), Array.empty),
    FeatureDistrib("D", Some("2"), 0, 0, Array(0, 0, 0, 0, 0), Array.empty)
  )

  Spec[Summary] should "be correctly created from a sequence of features" in {
    val f1 = Left(Seq("a", "b", "c"))
    val f2 = Right(Seq(0.5, 1.0))
    val f1s = Summary(f1)
    val f2s = Summary(f2)
    f1s.min shouldBe 3
    f1s.max shouldBe 3
    f2s.min shouldBe 0.5
    f2s.max shouldBe 1.0
  }

  Spec[FeatureDistrib] should "be correctly created for features" in {
    val features = Array(survived, age, gender, height, weight).map(TransientFeature.apply)
    val values: Array[(Boolean, FeatureDistrib.ProcessedSeq)] = Array(
      (false, Right(Seq(1.0))), (true, Right(Seq.empty[Double])), (false, Left(Seq("male", "female"))),
      (true, Left(Seq.empty[String])), (false, Right(Seq(1.0, 3.0, 5.0)))
    )
    val summary = Array(Summary(0.0, 1.0), Summary(-1.6, 10.6), Summary(0.0, 3.0), Summary(0.0, 0.0), Summary(1.0, 5.0))
    val bins = 10
    val hasher: HashingTF = new HashingTF(numFeatures = bins)
      .setBinary(false)
      .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)

    val distribs = FeatureDistrib.getDistributions(features, values, summary, bins, hasher)
    distribs.foreach{ d =>
      d.key shouldBe None
      d.count shouldBe 1
      d.distribution.length shouldBe bins
    }
    distribs(0).nulls shouldBe 0
    distribs(1).nulls shouldBe 1
    distribs(1).distribution.sum shouldBe 0
    distribs(2).distribution.sum shouldBe 2
    distribs(2).summaryInfo should contain theSameElementsAs Array(0.0, 3.0)
    distribs(3).distribution.sum shouldBe 0
    distribs(4).distribution.sum shouldBe 3
    distribs(4).summaryInfo.length shouldBe bins
  }

  it should "be correctly created for map features" in {
    val features = Array(stringMap, numericMap, booleanMap).map(TransientFeature.apply)
    val values: Array[Map[String, FeatureDistrib.ProcessedSeq]] = Array(
      Map("A" -> Left(Seq("male", "female"))),
      Map("A" -> Right(Seq(1.0)), "B" -> Right(Seq(1.0))),
      Map("B" -> Right(Seq(0.0))))
    val summary = Array(
      Map("A" -> Summary(0.0, 1.0), "B" -> Summary(0.0, 5.0)),
      Map("A" -> Summary(-1.6, 10.6), "B" -> Summary(0.0, 3.0)),
      Map("B" -> Summary(0.0, 0.0)))
    val bins = 10
    val hasher: HashingTF = new HashingTF(numFeatures = bins)
      .setBinary(false)
      .setHashAlgorithm(HashAlgorithm.MurMur3.toString.toLowerCase)

    val distribs = FeatureDistrib.getMapDistributions(features, values, summary, bins, hasher)
    distribs.length shouldBe 5
    distribs.foreach{ d =>
      d.key.contains("A") || d.key.contains("B") shouldBe true
      d.count shouldBe 1
      if (d.name != "booleanMap") d.distribution.length shouldBe bins
      else d.distribution.length shouldBe 2
    }
    distribs(0).nulls shouldBe 0
    distribs(0).summaryInfo should contain theSameElementsAs Array(0.0, 1.0)
    distribs(1).nulls shouldBe 1
    distribs(0).distribution.sum shouldBe 2
    distribs(1).distribution.sum shouldBe 0
    distribs(2).summaryInfo.length shouldBe bins
    distribs(2).distribution.sum shouldBe 1
    distribs(4).distribution(0) shouldBe 1
    distribs(4).distribution(1) shouldBe 0
    distribs(4).summaryInfo.length shouldBe 2
  }

  it should "correctly compare fill rates" in {
    val fd1 = FeatureDistrib("A", None, 10, 1, Array.empty, Array.empty)
    val fd2 = FeatureDistrib("A", None, 20, 20, Array.empty, Array.empty)
    fd1.relativeFillRate(fd2) shouldBe 0.9
  }

  it should "correctly compare relative fill rates" in {
    val fd1 = FeatureDistrib("A", None, 10, 1, Array.empty, Array.empty)
    val fd2 = FeatureDistrib("A", None, 20, 19, Array.empty, Array.empty)
    trainSummaries(0).relativeFillRatio(scoreSummaries(0)) shouldBe 4.5
    trainSummaries(2).relativeFillRatio(scoreSummaries(2)) shouldBe 1.0
    fd1.relativeFillRatio(fd2) shouldBe 18.0
  }

  it should "correctly compute the DS divergence" in {
    val fd1 = FeatureDistrib("A", None, 10, 1, Array(1, 4, 0, 0, 6), Array.empty)
    val fd2 = FeatureDistrib("A", None, 20, 20, Array(2, 8, 0, 0, 12), Array.empty)
    fd1.jsDivergence(fd2) should be < eps

    val fd3 = FeatureDistrib("A", None, 10, 1, Array(0, 0, 1000, 1000, 0), Array.empty)
    fd3.jsDivergence(fd3) should be < eps
    val fd4 = FeatureDistrib("A", None, 20, 20, Array(200, 800, 0, 0, 1200), Array.empty)
    (fd3.jsDivergence(fd4) - 1.0) should be < eps
  }

  Spec[RawFeatureFilter[_]] should "compute feature stats correctly" in {
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.8, Double.PositiveInfinity, 0.7)
    val summaries = filter.computeFeatureStats(passengersDataSet, features)

    summaries.featureSummaries.size shouldBe 7
    summaries.mapFeatureSummaries.size shouldBe 3
    summaries.featureDistributions.size shouldBe 13

    val surv = summaries.featureDistributions(0)
    surv.name shouldBe survived.name
    surv.key shouldBe None
    surv.count shouldBe 6
    surv.nulls shouldBe 4
    surv.distribution.sum shouldBe 2
    val strMapF = summaries.featureDistributions(7)
    strMapF.name shouldBe stringMap.name
    if (strMapF.key.contains("Female")) strMapF.nulls shouldBe 3 else strMapF.nulls shouldBe 4
    val strMapM = summaries.featureDistributions(8)
    strMapM.name shouldBe stringMap.name
    if (strMapM.key.contains("Male")) strMapM.nulls shouldBe 4 else strMapM.nulls shouldBe 3
  }

  it should "correctly determine which features to exclude based on the stats of training fill rate" in {
    // only fill rate matters
    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0)
    val (excludedTrainF, excludedTrainMK) = filter.getFeaturesToExclude(trainSummaries, Seq.empty)
    excludedTrainF.toSet shouldEqual Set("B", "D")
    excludedTrainMK.keySet shouldEqual Set("C")
    excludedTrainMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of training and scoring fill rate" in {
    // only fill rate matters

    val filter = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.2, 1.0, Double.PositiveInfinity, 1.0)
    val (excludedBothF, excludedBothMK) = filter.getFeaturesToExclude(trainSummaries, scoreSummaries)
    excludedBothF.toSet shouldEqual Set("B", "D")
    excludedBothMK.keySet shouldEqual Set("C")
    excludedBothMK.head._2 shouldEqual Set("2")
  }

  it should "correctly determine which features to exclude based on the stats of relative fill rate" in {
    // relative fill rate matters
    val filter2 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 0.5, Double.PositiveInfinity, 1.0)
    val (excludedBothRelF, excludedBothRelMK) = filter2.getFeaturesToExclude(trainSummaries, scoreSummaries)
    excludedBothRelF.toSet shouldEqual Set("A")
    excludedBothRelMK.isEmpty shouldBe true
  }

  it should "correctly determine which features to exclude based on the stats of fill rate ratio" in {
    // relative fill ratio matters
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, 2.0, 1.0)
    val (excludedBothRelFR, excludedBothRelMKR) = filter4.getFeaturesToExclude(trainSummaries, scoreSummaries)
    excludedBothRelFR.toSet shouldEqual Set("D", "A", "B")
    excludedBothRelMKR.isEmpty shouldBe true
  }

  it should "correctly determine which features to exclude based on the stats of js distance" in {
    // js distance
    val filter3 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.0, 1.0, Double.PositiveInfinity, 0.5)
    val (excludedBothDistF, excludedBothDistMK) = filter3.getFeaturesToExclude(trainSummaries, scoreSummaries)
    excludedBothDistF.isEmpty shouldEqual true
    excludedBothDistMK.keySet shouldEqual Set("C")
    excludedBothDistMK.head._2 shouldEqual Set("1")
  }

  it should "correctly determine which features to exclude based on all the stats" in {
    // all
    val filter4 = new RawFeatureFilter(simpleReader, Some(dataReader), 10, 0.1, 0.5, Double.PositiveInfinity, 0.5)
    val (excludedBothAllF, excludedBothAllMK) = filter4.getFeaturesToExclude(trainSummaries, scoreSummaries)
    excludedBothAllF.toSet shouldEqual Set("A", "B", "C", "D")
    excludedBothAllMK.isEmpty shouldBe true
  }

  it should "correctly clean the dataframe returned and give the features to blacklist" in {
    val params = new OpParams()
    val survPred = survived.copy(isResponse = false)
    val features: Array[OPFeature] =
      Array(survPred, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.0, 1.0, Double.PositiveInfinity, 1.0)
    val (df, toDrop) = filter.generateFilteredRaw(features, params)
    toDrop.isEmpty shouldBe true
    df.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields

    val filter1 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0)
    val (df1, toDrop1) = filter1.generateFilteredRaw(features, params)
    toDrop1 should contain theSameElementsAs Array(survPred)
    df1.schema.fields.exists(_.name == survPred.name) shouldBe false
    df1.collect(stringMap).foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
  }

  it should "not drop response features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.5, 0.5, Double.PositiveInfinity, 1.0)
    val (df, toDrop) = filter.generateFilteredRaw(features, params)
    toDrop.isEmpty shouldBe true
    df.schema.fields should contain theSameElementsAs passengersDataSet.schema.fields
    df.collect(stringMap).foreach(m => if (m.nonEmpty) m.value.keySet shouldEqual Set("Female"))
  }


  it should "not drop protected features" in {
    val params = new OpParams()
    val features: Array[OPFeature] =
      Array(survived, age, gender, height, weight, description, boarded)
    val filter = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2)
    val (df, toDrop) = filter.generateFilteredRaw(features, params)
    toDrop.toSet shouldEqual Set(age, gender, height, weight, description, boarded)
    df.schema.fields.map(_.name) should contain theSameElementsAs Array("key", "survived")

    val filter2 = new RawFeatureFilter(dataReader, Some(simpleReader), 10, 0.1, 0.1, 2, 0.2, Set("age", "gender"))
    val (df2, toDrop2) = filter2.generateFilteredRaw(features, params)
    toDrop2.toSet shouldEqual Set(height, weight, description, boarded)
    df2.schema.fields.map(_.name) should contain theSameElementsAs Array("key", "survived", "age", "gender")
  }
}
