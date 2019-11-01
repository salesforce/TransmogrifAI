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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.testkit.{RandomIntegral, RandomReal, RandomVector}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DataCutterTest extends FlatSpec with TestSparkContext with SplitterSummaryAsserts {
  import spark.implicits._

  val labels = RandomIntegral.integrals(0, 1000).withProbabilityOfEmpty(0).limit(100000)
  val labelsBiased = {
    RandomIntegral.integrals(0, 3).withProbabilityOfEmpty(0).limit(80000) ++
      RandomIntegral.integrals(3, 1000).withProbabilityOfEmpty(0).limit(20000)
  }
  val vectors = RandomVector.sparse(RandomReal.poisson(2), 10).limit(100000)

  val trainingLimitDefault = 1E6.toLong
  val data = labels.zip(vectors).zip(labelsBiased)
  val dataSize = data.size
  val randDF = sc.makeRDD(data.map { case ((l, v), b) => (l.toDouble.get, v.value, b.toString) }).toDF()
  val biasDF = sc.makeRDD(data.map { case ((l, v), b) => (b.toDouble.get, v.value, l.toString) }).toDF()
  val seed = 42L

  Spec[DataCutter] should "not filter out any data when the parameters are permissive" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100000)
    val s1 = dc1.preValidationPrepare(randDF)
    val split1 = dc1.validationPrepare(randDF)

    split1.count() shouldBe dataSize
    assertDataCutterSummary(s1.summaryOpt) { s =>
      s.labelsKept.length shouldBe 1000
      s.labelsDropped.length shouldBe 0
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc1.getLabelsToKeep,
        dc1.getLabelsToDrop,
        dc1.getLabelsDroppedTotal
      )
    }

    val dc2 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100000)
    val s2 = dc2.preValidationPrepare(biasDF)
    val split2 = dc2.validationPrepare(biasDF)

    split2.count() shouldBe dataSize
    assertDataCutterSummary(s2.summaryOpt) { s =>
      s.labelsKept.length shouldBe 1000
      s.labelsDropped.length shouldBe 0
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc2.getLabelsToKeep,
        dc2.getLabelsToDrop,
        dc2.getLabelsDroppedTotal
      )
    }
  }

  it should "set and get all data cutter params" in {
    val maxRows = dataSize / 2
    val downSampleFraction = maxRows / dataSize.toDouble

    val dataCutter = DataCutter()
      .setSeed(seed)
      .setReserveTestFraction(0.0)
      .setMaxLabelCategories(100000)
      .setMinLabelFraction(0.0)
      .setMaxTrainingSample(maxRows)
      .setDownSampleFraction(downSampleFraction)

    dataCutter.getSeed shouldBe seed
    dataCutter.getReserveTestFraction shouldBe 0.0
    dataCutter.getMaxLabelCategories shouldBe 100000
    dataCutter.getMinLabelFraction shouldBe 0.0
    dataCutter.getMaxTrainingSample shouldBe maxRows
    dataCutter.getDownSampleFraction shouldBe downSampleFraction
  }

  it should "down-sample when the data count is above the default training limit" in {
    val numRows = trainingLimitDefault * 2
    val dataCutter = DataCutter()
    val data =
      RandomRDDs.normalVectorRDD(sc, numRows, 3, seed = seed)
        .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDF()

    dataCutter.preValidationPrepare(data)
    val dataBalanced = dataCutter.validationPrepare(data)
    // validationPrepare calls the data sample method that samples the data to a target ratio but there is an epsilon
    // to how precise this function is which is why we need to check around that epsilon
    val samplingErrorEpsilon = (0.1 * trainingLimitDefault).toLong

    dataCutter.getDownSampleFraction shouldBe 0.5
    dataBalanced.count() shouldBe trainingLimitDefault +- samplingErrorEpsilon
  }

  it should "throw an error when all the data is filtered out" in {
    val dataCutter = DataCutter(seed = seed, minLabelFraction = 0.4)
    assertThrows[RuntimeException](dataCutter.preValidationPrepare(randDF))
  }

  it should "throw an error when prepare is called before examine" in {
    val dataCutter = DataCutter(seed = seed, minLabelFraction = 0.4)
    intercept[RuntimeException](dataCutter.validationPrepare(randDF)).getMessage shouldBe
      "requirement failed: Cannot call validationPrepare until preValidationPrepare has been called"
  }

  it should "filter out all but the top N label categories" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100, reserveTestFraction = 0.5)
    val s1 = dc1.preValidationPrepare(randDF)
    val split1 = dc1.validationPrepare(s1.dataFrame.get)

    findDistinct(split1).count() shouldBe 100
    assertDataCutterSummary(s1.summaryOpt) { s =>
      s.labelsKept.length shouldBe 100
      s.labelsDropped.length shouldBe 10
      s.labelsDroppedTotal shouldBe 900
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc1.getLabelsToKeep,
        dc1.getLabelsToDrop,
        dc1.getLabelsDroppedTotal
      )
    }

    val dc2 = DataCutter(seed = seed).setMaxLabelCategories(3)
    val s2 = dc2.preValidationPrepare(biasDF)
    val split2 = dc2.validationPrepare(s2.dataFrame.get)

    findDistinct(split2).collect().toSet shouldEqual Set(0.0, 1.0, 2.0)
    assertDataCutterSummary(s2.summaryOpt) { s =>
      s.labelsKept.length shouldBe 3
      s.labelsDropped.length shouldBe 10
      s.labelsDroppedTotal shouldBe 997
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc2.getLabelsToKeep,
        dc2.getLabelsToDrop,
        dc2.getLabelsDroppedTotal
      )
    }
  }

  it should "filter out anything that does not have at least the specified data fraction" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0012, maxLabelCategories = 100000, reserveTestFraction = 0.5)
    val s1 = dc1.preValidationPrepare(randDF)
    val split1 = dc1.validationPrepare(s1.dataFrame.get)

    val distinct = findDistinct(randDF).count()
    val distTrain = findDistinct(split1)
    distTrain.count() < distinct shouldBe true
    distTrain.count() > 0 shouldBe true
    assertDataCutterSummary(s1.summaryOpt) { s =>
      s.labelsKept.length + s.labelsDroppedTotal shouldBe distinct
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc1.getLabelsToKeep,
        dc1.getLabelsToDrop,
        dc1.getLabelsDroppedTotal
      )
    }

    val dc2 = DataCutter(seed = seed, minLabelFraction = 0.2, reserveTestFraction = 0.5)
    val s2 = dc2.preValidationPrepare(biasDF)
    val split2 = dc2.validationPrepare(s2.dataFrame.get)
    findDistinct(split2).count() shouldBe 3
    assertDataCutterSummary(s2.summaryOpt) { s =>
      s.labelsKept.length shouldBe 3
      s.labelsDroppedTotal shouldBe 997
      s.labelsDropped.length shouldBe 10
      s shouldBe DataCutterSummary(
        preSplitterDataCount = dataSize,
        downSamplingFraction = 1.0,
        dc2.getLabelsToKeep,
        dc2.getLabelsToDrop,
        dc2.getLabelsDroppedTotal
      )
    }
  }

  it should "filter out using labels to keep/drop params" in {
    val keep = Seq(0.0, 1.0)
    val drop = Seq(5.0, 7.0)
    val dc = DataCutter(seed = seed).setLabels(keep = keep, dropTopK = drop, labelsDropped = 2)
    val s1 = dc.preValidationPrepare(randDF)
    val split = dc.validationPrepare(s1.dataFrame.get)

    findDistinct(split).collect().sorted shouldBe Array(0.0, 1.0)
    dc.getLabelsToKeep shouldBe keep.toArray
    dc.getLabelsToDrop shouldBe drop.toArray
  }

  private def findDistinct(d: Dataset[_]): Dataset[Double] = d.toDF().map(_.getDouble(0)).distinct()

}
