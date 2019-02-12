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

  val data = labels.zip(vectors).zip(labelsBiased)
  val dataSize = data.size
  val randDF = sc.makeRDD(data.map { case ((l, v), b) => (l.toDouble.get, v.value, b.toString) }).toDF()
  val biasDF = sc.makeRDD(data.map { case ((l, v), b) => (b.toDouble.get, v.value, l.toString) }).toDF()
  val seed = 42L

  Spec[DataCutter] should "not filter out any data when the parameters are permissive" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100000)
    val split1 = dc1.prepare(randDF)

    split1.train.count() shouldBe dataSize
    assertDataCutterSummary(split1.summary) { s =>
      s.labelsKept.length shouldBe 1000
      s.labelsDropped.length shouldBe 0
      s shouldBe DataCutterSummary(dc1.getLabelsToKeep, dc1.getLabelsToDrop)
    }

    val dc2 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100000)
    val split2 = dc2.prepare(biasDF)

    split2.train.count() shouldBe dataSize
    assertDataCutterSummary(split2.summary) { s =>
      s.labelsKept.length shouldBe 1000
      s.labelsDropped.length shouldBe 0
      s shouldBe DataCutterSummary(dc2.getLabelsToKeep, dc2.getLabelsToDrop)
    }
  }

  it should "throw an error when all the data is filtered out" in {
    val dataCutter = DataCutter(seed = seed, minLabelFraction = 0.4)
    assertThrows[RuntimeException](dataCutter.prepare(randDF))
  }

  it should "filter out all but the top N label categories" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0, maxLabelCategories = 100, reserveTestFraction = 0.5)
    val split1 = dc1.prepare(randDF)

    findDistinct(split1.train).count() shouldBe 100
    assertDataCutterSummary(split1.summary) { s =>
      s.labelsKept.length shouldBe 100
      s.labelsDropped.length shouldBe 900
      s shouldBe DataCutterSummary(dc1.getLabelsToKeep, dc1.getLabelsToDrop)
    }

    val dc2 = DataCutter(seed = seed).setMaxLabelCategories(3)
    val split2 = dc2.prepare(biasDF)

    findDistinct(split2.train).collect().toSet shouldEqual Set(0.0, 1.0, 2.0)
    assertDataCutterSummary(split2.summary) { s =>
      s.labelsKept.length shouldBe 3
      s.labelsDropped.length shouldBe 997
      s shouldBe DataCutterSummary(dc2.getLabelsToKeep, dc2.getLabelsToDrop)
    }
  }

  it should "filter out anything that does not have at least the specified data fraction" in {
    val dc1 = DataCutter(seed = seed, minLabelFraction = 0.0012, maxLabelCategories = 100000, reserveTestFraction = 0.5)
    val split1 = dc1.prepare(randDF)

    val distinct = findDistinct(randDF).count()
    val distTrain = findDistinct(split1.train)
    distTrain.count() < distinct shouldBe true
    distTrain.count() > 0 shouldBe true
    assertDataCutterSummary(split1.summary) { s =>
      s.labelsKept.length + s.labelsDropped.length shouldBe distinct
      s shouldBe DataCutterSummary(dc1.getLabelsToKeep, dc1.getLabelsToDrop)
    }

    val dc2 = DataCutter(seed = seed, minLabelFraction = 0.2, reserveTestFraction = 0.5)
    val split2 = dc2.prepare(biasDF)
    findDistinct(split2.train).count() shouldBe 3
    assertDataCutterSummary(split2.summary) { s =>
      s.labelsKept.length shouldBe 3
      s.labelsDropped.length shouldBe 997
      s shouldBe DataCutterSummary(dc2.getLabelsToKeep, dc2.getLabelsToDrop)
    }
  }

  it should "filter out using labels to keep/drop params" in {
    val keep = Set(0.0, 1.0)
    val drop = Set(5.0, 7.0)
    val dc = DataCutter(seed = seed).setLabels(keep = keep, drop = drop)
    val split = dc.prepare(randDF)

    findDistinct(split.train).collect().sorted shouldBe Array(0.0, 1.0)
    dc.getLabelsToKeep shouldBe keep.toArray
    dc.getLabelsToDrop shouldBe drop.toArray
    assertDataCutterSummary(split.summary) { s =>
      s shouldBe DataCutterSummary(dc.getLabelsToKeep, dc.getLabelsToDrop)
    }
  }

  private def findDistinct(d: Dataset[_]): Dataset[Double] = d.toDF().map(_.getDouble(0)).distinct()

}
