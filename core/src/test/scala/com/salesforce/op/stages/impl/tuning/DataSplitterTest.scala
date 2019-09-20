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
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DataSplitterTest extends FlatSpec with TestSparkContext with SplitterSummaryAsserts {
  import spark.implicits._

  val seed = 1234L
  val dataCount = 1000
  val MaxTrainingSampleDefault = 1E6.toInt

  val data =
    RandomRDDs.normalVectorRDD(sc, 1000, 3, seed = seed)
      .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDF()

  val dataSplitter = DataSplitter(seed = seed)

  Spec[DataSplitter] should "split the data in the appropriate proportion - 0.0" in {
    val (train, test) = dataSplitter.setReserveTestFraction(0.0).split(data)
    test.count() shouldBe 0
    train.count() shouldBe dataCount
  }

  it should "down-sample when the data count is above the max allowed" in {
    val numRows = 1E6.toInt + 1E6.toInt
    val data =
      RandomRDDs.normalVectorRDD(sc, numRows, 3, seed = seed)
        .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDF()
    dataSplitter.preValidationPrepare(data)
    val dataBalanced = dataSplitter.validationPrepare(data)

    // validationPrepare calls the data sample method that samples the data to a target ratio but there is an epsilon
    // to how precise this function is which is why we need to check around that epsilon
    val samplingErrorEpsilon = (0.1 * MaxTrainingSampleDefault).toInt

    dataBalanced.count() shouldBe MaxTrainingSampleDefault.toLong +- samplingErrorEpsilon.toLong
  }

  it should "set and get maxTrainingSample" in {
    val numRows = 2000
    val maxRows = numRows / 2

    dataSplitter
      .setReserveTestFraction(0.0)
      .setMaxTrainingSample(maxRows)

    dataSplitter.getMaxTrainingSample shouldBe maxRows
  }

  it should "split the data in the appropriate proportion - 0.2" in {
    val (train, test) = dataSplitter.setReserveTestFraction(0.2).split(data)
    math.abs(test.count() - 200) < 30 shouldBe true
    math.abs(train.count() - 800) < 30 shouldBe true
  }

  it should "split the data in the appropriate proportion - 0.6" in {
    val (train, test) = dataSplitter.setReserveTestFraction(0.6).split(data)
    math.abs(test.count() - 600) < 30 shouldBe true
    math.abs(train.count() - 400) < 30 shouldBe true
  }

  it should "keep the data unchanged when prepare is called" in {
    val summary = dataSplitter.preValidationPrepare(data)
    val train = dataSplitter.validationPrepare(data)
    val sampleF = MaxTrainingSampleDefault / dataCount.toDouble
    val downSampleFraction = if (sampleF < 1) sampleF else 1
    train.collect().zip(data.collect()).foreach { case (a, b) => a shouldBe b }
    assertDataSplitterSummary(summary.summaryOpt) { s => s shouldBe DataSplitterSummary(downSampleFraction) }
  }

}
