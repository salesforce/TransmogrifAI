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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DataBalancerTest extends FlatSpec with TestSparkContext {
  import spark.implicits._

  val seed = 1234L
  val smallCount = 200
  val bigCount = 800
  val sampleFraction = 0.4
  val maxTrainingSample = 100

  // Generate positive observations following a distribution ~ N((0.0, 0.0, 0.0), I_3)
  val positiveData = {
    RandomRDDs.normalVectorRDD(sc, bigCount, 3, seed = seed)
      .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDF()
  }
  // Generate negative observations following a distribution ~ N((10.0, 10.0, 10.0), I_3)
  val negativeData = {
    RandomRDDs.normalVectorRDD(sc, smallCount, 3, seed = seed)
      .map(v => (0.0, Vectors.dense(v.toArray.map(_ + 10.0)), "B")).toDF()
  }

  val data = positiveData.union(negativeData)

  val dataBalancer = new DataBalancer().setSeed(seed)

  Spec[DataBalancer] should "compute the sample proportions" in {
    dataBalancer.getProportions(100, 9900, 0.5, 100000) shouldEqual(50.0 / 99.0, 50.0)
    dataBalancer.getProportions(100, 900, 0.1, 900) shouldEqual(0.9, 0.9)
    dataBalancer.getProportions(100, 400, 0.5, 900) shouldEqual(0.75, 3.0)
    dataBalancer.getProportions(100, 400000, 0.5, 12000) shouldEqual(1.0 / 80.0, 50.0)
    dataBalancer.getProportions(100, 12000, 0.5, 30000) shouldEqual(5.0 / 6.0, 100.0)
    dataBalancer.getProportions(200, 300, 0.5, 1000) shouldEqual(2.0 / 3.0, 1.0)
  }

  it should "rebalance the dataset correctly" in {
    val (downSample, upSample) = dataBalancer.getProportions(smallCount, bigCount, sampleFraction, maxTrainingSample)
    val reSampled = dataBalancer.rebalance(negativeData, upSample, positiveData, downSample, seed)

    val Array(negData, posData) = Array(0.0, 1.0).map(label => reSampled.filter(_.getDouble(0) == label).persist())
    val negativeCount = negData.count()
    val positiveCount = posData.count()

    math.abs(negativeCount.toDouble / (negativeCount + positiveCount) - sampleFraction) should be < 0.05
  }

  it should "not split the dataset when splitData = false" in {
    val (train, test) = dataBalancer.setReserveTestFraction(0.0).split(negativeData.union(positiveData))
    test.count() shouldBe 0
    train.count() shouldBe smallCount + bigCount
  }

  it should "balance and remember the fractions" in {
    val fraction = 0.4
    val maxSize = 2000

    val balancer = new DataBalancer()
      .setSampleFraction(fraction)
      .setMaxTrainingSample(maxSize)
      .setSeed(11L)

    val ModelData(expected, _) = balancer.prepare(data)

    val (downSample, upSample) = balancer.getProportions(smallCount, bigCount, fraction, maxSize)

    balancer.getUpSampleFraction shouldBe upSample
    balancer.getDownSampleFraction shouldBe downSample
    balancer.getIsPositiveSmall shouldBe false


    // Rerun balancer with set params
    val metadata = balancer.metadataBuilder
    val ModelData(expected2, _) = balancer.prepare(data)
    withClue("Data balancer should no update the metadata"){
      balancer.metadataBuilder shouldBe metadata
    }
    expected.collect() shouldBe expected2.collect()
  }

  it should "remember that data is already balanced" in {
    val fraction = 0.01
    val maxSize = 20000

    val balancer = new DataBalancer()
      .setSampleFraction(fraction)
      .setMaxTrainingSample(maxSize)
      .setSeed(11L)

    val ModelData(expected, _) = balancer.prepare(data)
    balancer.getAlreadyBalancedFraction shouldBe 1.0

    // Rerun balancer with set params
    val metadata = balancer.metadataBuilder
    val ModelData(expected2, _) = balancer.prepare(data)
    withClue("Data balancer should no update the metadata"){
      balancer.metadataBuilder shouldBe metadata
    }
    expected.collect() shouldBe expected2.collect()

  }


  it should "remember that data is already balanced, but needs to be sample because too big" in {
    val fraction = 0.01
    val maxSize = 100

    val balancer = new DataBalancer()
      .setSampleFraction(fraction)
      .setMaxTrainingSample(maxSize)
      .setSeed(11L)

    val ModelData(expected, _) = balancer.prepare(data)
    balancer.getAlreadyBalancedFraction shouldBe maxSize.toDouble / (smallCount + bigCount)

    // Rerun balancer with set params
    val metadata = balancer.metadataBuilder
    val ModelData(expected2, _) = balancer.prepare(data)
    withClue("Data balancer should no update the metadata"){
      balancer.metadataBuilder shouldBe metadata
    }
    expected.collect() shouldBe expected2.collect()

  }

  it should "set the data balancing params correctly" in {
    dataBalancer
      .setSampleFraction(0.4)
      .setMaxTrainingSample(80)
      .setSeed(11L)
      .setReserveTestFraction(0.0)

    dataBalancer.getSampleFraction shouldBe 0.4
    dataBalancer.getMaxTrainingSample shouldBe 80
    dataBalancer.getSeed shouldBe 11L
  }

}
