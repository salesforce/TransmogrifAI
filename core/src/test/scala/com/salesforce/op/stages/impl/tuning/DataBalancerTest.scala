/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
      .map(v => (1.0, Vectors.dense(v.toArray), "A")).toDS()
  }
  // Generate negative observations following a distribution ~ N((10.0, 10.0, 10.0), I_3)
  val negativeData = {
    RandomRDDs.normalVectorRDD(sc, smallCount, 3, seed = seed)
      .map(v => (0.0, Vectors.dense(v.toArray.map(_ + 10.0)), "B")).toDS()
  }

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
    val (resampled, downSample, upSample) =
      dataBalancer.getTrainingSplit(negativeData, smallCount, positiveData, bigCount, sampleFraction, maxTrainingSample)

    val Array(negData, posData) = Array(0.0, 1.0).map(label => resampled.filter(_._1 == label).persist())

    val negativeCount = negData.count()
    val positiveCount = posData.count()

    println(math.abs(negativeCount.toDouble / (negativeCount + positiveCount) - sampleFraction))
    math.abs(negativeCount.toDouble / (negativeCount + positiveCount) - sampleFraction) should be < 0.05
  }

  it should "not split the dataset when splitData = false" in {
    val (train, test) = dataBalancer.setReserveTestFraction(0.0).split(negativeData.union(positiveData))
    test.count() shouldBe 0
    train.count() shouldBe smallCount + bigCount
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
