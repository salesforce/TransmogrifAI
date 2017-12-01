/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.classification

import com.salesforce.op.stages.impl.tuning.DataBalancer
import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.random.RandomRDDs._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class DataBalancerTest extends FlatSpec with TestSparkContext {
  val seed = 1234L
  val smallCount = 200
  val bigCount = 800
  val sampleFraction = 0.4
  val maxTrainingSample = 100

  // scalastyle:off
  import spark.implicits._

  // scalastyle:on

  // Generate positive observations following a distribution ~ N((0.0, 0.0, 0.0), I_3)
  val positiveData =
  normalVectorRDD(spark.sparkContext, bigCount, 3, seed = seed)
    .map(v => 1.0 -> Vectors.dense(v.toArray)).toDS()

  // Generate negative observations following a distribution ~ N((10.0, 10.0, 10.0), I_3)
  val negativeData =
  normalVectorRDD(spark.sparkContext, smallCount, 3, seed = seed)
    .map(v => 0.0 -> Vectors.dense(v.toArray.map(_ + 10.0))).toDS()

  val dataBalancer = new DataBalancer()

  Spec[DataBalancer] should "compute the sample proportions" in {
    dataBalancer.getProportions(100, 9900, 0.5, 100000) shouldEqual(50.0 / 99.0, 50.0)
    dataBalancer.getProportions(100, 900, 0.1, 900) shouldEqual(0.9, 0.9)
    dataBalancer.getProportions(100, 400, 0.5, 900) shouldEqual(0.75, 3.0)
    dataBalancer.getProportions(100, 400000, 0.5, 12000) shouldEqual(1.0 / 80.0, 50.0)
    dataBalancer.getProportions(100, 12000, 0.5, 30000) shouldEqual(5.0 / 6.0, 100.0)
    dataBalancer.getProportions(200, 300, 0.5, 1000) shouldEqual(2.0 / 3.0, 1.0)
  }

  it should "rebalance the dataset correctly" in {
    val (bigDataTrain, bigDataTest, smallDataTrain, smallDataTest, downSample, upSample) =
      dataBalancer.getTrainingSplit(negativeData, smallCount, positiveData, bigCount, sampleFraction,
        maxTrainingSample)

    val bigDataTrainCount = bigDataTrain.count()
    val smallDataTrainCount = smallDataTrain.count()
    val smallDataTestCount = smallDataTest.count()
    val bigDataTestCount = bigDataTest.count()
    smallDataTestCount should be < math.round(sampleFraction * (smallDataTestCount + bigDataTestCount))
  }

  it should "not split the dataset when splitData = false" in {
    val (_, bigDataTest, _, smallDataTest, _, _) =
      dataBalancer.setSplitData(false)
        .getTrainingSplit(negativeData, smallCount, positiveData, bigCount, sampleFraction, maxTrainingSample)

    smallDataTest.count() shouldBe 0
    bigDataTest.count() shouldBe 0
  }

  it should "set the data balancing params correctly" in {
    dataBalancer
      .setSampleFraction(0.4)
      .setMaxTrainingSample(80)
      .setSeed(11L)
      .setSplitData(true)


    dataBalancer.getSampleFraction shouldBe 0.4
    dataBalancer.getMaxTrainingSample shouldBe 80
    dataBalancer.getSeed shouldBe 11L
    dataBalancer.getSplitData shouldBe true
  }

}
