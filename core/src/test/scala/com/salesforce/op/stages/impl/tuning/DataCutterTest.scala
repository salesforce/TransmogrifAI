/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit.{RandomIntegral, RandomReal, RandomVector}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DataCutterTest extends FlatSpec with TestSparkContext {

  // scalastyle:off
  import spark.implicits._
  // scalastyle:on

  val labels = RandomIntegral.integrals(0, 1000).withProbabilityOfEmpty(0).limit(100000)
  val labelsBiased = RandomIntegral.integrals(0, 3).withProbabilityOfEmpty(0).limit(80000) ++
    RandomIntegral.integrals(3, 1000).withProbabilityOfEmpty(0).limit(20000)
  val vectors = RandomVector.sparse(RandomReal.poisson(2), 10).limit(100000)

  val data = labels.zip(vectors).zip(labelsBiased)
  val dataSize = data.size
  val randDF = spark.sparkContext
    .makeRDD(data.map { case ((l, v), b) => (l.toDouble.get, v.value, b.toDouble.get) }).toDS()
  val biasDF = spark.sparkContext
    .makeRDD(data.map { case ((l, v), b) => (b.toDouble.get, v.value, l.toDouble.get) }).toDS()
  val dataCutter = DataCutter(42)

  Spec[DataCutter] should "not filter out any data when the parameters are permissive" in {
    dataCutter
      .setMinLabelFraction(0.0)
      .setMaxLabelCategores(100000)

    val split = dataCutter.split(randDF)
    split.test.count() + split.train.count() shouldBe dataSize
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 1000
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 0

    val split2 = dataCutter.split(biasDF)
    split2.test.count() + split2.train.count() shouldBe dataSize
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 1000
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 0

  }

  it should "throw an error when all the data is filtered out" in {
    dataCutter.setMinLabelFraction(0.4)
    assertThrows[RuntimeException] {
      dataCutter.split(randDF)
    }
  }

  it should "filter out all but the top N label categories" in {
    val split = dataCutter
      .setMinLabelFraction(0.0)
      .setMaxLabelCategores(100)
      .setReserveTestFraction(0.5)
      .split(randDF)
    findDistict(split.train).count() shouldBe 100
    findDistict(split.test).count() shouldBe 100
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 100
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 900


    val split2 = dataCutter.setMaxLabelCategores(3).split(biasDF)
    findDistict(split2.train).collect().toSet shouldEqual Set(0.0, 1.0, 2.0)
    findDistict(split2.test).collect().toSet shouldEqual Set(0.0, 1.0, 2.0)
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 997
  }

  it should "filter out anything that does not have at least the specified data fraction" in {
    val split = dataCutter
      .setMinLabelFraction(0.0012)
      .setMaxLabelCategores(100000)
      .setReserveTestFraction(0.5)
      .split(randDF)
    val distinct = findDistict(randDF).count()
    val distTest = findDistict(split.test)
    val distTrain = findDistict(split.train)
    distTrain.count() shouldEqual distTest.count()
    distTest.count() shouldEqual distTrain.count()
    distTrain.count() < distinct shouldBe true
    distTrain.count() > 0 shouldBe true
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length +
      split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe distinct

    val split2 = dataCutter.setMinLabelFraction(0.20).setReserveTestFraction(0.5).split(biasDF)
    findDistict(split2.train).count() shouldBe 3
    findDistict(split2.test).count() shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 997
  }

  private def findDistict(d: Dataset[_]): Dataset[Double] = d.toDF().map(_.getDouble(0)).distinct()

}