/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.stages.impl.selector.ModelSelectorBaseNames
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.testkit.{RandomIntegral, RandomReal, RandomVector}
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DataCutterTest extends FlatSpec with TestSparkContext {

  import spark.implicits._

  val labels = RandomIntegral.integrals(0, 1000).withProbabilityOfEmpty(0).limit(100000)
  val labelsBiased = {
    RandomIntegral.integrals(0, 3).withProbabilityOfEmpty(0).limit(80000) ++
      RandomIntegral.integrals(3, 1000).withProbabilityOfEmpty(0).limit(20000)
  }
  val vectors = RandomVector.sparse(RandomReal.poisson(2), 10).limit(100000)

  val data = labels.zip(vectors).zip(labelsBiased)
  val dataSize = data.size
  val randDF = sc.makeRDD(data.map { case ((l, v), b) => (l.toDouble.get, v.value, b.toString) }).toDS()
  val biasDF = sc.makeRDD(data.map { case ((l, v), b) => (b.toDouble.get, v.value, l.toString) }).toDS()
  val seed = 42L

  Spec[DataCutter] should "not filter out any data when the parameters are permissive" in {
    val dataCutter = DataCutter(seed = seed).setMinLabelFraction(0.0).setMaxLabelCategories(100000)
    val split = dataCutter.prepare(randDF)
    split.train.count() shouldBe dataSize
    val keptMeta = split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept)
    keptMeta.length shouldBe 1000
    keptMeta should contain theSameElementsAs dataCutter.getLabelsToKeep
    val dropMeta = split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped)
    dropMeta.length shouldBe 0
    dropMeta should contain theSameElementsAs dataCutter.getLabelsToDrop

    val split2 = DataCutter(seed = seed)
      .setMinLabelFraction(0.0)
      .setMaxLabelCategories(100000)
      .prepare(biasDF)
    split2.train.count() shouldBe dataSize
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 1000
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 0
  }

  it should "throw an error when all the data is filtered out" in {
    val dataCutter = DataCutter(seed = seed)
      .setMinLabelFraction(0.4)
    assertThrows[RuntimeException] {
      dataCutter.prepare(randDF)
    }
  }

  it should "filter out all but the top N label categories" in {
    val split = DataCutter(seed = seed)
      .setMinLabelFraction(0.0)
      .setMaxLabelCategories(100)
      .setReserveTestFraction(0.5)
      .prepare(randDF)

    findDistinct(split.train).count() shouldBe 100
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 100
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 900

    val split2 = DataCutter(seed = seed).setMaxLabelCategories(3).prepare(biasDF)
    findDistinct(split2.train).collect().toSet shouldEqual Set(0.0, 1.0, 2.0)
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 997
  }

  it should "filter out anything that does not have at least the specified data fraction" in {
    val split = DataCutter(seed = seed)
      .setMinLabelFraction(0.0012)
      .setMaxLabelCategories(100000)
      .setReserveTestFraction(0.5)
      .prepare(randDF)

    val distinct = findDistinct(randDF).count()
    val distTrain = findDistinct(split.train)
    distTrain.count() < distinct shouldBe true
    distTrain.count() > 0 shouldBe true
    split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length +
      split.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe distinct

    val split2 = DataCutter(seed = seed).setMinLabelFraction(0.20).setReserveTestFraction(0.5).prepare(biasDF)
    findDistinct(split2.train).count() shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsKept).length shouldBe 3
    split2.metadata.getDoubleArray(ModelSelectorBaseNames.LabelsDropped).length shouldBe 997
  }

  it should "filter out using the var labelsToKeep" in {
    val keep = Set(0.0, 1.0)
    val drop = Set(5.0, 7.0)
    val dataCutter = DataCutter(seed = seed).setLabels(keep = keep, drop = drop)
    val split = dataCutter.prepare(randDF)
    findDistinct(split.train).collect().sorted shouldBe Array(0.0, 1.0)
    dataCutter.getLabelsToKeep shouldBe keep.toArray
    dataCutter.getLabelsToDrop shouldBe drop.toArray
  }

  private def findDistinct(d: Dataset[_]): Dataset[Double] = d.toDF().map(_.getDouble(0)).distinct()

}
