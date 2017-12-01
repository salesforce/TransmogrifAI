/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.StringIndexerHandleInvalid.Skip
import com.salesforce.op.stages.sparkwrappers.generic.SwUnaryModel
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.feature.StringIndexerModel
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpStringIndexerNoFilterTest extends FlatSpec with TestSparkContext {

  val txtData = Seq("a", "b", "c", "a", "a", "c").map(_.toText)
  val (ds, txtF) = TestFeatureBuilder(txtData)
  val expected = Array(0.0, 2.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN)

  val txtDataNew = Seq("a", "b", "c", "a", "a", "c", "d", "e").map(_.toText)
  val (dsNew, txtFNew ) = TestFeatureBuilder(txtDataNew)
  val expectedNew = Array(0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0).map(_.toRealNN)


  Spec[OpStringIndexerNoFilter[_]] should "correctly index a text column" in {
    val stringIndexer = new OpStringIndexerNoFilter[Text]().setInput(txtF)
    val indices = stringIndexer.fit(ds).transform(ds).collect(stringIndexer.getOutput())

    indices shouldBe expected
  }

  it should "correctly index a text column (shortcut)" in {
    val indexed = txtF.indexed()
    val indices = indexed.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(ds).transform(ds).collect(indexed)
    indices shouldBe expected

    val indexed2 = txtF.indexed(handleInvalid = Skip)
    val indicesfit = indexed2.originStage.asInstanceOf[OpStringIndexer[_]].fit(ds)
    val indices2 = indicesfit.transform(ds).collect(indexed2)
    val indices3 = indicesfit.asInstanceOf[SwUnaryModel[Text, RealNN, StringIndexerModel]]
      .setInput(txtFNew).transform(dsNew).collect(indexed2)
    indices2 shouldBe expected
    indices3 shouldBe expected
  }

  it should "correctly deinxed a numeric column" in {
    val indexed = txtF.indexed()
    val indices = indexed.originStage.asInstanceOf[OpStringIndexerNoFilter[_]].fit(ds).transform(ds)
    val deindexed = indexed.deindexed()
    val deindexedData = deindexed.originStage.asInstanceOf[OpIndexToStringNoFilter]
      .transform(indices).collect(deindexed)
    deindexedData shouldBe txtData
  }

  it should "assign new strings to the unseen string category" in {
    val stringIndexer = new OpStringIndexerNoFilter[Text]().setInput(txtF)
    val indices = stringIndexer.fit(ds).setInput(txtFNew).transform(dsNew).collect(stringIndexer.getOutput())

    indices shouldBe expectedNew
  }

  Spec[OpStringIndexer[_]] should "correctly index a text column" in {
    val stringIndexer = new OpStringIndexer[Text]().setInput(txtF)
    val indices = stringIndexer.fit(ds).transform(ds).collect(stringIndexer.getOutput())

    indices shouldBe expected
  }

  it should "correctly deinxed a numeric column" in {
    val indexedStage = new OpStringIndexer[Text]().setInput(txtF)
    val indexed = indexedStage.getOutput()
    val indices = indexedStage.fit(ds).transform(ds)
    val deindexedStage = new OpIndexToString().setInput(indexed)
    val deindexed = deindexedStage.getOutput()
    val deindexedData = deindexedStage.transform(indices).collect(deindexed)
    deindexedData shouldBe txtData
  }

}
