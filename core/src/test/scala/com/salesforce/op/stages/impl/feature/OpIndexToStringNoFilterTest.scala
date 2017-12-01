/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class OpIndexToStringNoFilterTest extends FlatSpec with TestSparkContext {

  val (ds, indF) = TestFeatureBuilder(Seq(0.0, 2.0, 1.0, 0.0, 0.0, 1.0).map(_.toRealNN))
  val labels = Array("a", "c", "b")
  val expected = Array("a", "b", "c", "a", "a", "c").map(_.toText)

  val labelsNew = Array("a", "c")
  val expectedNew = Array("a", OpIndexToStringNoFilter.unseenDefault, "c", "a", "a", "c").map(_.toText)

  Spec[OpIndexToStringNoFilter] should "correctly deindex a numeric column" in {
    val indexToStr = new OpIndexToStringNoFilter().setInput(indF).setLabels(labels)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expected
  }

  it should "correctly deindex a numeric column (shortcut)" in {
    val str = indF.deindexed(labels)
    val strs = str.originStage.asInstanceOf[OpIndexToStringNoFilter].transform(ds).collect(str)
    strs shouldBe expected

    val str2 = indF.deindexed(labels, handleInvalid = IndexToStringHandleInvalid.Error)
    val strs2 = str2.originStage.asInstanceOf[OpIndexToString].transform(ds).collect(str2)
    strs2 shouldBe expected
  }

  it should "correctly deindex even if the lables list does not match the number of indicies" in {
    val indexToStr = new OpIndexToStringNoFilter().setInput(indF).setLabels(labelsNew)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expectedNew
  }

  Spec[OpIndexToString] should "correctly deindex a numeric column" in {
    val indexToStr = new OpIndexToString().setInput(indF).setLabels(labels)
    val strs = indexToStr.transform(ds).collect(indexToStr.getOutput())

    strs shouldBe expected
  }
}
