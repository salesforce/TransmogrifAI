/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.spark.SequenceAggregators.{SeqMapDouble, SeqMapLong}
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SequenceAggregatorsTest extends FlatSpec with TestSparkContext {
  import spark.implicits._
  val meanSeqMapDouble = SequenceAggregators.MeanSeqMapDouble(2)
  val modeSeqMapLong = SequenceAggregators.ModeSeqMapLong(2)
  val modeSeqNullInt = SequenceAggregators.ModeSeqNullInt(2)
  implicit val encMeanMapAgg = meanSeqMapDouble.outputEncoder
  implicit val encModeMapAgg = modeSeqMapLong.outputEncoder

  Spec(SequenceAggregators.getClass) should "correctly compute the mean by key of maps" in {
    val data = Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("z" -> 10.0)),
      (Map("c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map.empty[String, Double], Map.empty[String, Double])
    ).toDF("f1", "f2").map(Helper.toSeqMapDouble)

    val res = data.select(meanSeqMapDouble.toColumn).first()
    res shouldBe Seq(Map("a" -> 1.0, "c" -> 11.0, "b" -> 5.0), Map("z" -> 10.0, "y" -> 3.0, "x" -> 0.0))
  }

  it should "correctly compute the mean by key of maps again" in {
    val meanData = Seq(
      (Map("a" -> 1.0, "b" -> 5.0), Map("y" -> 4.0, "x" -> 0.0, "z" -> 10.0)),
      (Map("a" -> -3.0, "b" -> 3.0, "c" -> 11.0), Map("y" -> 3.0, "x" -> 0.0)),
      (Map("a" -> 1.0, "b" -> 5.0), Map("y" -> 1.0, "x" -> 0.0, "z" -> 5.0))
    ).toDF("f1", "f2").map(Helper.toSeqMapDouble)

    val res = meanData.select(meanSeqMapDouble.toColumn).first()
    res shouldBe Seq(Map("a" -> -1.0 / 3, "c" -> 11.0, "b" -> 13.0 / 3), Map("z" -> 7.5, "y" -> 8.0 / 3, "x" -> 0.0))
  }

  it should "correctly compute the mode by key of maps" in {
    val data = Seq(
      (Map("a" -> 1L, "b" -> 5L), Map("z" -> 10L)),
      (Map("c" -> 11L), Map("y" -> 3L, "x" -> 0L)),
      (Map.empty[String, Long], Map.empty[String, Long])
    ).toDF("f1", "f2").map(Helper.toSeqMapLong)

    val res = data.select(modeSeqMapLong.toColumn).first()
    res shouldBe Seq(Map("a" -> 1L, "b" -> 5L, "c" -> 11L), Map("x" -> 0L, "y" -> 3L, "z" -> 10L))
  }

  it should "correctly compute the mode by key of maps again" in {
    val modeData = Seq(
      (Map("a" -> 1L, "b" -> 5L), Map("y" -> 4L, "x" -> 0L, "z" -> 10L)),
      (Map("a" -> -3L, "b" -> 3L, "c" -> 11L), Map("y" -> 3L, "x" -> 0L)),
      (Map("a" -> 1L, "b" -> 5L), Map("y" -> 1L, "x" -> 0L, "z" -> 5L))
    ).toDF("f1", "f2").map(Helper.toSeqMapLong)

    val res = modeData.select(modeSeqMapLong.toColumn).first()
    res shouldBe Seq(Map("a" -> 1L, "b" -> 5L, "c" -> 11L), Map("x" -> 0L, "y" -> 1L, "z" -> 5L))
  }

  it should "correctly compute the mode" in {
    val data = Seq(
      (Some(3L), None),
      (Some(3L), Some(2L)),
      (Some(1L), Some(5L))
    ).toDF("f1", "f2").map(r => Seq(if (r.isNullAt(0)) None else Option(r.getLong(0)),
      if (r.isNullAt(1)) None else Option(r.getLong(1))))

    val res = data.select(modeSeqNullInt.toColumn).first()
    res shouldBe Seq(3L, 2L)
  }
}

private object Helper {
  def toSeqMapDouble(r: Row): SeqMapDouble = Seq(r.getMap[String, Double](0).toMap, r.getMap[String, Double](1).toMap)
  def toSeqMapLong(r: Row): SeqMapLong = Seq(r.getMap[String, Long](0).toMap, r.getMap[String, Long](1).toMap)
}
