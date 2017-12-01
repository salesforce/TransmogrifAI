/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json

import com.salesforce.op.test.TestCommon
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Extraction}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SpecialDoubleSerializerTest extends FlatSpec with TestCommon {

  implicit val formats = DefaultFormats + new SpecialDoubleSerializer

  val data = Map(
    "normal" -> Seq(-1.1, 0.0, 2.3),
    "infs" -> Seq(Double.NegativeInfinity, Double.PositiveInfinity),
    "minMax" -> Seq(Double.MinValue, Double.MaxValue),
    "nan" -> Seq(Double.NaN)
  )

  val dataJson = """{"normal":[-1.1,0.0,2.3],"infs":["-Infinity","Infinity"],"minMax":[-1.7976931348623157E308,1.7976931348623157E308],"nan":["NaN"]}""" // scalastyle:off

  Spec[SpecialDoubleSerializer] should "write double entries" in {
    compact(Extraction.decompose(data)) shouldBe dataJson
  }
  it should "read double entries" in {
    val parsed = parse(dataJson).extract[Map[String, Seq[Double]]]
    parsed.keys shouldBe data.keys

    parsed zip data foreach {
      case (("nan", a), ("nan", b)) => a.foreach(_.isNaN shouldBe true)
      case ((_, a), (_, b)) => a should contain theSameElementsAs b
    }
  }
}
