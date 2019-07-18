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

package com.salesforce.op.utils.json

import com.salesforce.op.test.TestCommon
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, Extraction, Formats}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class SpecialDoubleSerializerTest extends FlatSpec with TestCommon {

  val data = Map(
    "normal" -> Seq(-1.1, 0.0, 2.3),
    "infs" -> Seq(Double.NegativeInfinity, Double.PositiveInfinity),
    "minMax" -> Seq(Double.MinValue, Double.MaxValue),
    "nan" -> Seq(Double.NaN)
  )

  Spec[SpecialDoubleSerializer] should behave like
    readWriteDoubleValues(data)(
      json = """{"normal":[-1.1,0.0,2.3],"infs":["-Infinity","Infinity"],"minMax":[-1.7976931348623157E308,1.7976931348623157E308],"nan":["NaN"]}""" // scalastyle:off
    )(DefaultFormats + new SpecialDoubleSerializer)

  Spec[SpecialDoubleSerializer] + " (with big decimal)" should behave like
    readWriteDoubleValues(data)(
      json = """{"normal":[-1.1,0.0,2.3],"infs":["-Infinity","Infinity"],"minMax":[-1.7976931348623157E+308,1.7976931348623157E+308],"nan":["NaN"]}""" // scalastyle:off
    )(DefaultFormats.withBigDecimal + new SpecialDoubleSerializer)


  def readWriteDoubleValues(input: Map[String, Seq[Double]])(json: String)(implicit formats: Formats): Unit = {
    it should "write double entries" in {
      compact(Extraction.decompose(input)) shouldBe json
    }
    it should "read double entries" in {
      val parsed = parse(json).extract[Map[String, Seq[Double]]]
      parsed.keys shouldBe input.keys
      parsed zip input foreach {
        case (("nan", a), ("nan", b)) => a.foreach(_.isNaN shouldBe true)
        case ((_, a), (_, b)) => a should contain theSameElementsAs b
      }
    }
  }
}
