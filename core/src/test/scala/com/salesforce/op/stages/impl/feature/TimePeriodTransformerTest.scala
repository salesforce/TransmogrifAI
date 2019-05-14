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

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.joda.time.{DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TimePeriodTransformerTest extends OpTransformerSpec[Integral, TimePeriodTransformer[Date]] {

  val (inputData, f1) = TestFeatureBuilder(Seq[Date](
    new JDateTime(1879, 3, 14, 0, 0, DateTimeUtils.DefaultTimeZone).getMillis.toDate,
    new JDateTime(1955, 11, 12, 10, 4, DateTimeUtils.DefaultTimeZone).getMillis.toDate,
    new JDateTime(1999, 3, 8, 12, 0, DateTimeUtils.DefaultTimeZone).getMillis.toDate,
    Date.empty,
    new JDateTime(2019, 4, 30, 13, 0, DateTimeUtils.DefaultTimeZone).getMillis.toDate
  ))

  override val transformer: TimePeriodTransformer[Date] = new TimePeriodTransformer(TimePeriod.DayOfMonth).setInput(f1)

  override val expectedResult: Seq[Integral] =
    Seq(Integral(14), Integral(12), Integral(8), Integral.empty, Integral(30))

  it should "correctly transform for all TimePeriod types" in {
    def assertFeature(feature: FeatureLike[Integral], expected: Seq[Integral]): Unit = {
      val transformed = feature.originStage.asInstanceOf[Transformer].transform(inputData)
      val actual = transformed.collect(feature)
      actual shouldBe expected
    }

    TimePeriod.values.foreach(tp => {
      val expected = tp match {
        case TimePeriod.DayOfMonth => Array(Integral(14), Integral(12), Integral(8), Integral.empty, Integral(30))
        case TimePeriod.DayOfWeek => Array(Integral(5), Integral(6), Integral(1), Integral.empty, Integral(2))
        case TimePeriod.DayOfYear => Array(Integral(73), Integral(316), Integral(67), Integral.empty, Integral(120))
        case TimePeriod.HourOfDay => Array(Integral(0), Integral(10), Integral(12), Integral.empty, Integral(13))
        case TimePeriod.MonthOfYear => Array(Integral(3), Integral(11), Integral(3), Integral.empty, Integral(4))
        case TimePeriod.WeekOfMonth => Array(Integral(2), Integral(1), Integral(1), Integral.empty, Integral(4))
        case TimePeriod.WeekOfYear => Array(Integral(11), Integral(45), Integral(10), Integral.empty, Integral(18))
        case _ => throw new Exception(s"Unexpected TimePeriod encountered, $tp")
      }

      withClue(s"Assertion failed for TimePeriod $tp: ") {
        assertFeature(f1.toTimePeriod(tp), expected)
      }
    })
  }
}
