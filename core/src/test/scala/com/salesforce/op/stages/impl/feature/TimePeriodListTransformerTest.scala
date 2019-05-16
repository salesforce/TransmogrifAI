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
class TimePeriodListTransformerTest extends OpTransformerSpec[OPVector, TimePeriodListTransformer[DateList]] {

  val dateList: DateList = Seq[Long](
    new JDateTime(1879, 3, 14, 0, 0, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(1955, 11, 12, 10, 4, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(1999, 3, 8, 12, 0, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(2019, 4, 30, 13, 0, DateTimeUtils.DefaultTimeZone).getMillis
  ).toDateList

  val (inputData, f1) = TestFeatureBuilder(Seq(dateList))

  override val transformer: TimePeriodListTransformer[DateList] =
    new TimePeriodListTransformer(TimePeriod.DayOfMonth).setInput(f1)

  override val expectedResult: Seq[OPVector] = Seq(Seq(14, 12, 8, 30).map(_.toDouble).toVector.toOPVector)

  it should "transform with rich shortcuts" in {
    val dlist = List(new JDateTime(1879, 3, 14, 0, 0, DateTimeUtils.DefaultTimeZone).getMillis)
    val (inputData2, d1, d2) = TestFeatureBuilder(
      Seq[(DateList, DateTimeList)]((dlist.toDateList, dlist.toDateTimeList))
    )

    def assertFeature(feature: FeatureLike[OPVector], expected: Seq[OPVector]): Unit = {
      val transformed = feature.originStage.asInstanceOf[Transformer].transform(inputData2)
      val actual = transformed.collect(feature)
      actual shouldBe expected
    }

    assertFeature(d1.toTimePeriod(TimePeriod.DayOfMonth), Seq(Vector(14.0).toOPVector))
    assertFeature(d2.toTimePeriod(TimePeriod.DayOfMonth), Seq(Vector(14.0).toOPVector))
  }
}
