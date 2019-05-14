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

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.date.DateTimeUtils
import org.joda.time.{DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TimePeriodMapTransformerTest extends OpTransformerSpec[IntegralMap, TimePeriodMapTransformer[DateMap]] {

  val names: Seq[String] = Seq("n1", "n2", "n3", "n4")

  val dates: Seq[Long] = Seq(
    new JDateTime(1879, 3, 14, 0, 0, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(1955, 11, 12, 10, 4, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(1999, 3, 8, 12, 0, DateTimeUtils.DefaultTimeZone).getMillis,
    new JDateTime(2019, 4, 30, 13, 0, DateTimeUtils.DefaultTimeZone).getMillis
  )

  val dateMap: DateMap = names.zip(dates).toMap.toDateMap

  val (inputData, f1) = TestFeatureBuilder(Seq[DateMap](dateMap))

  override val transformer: TimePeriodMapTransformer[DateMap] = new TimePeriodMapTransformer(TimePeriod.DayOfMonth).setInput(f1)

  override val expectedResult: Seq[IntegralMap] = Seq(
    names.zip(Seq(14L, 12L, 8L, 30L)).toMap.toIntegralMap
  )
}
