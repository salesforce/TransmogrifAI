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

import com.salesforce.op.utils.date.DateTimeUtils
import enumeratum.{Enum, EnumEntry}
import org.joda.time.{DateTime => JDateTime}

case class TimePeriodVal(value: Int, min: Int, max: Int)

sealed abstract class TimePeriod(extractFn: JDateTime => TimePeriodVal) extends EnumEntry with Serializable {
  def extractTimePeriodVal: Long => TimePeriodVal =
    ((millis: Long) => new JDateTime(millis, DateTimeUtils.DefaultTimeZone))
      .andThen(extractFn)

  def extractIntFromMillis: Long => Int = extractTimePeriodVal.andThen((x: TimePeriodVal) => x.value)
}

object TimePeriod extends Enum[TimePeriod] {
  val values: Seq[TimePeriod] = findValues
  case object DayOfMonth extends TimePeriod(dt => TimePeriodVal(dt.dayOfMonth.get, 1, 31))
  case object DayOfWeek extends TimePeriod(dt => TimePeriodVal(dt.dayOfWeek.get, 1, 7))
  case object DayOfYear extends TimePeriod(dt => TimePeriodVal(dt.dayOfYear.get, 1, 366))
  case object HourOfDay extends TimePeriod(dt => TimePeriodVal(dt.hourOfDay.get, 0, 24))
  case object MonthOfYear extends TimePeriod(dt => TimePeriodVal(dt.monthOfYear.get, 1, 12))
  case object WeekOfMonth extends TimePeriod(dt =>
    // case 1) if the first day of the month is the first day of the week
    //         ceil(dayOfMonth / 7.0)
    // case 2) otherwise week of month is the same as when the first of the month is moved back
    //         to the first day of the week, applying case 1
    TimePeriodVal(
      math.ceil((dt.dayOfMonth.get + (dt.withDayOfMonth(1).dayOfWeek.get - 1)) / 7.0)
        .toInt, 1, 6))
  case object WeekOfYear extends TimePeriod(dt => TimePeriodVal(dt.weekOfWeekyear.get, 1, 53))
}
