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


sealed abstract class TimePeriod extends EnumEntry with Serializable {
  def longToDateTime(t: Long): JDateTime = new JDateTime(t, DateTimeUtils.DefaultTimeZone)
  def extractFromTime(t: Long): Int
}

object TimePeriod extends Enum[TimePeriod] {
  val values: Seq[TimePeriod] = findValues
  case object DayOfMonth extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).dayOfMonth.get }
  case object DayOfWeek extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).dayOfWeek.get }
  case object DayOfYear extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).dayOfYear.get }
  case object HourOfDay extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).hourOfDay.get }
  case object MonthOfYear extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).monthOfYear.get }
  case object WeekOfMonth extends TimePeriod {
    def extractFromTime(t: Long): Int = {
      val dt = longToDateTime(t)
      dt.weekOfWeekyear.get - dt.withDayOfMonth(1).weekOfWeekyear.get
    }
  }
  case object WeekOfYear extends TimePeriod { def extractFromTime(t: Long): Int = longToDateTime(t).weekOfWeekyear.get }
}
