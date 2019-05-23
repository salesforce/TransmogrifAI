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

package com.salesforce.op.aggregators

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

import com.salesforce.op.utils.date.DateTimeUtils

/**
 * A cut off time to be used for aggregating features extracted from the events
 *
 * @param cType  cut off type
 * @param timeMs cut off time value in millis
 */
case class CutOffTime(cType: CutOffTimeType, timeMs: Option[Long])

object CutOffTime {

  // scalastyle:off
  def UnixEpoch(sinceEpoch: Long): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.UnixEpoch,
    timeMs = Some(math.max(sinceEpoch, 0L))
  )

  def DaysAgo(daysAgo: Int): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.DaysAgo,
    timeMs = Some(DateTimeUtils.getMillis(DateTimeUtils.now().minusDays(daysAgo).toLocalDate.atStartOfDay))
  )

  def WeeksAgo(weeksAgo: Int): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.WeeksAgo,
    timeMs = Some(DateTimeUtils.getMillis(DateTimeUtils.now().minusWeeks(weeksAgo).toLocalDate.atStartOfDay))
  )

  val format = DateTimeFormatter.ofPattern("ddMMyyyy", Locale.ENGLISH)
  def DDMMYYYY(ddMMyyyy: String): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.DDMMYYYY,
    timeMs = Some(DateTimeUtils.getMillis(LocalDateTime.parse(ddMMyyyy, format))
    )
  )

  def NoCutoff(): CutOffTime = CutOffTime(cType = CutOffTimeTypes.NoCutoff, timeMs = None)
  // scalastyle:on

}
