/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.utils.date.DateTimeUtils
import org.joda.time.format.DateTimeFormat

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
    timeMs = Some(DateTimeUtils.now().withTimeAtStartOfDay().minusDays(daysAgo).getMillis)
  )

  def WeeksAgo(weeksAgo: Int): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.WeeksAgo,
    timeMs = Some(DateTimeUtils.now().withTimeAtStartOfDay().minusWeeks(weeksAgo).getMillis)
  )

  def DDMMYYYY(ddMMyyyy: String): CutOffTime = CutOffTime(
    cType = CutOffTimeTypes.DDMMYYYY,
    timeMs = Some(
      DateTimeFormat.forPattern("ddMMyyyy").parseDateTime(ddMMyyyy).withZone(DateTimeUtils.DefaultTimeZone).getMillis
    )
  )

  def NoCutoff(): CutOffTime = CutOffTime(cType = CutOffTimeTypes.NoCutoff, timeMs = None)
  // scalastyle:on

}
