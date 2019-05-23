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

package com.salesforce.op.utils.date

import java.time.format.{DateTimeFormatter, DateTimeFormatterBuilder}
import java.time.temporal.ChronoUnit
import java.time.{DayOfWeek, Instant, LocalDateTime, ZoneId}
import java.util.Locale
import java.util.concurrent.TimeUnit

object DateTimeUtils {

  val DAYS_PER_WEEK = DayOfWeek.values().size
  val MILLIS_PER_HOUR = TimeUnit.HOURS.toMillis(1)
  val HOURS_PER_DAY = TimeUnit.DAYS.toHours(1).toInt
  val MILLIS_PER_DAY = TimeUnit.DAYS.toMillis(1)


  val DefaultTimeZoneStr = "UTC"
  val DefaultTimeZone = ZoneId.of(DefaultTimeZoneStr)

  /**
   * Get current time in default TZ
   *
   * @return
   */
  def now(timeZone: ZoneId = DefaultTimeZone): LocalDateTime = LocalDateTime.now(DefaultTimeZone)

  /**
   * Converts a date string with a specified time zone offset from "yyyy-MM-dd HH:mm:ss.SSS", "yyyy/MM/dd", "M/d/yyyy"
   * or ISO 8601 ( e.g. "1997-07-16T19:20:30.456") format to unix timestamp in milliseconds.
   *
   * @param date           date string in ISO 8601, "yyyy-MM-dd HH:mm:ss.SSS", "yyyy/MM/dd", "M/d/yyyy" formats
   * @param timeZoneString time zone of input date string, e.g., "GMT-5" or "GMT+5" or "US/Eastern"
   * @return unix timestamp in milliseconds
   */
  def parse(date: String, timeZoneString: String = DefaultTimeZoneStr): Long = {
    parseToDateTime(date, timeZoneString).atZone(ZoneId.of(timeZoneString)).toInstant.toEpochMilli
  }

  def parseToDateTime(date: String, timeZoneString: String = DefaultTimeZoneStr): LocalDateTime = {
    parseDateTime(date, ZoneId.of(timeZoneString))
  }


  val yyyyMMdd = DateTimeFormatter.ofPattern("yyyy/MM/dd", Locale.ENGLISH)
  val yyyyMMddHHmmssSSS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS", Locale.ENGLISH)
  val yyyyMMddHHmmss = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss", Locale.ENGLISH)
  val Mdyyyy = DateTimeFormatter.ofPattern("M/d/yyyy", Locale.ENGLISH)

  val formatter: DateTimeFormatter =
    new DateTimeFormatterBuilder()
      .appendOptional(yyyyMMddHHmmssSSS)
      .appendOptional(yyyyMMddHHmmss)
      .appendOptional(yyyyMMdd)
      .appendOptional(Mdyyyy)
      .appendOptional(DateTimeFormatter.ISO_OFFSET_DATE_TIME)
      .toFormatter(Locale.ENGLISH)

  /**
   * Converts a date string in ISO 8601 format ( e.g. "1997-07-16T19:20:30.456") or formatted as in
   * "yyyy-MM-dd HH:mm:ss.SSS", "yyyy/MM/dd", or "M/d/yyyy" to a DateTime object
   *
   * @param date     date string in ISO 8601, "yyyy-MM-dd HH:mm:ss.SSS", "yyyy/MM/dd", "M/d/yyyy" formats
   * @param timeZone time zone of input date string. E.g. "US/Eastern"
   * @return
   */
  def parseDateTime(date: String, timeZone: ZoneId = DefaultTimeZone): LocalDateTime = {
    LocalDateTime.parse(date, formatter.withZone(timeZone))
  }


  /**
   * Parses a unix timestamp in milliseconds to YYYY/MM/dd.
   *
   * @param timestampInMillis unix timestamp in milliseconds
   * @return YYYY/MM/dd
   */
  def parseUnix(timestampInMillis: Long): String = {
    val timestamp = LocalDateTime.ofInstant(Instant.ofEpochMilli(timestampInMillis), DefaultTimeZone)
    timestamp.format(yyyyMMdd)
  }

  def parseUnixToDateTime(timestampInMillis: Long): LocalDateTime = {
    LocalDateTime.ofInstant(Instant.ofEpochMilli(timestampInMillis), DefaultTimeZone)
  }

  /**
   * Get sequence of date strings between two dates
   *
   * @param startDate String YYYY/MM/dd
   * @param endDate   String YYYY/MM/dd
   * @return sequence of YYYY/MM/dd strings from the start to the end dates inclusive
   */
  def getRange(startDate: String, endDate: String): Seq[String] = {
    val start = parseDateTime(startDate, DefaultTimeZone)
    val end = parseDateTime(endDate, DefaultTimeZone)
    val days = ChronoUnit.DAYS.between(start, end).toInt
    (0 to days).map(d => start.plusDays(d).format(yyyyMMdd))
  }

  /**
   * Get date difference days from start date
   *
   * @param startDate  String YYYY/MM/dd
   * @param difference integer difference want date for
   * @return a YYYY/MM/dd string for the day difference days from the start
   */
  def getDatePlusDays(startDate: String, difference: Int): String = {
    val start = parseDateTime(startDate, DefaultTimeZone)
    start.plusDays(difference).format(yyyyMMdd)
  }

  /**
   * Convert local date time to epoch millis
   *
   * @param ldt local date time to convert
   * @return millis
   */
  def getMillis(ldt: LocalDateTime): Long = {
    ldt.atZone(DefaultTimeZone).toInstant.toEpochMilli
  }
}
