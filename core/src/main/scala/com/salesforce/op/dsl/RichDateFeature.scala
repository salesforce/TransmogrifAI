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

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.{DateListPivot, DateToUnitCircleTransformer, TimePeriod, TransmogrifierDefaults}
import org.joda.time.{DateTime => JDateTime}


trait RichDateFeature {
  self: RichFeature with RichListFeature =>

  /**
   * Enrichment functions for Date Feature
   *
   * @param f Date Feature
   */
  implicit class RichDateFeature(val f: FeatureLike[Date]) extends TimePeriodTransformers[Date] {

    /**
     * Convert to DateList feature
     * @return
     */
    def toDateList(): FeatureLike[DateList] = {
      f.transformWith(
        new UnaryLambdaTransformer[Date, DateList](operationName = "dateToList", _.value.toSeq.toDateList)
      )
    }

    /**
     * transforms a Date field into a cartesian coordinate representation
     * of an extracted time period on the unit circle
     *
     * @param timePeriod The time period to extract from the timestamp
     * @param others     Other features of same type
     * enum from: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay, WeekOfMonth, WeekOfYear
     */
    def toUnitCircle
    (
      timePeriod: TimePeriod = TimePeriod.HourOfDay,
      others: Array[FeatureLike[Date]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateToUnitCircleTransformer[Date]().setTimePeriod(timePeriod).setInput(f +: others).getOutput()
    }

    /**
     * Converts DateTime features into cartesian coordinate representation of an extracted time periods
     * (specified in circularDateRepresentations as seq of: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay,
     * WeekOfMonth, WeekOfYear) on the unit circle.
     * Also converts a sequence of Date features into DateList feature and then applies DateList vectorizer.
     *
     * DateListPivot can specify:
     * 1) SinceFirst - replace the feature by the number of days between the first event and reference date
     * 2) SinceLast - replace the feature by the number of days between the last event and reference date
     * 3) ModeDay - replace the feature by a pivot that indicates the mode of the day of the week
     * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
     * 4) ModeMonth - replace the feature by a pivot that indicates the mode of the month
     * 5) ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day.
     *
     * @param others        other features of same type
     * @param dateListPivot name of the pivot type from [[DateListPivot]] enum
     * @param referenceDate reference date to compare against when [[DateListPivot]] is [[SinceFirst]] or [[SinceLast]]
     * @param trackNulls    option to keep track of values that were missing
     * @param circularDateReps list of all the circular date representations that should be included in feature vector
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: JDateTime = TransmogrifierDefaults.ReferenceDate,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      circularDateReps: Seq[TimePeriod] = TransmogrifierDefaults.CircularDateRepresentations,
      others: Array[FeatureLike[Date]] = Array.empty
    ): FeatureLike[OPVector] = {
      val timePeriods = circularDateReps.map(tp => f.toUnitCircle(tp, others))
      val time = f.toDateList().vectorize(dateListPivot = dateListPivot, referenceDate = referenceDate,
        trackNulls = trackNulls, others = others.map(_.toDateList()))
      if (timePeriods.isEmpty) time else (timePeriods :+ time).combine()
    }

  }

  /**
   * Enrichment functions for DateTime Feature
   *
   * @param f DateTime Feature
   */
  implicit class RichDateTimeFeature(val f: FeatureLike[DateTime]) extends TimePeriodTransformers[DateTime] {

    /**
     * Convert to DateTimeList feature
     * @return
     */
    def toDateTimeList(): FeatureLike[DateTimeList] = {
      f.transformWith(
        new UnaryLambdaTransformer[DateTime, DateTimeList](
          operationName = "dateTimeToList",
          _.value.toSeq.toDateTimeList
        )
      )
    }

    /**
     * transforms a DateTime field into a cartesian coordinate representation
     * of an extracted time period on the unit circle
     *
     * @param timePeriod The time period to extract from the timestamp
     * @param others     Other features of same type
     * enum from: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay, WeekOfMonth, WeekOfYear
     */
    def toUnitCircle(
      timePeriod: TimePeriod = TimePeriod.HourOfDay,
      others: Array[FeatureLike[DateTime]] = Array.empty
    ): FeatureLike[OPVector] = {
      new DateToUnitCircleTransformer[DateTime]().setTimePeriod(timePeriod).setInput(f +: others).getOutput()
    }

    /**
     * Converts DateTime features into cartesian coordinate representation of an extracted time periods
     * (specified in circularDateRepresentations as seq of: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay,
     * WeekOfMonth, WeekOfYear) on the unit circle.
     * Also converts a sequence of DateTime features into DateTimeList feature and then applies DateTimeList vectorizer.
     *
     * DateListPivot can specify:
     * 1) SinceFirst - replace the feature by the number of days between the first event and reference date
     * 2) SinceLast - replace the feature by the number of days between the last event and reference date
     * 3) ModeDay - replace the feature by a pivot that indicates the mode of the day of the week
     * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
     * 4) ModeMonth - replace the feature by a pivot that indicates the mode of the month
     * 5) ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day.
     *
     * @param others        other features of same type
     * @param dateListPivot name of the pivot type from [[DateListPivot]] enum
     * @param referenceDate reference date to compare against when [[DateListPivot]] is [[SinceFirst]] or [[SinceLast]]
     * @param trackNulls    option to keep track of values that were missing
     * @param circularDateReps list of all the circular date representations that should be included in feature vector
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: JDateTime = TransmogrifierDefaults.ReferenceDate,
      trackNulls: Boolean = TransmogrifierDefaults.TrackNulls,
      circularDateReps: Seq[TimePeriod] = TransmogrifierDefaults.CircularDateRepresentations,
      others: Array[FeatureLike[DateTime]] = Array.empty
    ): FeatureLike[OPVector] = {
      val timePeriods = circularDateReps.map(tp => f.toUnitCircle(tp, others))
      val time = f.toDateTimeList().vectorize(dateListPivot = dateListPivot, referenceDate = referenceDate,
        trackNulls = trackNulls, others = others.map(_.toDateTimeList()))
      if (timePeriods.isEmpty) time else (timePeriods :+ time).combine()
    }

  }

}

trait TimePeriodTransformers[T <: Date] {
  val f: FeatureLike[T]

  /**
   * Converts a Date or DateTime feature into a selected time period.
   *
   * @param period type of [[TimePeriod]] to convert date feature to
   * @return Integer value of time period
   */
  def toTimePeriod(period: TimePeriod): FeatureLike[Integral] = {
    val periodFun: Long => Int = period match {
      case TimePeriod.DayOfMonth => t => new JDateTime(t).dayOfMonth.get
      case TimePeriod.DayOfWeek => t => new JDateTime(t).dayOfWeek.get
      case TimePeriod.DayOfYear => t => new JDateTime(t).dayOfYear.get
      case TimePeriod.HourOfDay => t => new JDateTime(t).hourOfDay.get
      case TimePeriod.MonthOfYear => t => new JDateTime(t).monthOfYear.get
      case TimePeriod.WeekOfMonth => t => {
        val dt = new JDateTime(t)
        dt.weekOfWeekyear.get - dt.withDayOfMonth(1).weekOfWeekyear.get
      }
      case TimePeriod.WeekOfYear => t => new JDateTime(t).weekyear.get
    }
    f.transformWith(
      new UnaryLambdaTransformer[T, Integral](operationName = "dateToTimePeriod",
        transformFn = _.value.map(t => periodFun(t).toLong).toIntegral)
    )
  }

  def toDayOfMonth(): FeatureLike[Integral] = toTimePeriod(TimePeriod.DayOfMonth)
  def toDayOfWeek(): FeatureLike[Integral] = toTimePeriod(TimePeriod.DayOfWeek)
  def toDayOfYear(): FeatureLike[Integral] = toTimePeriod(TimePeriod.DayOfYear)
  def toHourOfDay(): FeatureLike[Integral] = toTimePeriod(TimePeriod.HourOfDay)
  def toMonthOfYear(): FeatureLike[Integral] = toTimePeriod(TimePeriod.MonthOfYear)
  def toWeekOfMonth(): FeatureLike[Integral] = toTimePeriod(TimePeriod.WeekOfMonth)
  def toWeekOfYear(): FeatureLike[Integral] = toTimePeriod(TimePeriod.WeekOfYear)
}