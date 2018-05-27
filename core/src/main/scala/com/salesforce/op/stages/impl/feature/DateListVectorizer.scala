/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import java.text.DateFormatSymbols

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.stages.impl.feature.VectorizerUtils._
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.OpVectorColumnMetadata
import enumeratum._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, LongParam, ParamValidators}
import org.joda.time.{DateTime, DateTimeConstants, Days}

import scala.reflect.runtime.universe._


sealed trait DateListPivot extends EnumEntry with Serializable

/**
 * Enumeration object that contains the option to pivot the DateList feature
 *
 * 1) SinceFirst - replace the feature by the number of days between the first event and reference date
 *
 * 2) SinceLast - replace the feature by the number of days between the last event and reference date
 *
 * 3) ModeDay - replace the feature by a pivot that indicates the mode of the day of the week
 * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
 *
 * 4) ModeMonth - replace the feature by a pivot that indicates the mode of the month
 *
 * 5) ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day.
 */
object DateListPivot extends Enum[DateListPivot] {
  val values = findValues

  /**
   * SinceFirst - replace the feature by the number of days between the first event and reference date
   */
  case object SinceFirst extends DateListPivot

  /**
   * SinceLast - replace the feature by the number of days between the last event and reference date
   */
  case object SinceLast extends DateListPivot

  /**
   * Replace the feature by a pivot that indicates the mode of the day of the week
   * Example : If the mode is Monday then it will return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
   */
  case object ModeDay extends DateListPivot

  /**
   * ModeMonth - replace the feature by a pivot that indicates the mode of the month
   */
  case object ModeMonth extends DateListPivot

  /**
   * ModeHour - replace the feature by a pivot that indicates the mode of the hour of the day
   */
  case object ModeHour extends DateListPivot

}


/**
 * Converts a sequence of DateLists features into a vector feature.
 * Can choose how to pivot the features
 *
 * @param uid uid for instance
 */
class DateListVectorizer[T <: OPList[Long]]
(
  uid: String = UID[DateListVectorizer[T]]
)(implicit tti: TypeTag[T]) extends SequenceTransformer[T, OPVector](operationName = "vecDateList", uid = uid)
  with VectorizerDefaults with TrackNullsParam {

  import DateListPivot._

  final val first = new BooleanParam(this,
    name = "first",
    doc = s"boolean to choose between $SinceFirst and $SinceLast"
  )
  setDefault(first, true)

  final val fillValue = new DoubleParam(this,
    name = "fillValue",
    doc = s"default value in case of an empty DateList in $SinceFirst or $SinceLast"
  )
  setDefault(fillValue, 0.0)

  def setFillValue(value: Double): this.type = set(fillValue, value)

  final val withTimeSince = new BooleanParam(this,
    name = "withTimeSince",
    doc =
      "option to transform a sequence of times to the number of days between the first or last event and reference date"
  )
  setDefault(withTimeSince, true)

  final val withModeDay = new BooleanParam(this,
    name = "fillWithPivotModeDay",
    doc = "option to pivot  each DateList regarding the mode of the day of the week"
  )
  setDefault(withModeDay, false)

  final val withModeMonth = new BooleanParam(this,
    name = "fillWithPivotModeMonth",
    doc = "option to pivot each DateList regarding the mode of the month"
  )
  setDefault(withModeMonth, false)

  final val withModeHour = new BooleanParam(this,
    name = "fillWithPivotModeHour",
    doc = "option to pivot each DateList regarding the mode of the hour"
  )
  setDefault(withModeHour, false)

  final val referenceDate = new LongParam(this,
    name = "referenceDate",
    doc = "reference date to compare against (the milliseconds from 1970-01-01T00:00:00Z in UTC)",
    isValid = ParamValidators.gtEq(0L)
  )
  setDefault(referenceDate, TransmogrifierDefaults.ReferenceDate.getMillis)

  def getReferenceDate(): DateTime = new DateTime($(referenceDate), DateTimeUtils.DefaultTimeZone)

  private def setAllFalse(): this.type = {
    set(first, false)
    set(withTimeSince, false)
    set(withModeDay, false)
    set(withModeMonth, false)
    set(withModeHour, false)
  }

  /**
   * Set the pivot for the DateList features
   *
   * @param dateListPivot way to pivot : SinceFirst, SinceLast, ModeDay, ModeMonth, ModeHour
   * @return this vectorizer with option set
   */
  def setPivot(dateListPivot: DateListPivot): this.type = {
    setAllFalse()
    dateListPivot match {
      case SinceFirst =>
        set(first, true)
        set(withTimeSince, true)
      case SinceLast =>
        set(first, false)
        set(withTimeSince, true)
      case ModeDay => set(withModeDay, true)
      case ModeHour => set(withModeHour, true)
      case ModeMonth => set(withModeMonth, true)
    }
  }

  /**
   * Set the reference date
   *
   * @param date reference date to compare against
   * @return this vectorizer with option set
   */
  def setReferenceDate(date: DateTime): this.type = {
    set(referenceDate, date.getMillis)
  }

  /**
   * Transforms a sequence of times to the number of days between the first event and reference date
   */
  private val timeAgoTransformFn: T => Seq[Double] = (dt: T) => {
    val days: Seq[Double] =
      if (dt.isEmpty) Seq($(fillValue))
      else {
        val compareDate = if ($(first)) dt.v.min else dt.v.max
        Seq(Days.daysBetween(new DateTime(compareDate, DateTimeUtils.DefaultTimeZone), getReferenceDate()).getDays)
      }
    if ($(trackNulls)) days :+ (dt.isEmpty : Double) else days
  }

  /**
   * Pivot each DateList regarding the mode of the day of the week"
   */
  private val modeDayTransformFn: T => Seq[Double] = (dt: T) => {
    val day: Seq[Double] =
      if (dt.isEmpty) List.fill(DateTimeConstants.DAYS_PER_WEEK)(0.0)
      else {
        val countDays =
          dt.v.map(d => new DateTime(d, DateTimeUtils.DefaultTimeZone).getDayOfWeek)
            .groupBy(identity).mapValues(_.size).toArray
        val modeDay = countDays.minBy { case (w, c) => (-c, w) }._1
        oneHot(modeDay - 1, DateTimeConstants.DAYS_PER_WEEK) // oneHot is zero based so subtracting one
      }
    if ($(trackNulls)) day :+ (dt.isEmpty : Double) else day
  }

  /**
   * Pivot each DateList regarding the mode of the month"
   */
  private val modeMonthTransformFn: T => Seq[Double] = (dt: T) => {
    val month: Seq[Double] =
      if (dt.isEmpty) List.fill(12)(0.0)
      else {
        val countMonths =
          dt.v.map(d => new DateTime(d, DateTimeUtils.DefaultTimeZone).getMonthOfYear)
            .groupBy(identity).mapValues(_.size).toArray
        val modeMonth = countMonths.minBy { case (m, c) => (-c, m) }._1
        oneHot(modeMonth - 1, 12) // oneHot is zero based so subtracting one
      }
    if ($(trackNulls)) month :+ (dt.isEmpty : Double) else month
  }


  /**
   * Pivot each DateList regarding the mode of the hour"
   */
  private val modeHourTransformFn: T => Seq[Double] = (dt: T) => {
    val hour: Seq[Double] =
      if (dt.isEmpty) List.fill(DateTimeConstants.HOURS_PER_DAY)(0.0)
      else {
        val countHours =
          dt.v.map(d => new DateTime(d, DateTimeUtils.DefaultTimeZone).getHourOfDay)
            .groupBy(identity).mapValues(_.size).toArray
        val modeHour = countHours.minBy { case (h, c) => (-c, h) }._1
        oneHot(modeHour, DateTimeConstants.HOURS_PER_DAY)
      }
    if ($(trackNulls)) hour :+ (dt.isEmpty : Double) else hour
  }


  override def onGetMetadata(): Unit = {
    val metaData =
      if ($(withTimeSince)) {
        if ($(trackNulls)) vectorMetadataWithNullIndicators else vectorMetadataFromInputFeatures
      } else {
        val vectorMeta = vectorMetadataFromInputFeatures
        val pivotNames =
          if ($(withModeDay)) {
            // Roll the Sequence to start with Monday
            val weekDays = new DateFormatSymbols().getWeekdays.filterNot(_.isEmpty).toSeq
            weekDays.tail ++ Seq(weekDays.head)
          } else if ($(withModeMonth)) {
            new DateFormatSymbols().getMonths.filterNot(_.isEmpty).toSeq
          } else {
            (0 until DateTimeConstants.HOURS_PER_DAY).map(hours => s"$hours:00")
          }
        val allPivotNames = if ($(trackNulls)) pivotNames :+ TransmogrifierDefaults.NullString else pivotNames
        val updatedCols = for {
          col <- vectorMeta.columns
          pivotValue <- allPivotNames
        } yield OpVectorColumnMetadata(
          parentFeatureName = col.parentFeatureName,
          parentFeatureType = col.parentFeatureType,
          indicatorGroup = col.parentFeatureName,
          indicatorValue = Option(pivotValue)
        )
        vectorMeta.withColumns(updatedCols)
      }
    setMetadata(metaData.toMetadata)
  }

  override def transformFn: Seq[T] => OPVector = (row: Seq[T]) => {
    if ($(withTimeSince)) {
      val replaced: Seq[Double] = row.flatMap(timeAgoTransformFn)
      Vectors.dense(replaced.toArray)
    } else {
      val inputSize = getInputFeatures().length
      val nullSize = if ($(trackNulls)) inputSize else 0
      val (replaced, vectorSize): (Seq[Double], Int) = if ($(withModeDay)) {
        (row.flatMap(modeDayTransformFn), DateTimeConstants.DAYS_PER_WEEK * inputSize + nullSize)
      } else if ($(withModeMonth)) {
        (row.flatMap(modeMonthTransformFn), 12 * inputSize + nullSize)
      } else (row.flatMap(modeHourTransformFn), DateTimeConstants.HOURS_PER_DAY * inputSize + nullSize)
      Vectors.sparse(vectorSize, (0 until vectorSize).toArray, replaced.toArray).compressed
    }
  }.toOPVector

}
