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
import com.salesforce.op.stages.base.sequence.SequenceTransformer
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.{FeatureHistory, UID}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, Params}
import org.joda.time.{DateTime => JDateTime, DateTimeZone}

import scala.reflect.runtime.universe.TypeTag

trait DateToUnitCircleParams extends Params {

  final val timePeriod: Param[String] = new Param[String](parent = this,
    name = "timePeriods",
    doc = "The time period to extract from the timestamp",
    isValid = (value: String) => TimePeriod.values.map(_.entryName).contains(value)
  )

  setDefault(timePeriod, TimePeriod.HourOfDay.entryName)

  /** @group setParam */
  final def getTimePeriod: TimePeriod = TimePeriod.withName($(timePeriod))

  final def setTimePeriod(value: TimePeriod): this.type = set(timePeriod, value.entryName)
}

/**
 * Following: http://webspace.ship.edu/pgmarr/geo441/lectures/lec%2016%20-%20directional%20statistics.pdf
 * Transforms a Date or DateTime field into a cartesian coordinate representation
 * of an extracted time period on the unit circle
 *
 * parameter timePeriod The time period to extract from the timestamp
 * enum from: DayOfMonth, DayOfWeek, DayOfYear, HourOfDay, MonthOfYear, WeekOfMonth, WeekOfYear
 *
 * We extract the timePeriod from the timestamp and
 * map this onto the unit circle containing the number of time periods equally spaced.
 * For example, when timePeriod = HourOfDay, the timestamp 01/01/2018 6:37 maps to the point on the circle with
 * angle radians = 2*math.Pi*6/24
 * We return the cartesian coordinates of this point: (math.cos(radians), math.sin(radians))
 *
 * The first time period always has angle 0.
 *
 * Note: We use the ISO week date format https://en.wikipedia.org/wiki/ISO_week_date#First_week
 * Monday is the first day of the week
 * & the first week of the year is the week wit the first Monday after Jan 1.
 */
class DateToUnitCircleTransformer[T <: Date]
(
  uid: String = UID[DateToUnitCircleTransformer[_]]
)(implicit tti: TypeTag[T], val ttiv: TypeTag[T#Value]) extends SequenceTransformer[T, OPVector](
  operationName = "dateToUnitCircle",
  uid = uid
) with DateToUnitCircleParams {

  override def transformFn: Seq[T] => OPVector = timestamp => {
    val randians = timestamp.flatMap(ts => DateToUnitCircle.convertToRandians(ts.v, getTimePeriod)).toArray
    Vectors.dense(randians).toOPVector
  }

  override def onGetMetadata(): Unit = {
    super.onGetMetadata()
    val timePeriod = getTimePeriod
    val columns = inN.flatMap{
      f => DateToUnitCircle.metadataValues(timePeriod)
        .map(iv => f.toColumnMetaData().copy(descriptorValue = Option(iv)))
    }
    val history = inN.flatMap(f => Seq(f.name -> FeatureHistory(originFeatures = f.originFeatures, stages = f.stages)))
    setMetadata(OpVectorMetadata(getOutputFeatureName, columns, history.toMap).toMetadata)
  }
}

private[op] object DateToUnitCircle {

  def metadataValues(timePeriod: TimePeriod): Seq[String] = Seq(s"x_$timePeriod", s"y_$timePeriod")

  def convertToBin(timestamp: Long, timePeriodDesired: TimePeriod): Double =
    getPeriodWithSize(timestamp, timePeriodDesired)._1

  def convertToRandians(timestamp: Option[Long], timePeriodDesired: TimePeriod): Array[Double] =
    timestamp.map { ts =>
      val (timePeriod, periodSize) = getPeriodWithSize(ts, timePeriodDesired)
      val radians = (2 * math.Pi * timePeriod) / periodSize
      Array(math.cos(radians), math.sin(radians))
    }.getOrElse(Array(0.0, 0.0))

  private def getPeriodWithSize(timestamp: Long, timePeriod: TimePeriod): (Double, Int) = {
    val dt = new JDateTime(timestamp).withZone(DateTimeZone.UTC)
    timePeriod match {
      case TimePeriod.DayOfMonth => (dt.dayOfMonth.get.toDouble - 1, 31)
      case TimePeriod.DayOfWeek => (dt.dayOfWeek.get.toDouble - 1, 7)
      case TimePeriod.DayOfYear => (dt.dayOfYear.get.toDouble - 1, 366)
      case TimePeriod.HourOfDay => (dt.hourOfDay.get.toDouble, 24)
      case TimePeriod.MonthOfYear => (dt.monthOfYear.get.toDouble - 1, 12)
      case TimePeriod.WeekOfMonth =>
        ((dt.weekOfWeekyear.get - dt.withDayOfMonth(1).weekOfWeekyear.get).toDouble, 6)
      case TimePeriod.WeekOfYear => (dt.weekOfWeekyear.get.toDouble - 1, 53)
    }
  }
}
