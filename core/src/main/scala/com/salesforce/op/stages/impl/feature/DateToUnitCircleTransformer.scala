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
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel, SequenceTransformer}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import com.salesforce.op.{FeatureHistory, UID}
import enumeratum.{Enum, EnumEntry}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.Dataset
import org.joda.time.{DateTime => JDateTime}

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
      f => DateToUnitCircle.metadataValues(timePeriod).map(iv => f.toColumnMetaData().copy(indicatorValue = Option(iv)))
    }
    val history = inN.flatMap(f => Seq(f.name -> FeatureHistory(originFeatures = f.originFeatures, stages = f.stages)))
    setMetadata(OpVectorMetadata(getOutputFeatureName, columns, history.toMap).toMetadata)
  }
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
class DateMapToUnitCircleVectorizer[T <: DateMap]
(
  uid: String = UID[DateMapToUnitCircleVectorizer[_]]
)(implicit tti: TypeTag[T], override val ttiv: TypeTag[T#Value]) extends SequenceEstimator[T, OPVector](
  operationName = "dateMapToUnitCircle",
  uid = uid
) with DateToUnitCircleParams with MapVectorizerFuns[Long, T]  {

  override def makeVectorMetadata(allKeys: Seq[Seq[String]]): OpVectorMetadata = {
    val meta = vectorMetadataFromInputFeatures
    val timePeriod = getTimePeriod

    val cols = for {
      (keys, col) <- allKeys.zip(meta.columns)
      key <- keys
      dec <- DateToUnitCircle.metadataValues(timePeriod)
    } yield new OpVectorColumnMetadata(
      parentFeatureName = col.parentFeatureName,
      parentFeatureType = col.parentFeatureType,
      indicatorGroup = Option(key),
      indicatorValue = Option(dec) // TODO fix this cause will break sanity checker
    )
    meta.withColumns(cols.toArray)
  }

  override def fitFn(dataset: Dataset[Seq[Map[String, Long]]]): SequenceModel[T, OPVector] = {
    val shouldClean = $(cleanKeys)
    val allKeys = getKeyValues(dataset, shouldClean, shouldCleanValues = false)

    val meta = makeVectorMetadata(allKeys)
    setMetadata(meta.toMetadata)
    new DateMapToUnitCircleVectorizerModel[T](allKeys = allKeys, shouldClean = shouldClean,
      timePeriodString = getTimePeriod.entryName, operationName = operationName, uid = uid
    )
  }

}

final class DateMapToUnitCircleVectorizerModel[T <: DateMap] private[op]
(
  val allKeys: Seq[Seq[String]],
  val shouldClean: Boolean,
  val timePeriodString: String,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T]) extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
  with CleanTextMapFun {

  private val timePeriod: TimePeriod = TimePeriod.withNameInsensitive(timePeriodString)

  override def transformFn: Seq[T] => OPVector = row => {
    val eachPivoted: Array[Array[Double]] =
      row.map(_.value).zip(allKeys).flatMap { case (map, keys) =>
        val cleanedMap = cleanMap(map, shouldClean, shouldCleanValue = false)
        keys.map(k => {
          val vOpt = cleanedMap.get(k)
          DateToUnitCircle.convertToRandians(vOpt, timePeriod)
        })
      }.toArray
    Vectors.dense(eachPivoted.flatten).compressed.toOPVector
  }
}

private[op] object DateToUnitCircle {

  def metadataValues(timePeriod: TimePeriod): Seq[String] = Seq(s"x_$timePeriod", s"y_$timePeriod")

  def convertToRandians(timestamp: Option[Long], timePeriodDesired: TimePeriod): Array[Double] = {
    val datetime: Option[JDateTime] = timestamp.map(new JDateTime(_))
    val (timePeriod, periodSize) = timePeriodDesired match {
      case TimePeriod.DayOfMonth => (datetime.map(_.dayOfMonth().get() - 1), 31)
      case TimePeriod.DayOfWeek => (datetime.map(_.dayOfWeek().get() - 1), 7)
      case TimePeriod.DayOfYear => (datetime.map(_.dayOfYear().get() - 1), 366)
      case TimePeriod.HourOfDay => (datetime.map(_.hourOfDay().get()), 24)
      case TimePeriod.MonthOfYear => (datetime.map(_.monthOfYear().get() - 1), 12)
      case TimePeriod.WeekOfMonth => (
        datetime.map(x => x.weekOfWeekyear().get() - x.withDayOfMonth(1).weekOfWeekyear().get()),
        6)
      case TimePeriod.WeekOfYear => (datetime.map(_.weekOfWeekyear().get() - 1), 53)
    }
    val radians = timePeriod.map(2 * math.Pi * _ / periodSize)
    radians.map(r => Array(math.cos(r), math.sin(r))).
      getOrElse(Array(0.0, 0.0))
  }
}

sealed abstract class TimePeriod extends EnumEntry with Serializable

object TimePeriod extends Enum[TimePeriod] {
  val values: Seq[TimePeriod] = findValues
  case object DayOfMonth extends TimePeriod
  case object DayOfWeek extends TimePeriod
  case object DayOfYear extends TimePeriod
  case object HourOfDay extends TimePeriod
  case object MonthOfYear extends TimePeriod
  case object WeekOfMonth extends TimePeriod
  case object WeekOfYear extends TimePeriod
}
