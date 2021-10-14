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

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.features.types.{DateMap, OPVector}
import com.salesforce.op.stages.base.sequence.{SequenceEstimator, SequenceModel}
import com.salesforce.op.utils.spark.{OpVectorColumnMetadata, OpVectorMetadata}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset

import scala.reflect.runtime.universe.TypeTag

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
      grouping = Option(key),
      descriptorValue = Option(dec)
    )

    meta.withColumns(cols.toArray)
  }

  override def fitFn(dataset: Dataset[Seq[T#Value]]): SequenceModel[T, OPVector] = {
    val shouldClean = $(cleanKeys)
    val allKeys = getKeyValues(dataset, shouldClean, shouldCleanValues = false)

    val meta = makeVectorMetadata(allKeys)
    setMetadata(meta.toMetadata)
    new DateMapToUnitCircleVectorizerModel[T](allKeys = allKeys, shouldClean = shouldClean,
      timePeriod = getTimePeriod, operationName = operationName, uid = uid
    )
  }

}

/**
 * Model for DateMapToUnitCircleVectorizer
 * @param allKeys map keys in order to flatten data consistenly
 * @param shouldClean map keys are have text cleaned
 * @param timePeriod time period for circular representations
 * @param operationName unique name of the operation this stage performs
 * @param uid           uid for instance
 * @param tti           type tag for input
 * @tparam T            DateMap type
 */
final class DateMapToUnitCircleVectorizerModel[T <: DateMap] private[op]
(
  val allKeys: Seq[Seq[String]],
  val shouldClean: Boolean,
  val timePeriod: TimePeriod,
  operationName: String,
  uid: String
)(implicit tti: TypeTag[T]) extends SequenceModel[T, OPVector](operationName = operationName, uid = uid)
  with CleanTextMapFun {

  override def transformFn: Seq[T] => OPVector = row => {
    val eachPivoted: Array[Array[Double]] =
      row.map(_.value).zip(allKeys).flatMap { case (map, keys) =>
        val cleanedMap = cleanMap(map, shouldClean, shouldCleanValue = false)
        keys.map(k => {
          val vOpt = cleanedMap.get(k)
          DateToUnitCircle.convertToRadians(vOpt, timePeriod)
        })
      }.toArray
    Vectors.dense(eachPivoted.flatten).compressed.toOPVector
  }
}

