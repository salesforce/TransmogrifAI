/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.dsl

import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.{DateListPivot, Transmogrifier}
import org.joda.time.{DateTime => JDateTime}


trait RichDateFeature {
  self: RichFeature with RichListFeature =>

  /**
   * Enrichment functions for Date Feature
   *
   * @param f Date Feature
   */
  implicit class RichDateFeature(val f: FeatureLike[Date]) {

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
     * Converts a sequence of Date features into DateList feature and then applies DateList vectorizer.
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
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: JDateTime = Transmogrifier.ReferenceDate,
      trackNulls: Boolean = Transmogrifier.TrackNulls,
      others: Array[FeatureLike[Date]] = Array.empty
    ): FeatureLike[OPVector] = {
      // vectorize DateList
      f.toDateList().vectorize(dateListPivot = dateListPivot, referenceDate = referenceDate, trackNulls = trackNulls,
        others = others.map(_.toDateList()))
    }

  }

  /**
   * Enrichment functions for DateTime Feature
   *
   * @param f DateTime Feature
   */
  implicit class RichDateTimeFeature(val f: FeatureLike[DateTime]) {

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
     * Converts a sequence of DateTime features into DateTimeList feature and then applies DateTimeList vectorizer.
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
     * @return result feature of type Vector
     */
    def vectorize
    (
      dateListPivot: DateListPivot,
      referenceDate: JDateTime = Transmogrifier.ReferenceDate,
      trackNulls: Boolean = Transmogrifier.TrackNulls,
      others: Array[FeatureLike[DateTime]] = Array.empty
    ): FeatureLike[OPVector] = {
      f.toDateTimeList().vectorize(dateListPivot = dateListPivot, referenceDate = referenceDate,
        trackNulls = trackNulls, others = others.map(_.toDateTimeList()))
    }

  }

}
