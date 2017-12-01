/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird.MonoidAggregator
import org.joda.time.Duration

import scala.reflect.runtime.universe.WeakTypeTag


/**
 * Holds information for extracting features from data record
 *
 * @tparam I the type of the raw data from which the feature will be extracted
 * @tparam E the type of the data extracted into the event to be aggregated
 * @tparam A the type of the data that has been prepared for monoid aggregation
 * @tparam O the final type of the feature returned
 */
trait FeatureAggregator[I, E <: FeatureType, A, O <: FeatureType] extends Serializable {
  def extractFn: I => E

  def aggregator: MonoidAggregator[Event[E], A, O]

  def isResponse: Boolean

  protected def filterByDateWithCutoff(
    eventTime: Long,
    cutOffTime: CutOffTime,
    isResponse: Boolean,
    timeWindow: Option[Duration]
  ): Boolean

  final def extract(
    records: Iterable[I],
    timeStampFn: Option[I => Long],
    cutOffTime: CutOffTime,
    responseWindow: Option[Duration] = None,
    predictorWindow: Option[Duration] = None
  ): O = {
    var combined = aggregator.monoid.zero
    val timeWindow = if (isResponse) responseWindow else predictorWindow
    val iterator = records.iterator

    while (iterator.hasNext) {
      val record = iterator.next()
      val event = Event(date = timeStampFn.map(t => t(record)).getOrElse(0L), value = extractFn(record), isResponse)

      if (filterByDateWithCutoff(event.date, cutOffTime, isResponse, timeWindow)) {
        val prepared = aggregator.prepare(event)
        combined = aggregator.monoid.plus(combined, prepared)
      }
    }
    aggregator.present(combined)
  }

}

/**
 * Generic Feature Aggregator
 *
 * @param extractFn         function defining how to extract feature from the raw data type
 * @param aggregator        monoid defining how feature should be aggregated
 * @param isResponse        boolean describing whether feature is a response or predictor for aggregation purposes
 * @param specialTimeWindow time window for integration specific to this feature (will override time windows defined
 *                          by reader)
 * @tparam I the type of the raw data from which the feature will be extracted
 * @tparam E the type of the data extracted into the event to be aggregated
 * @tparam A the type of the data that has been prepared for monoid aggregation
 * @tparam O the final type of the feature returned
 */
case class GenericFeatureAggregator[I, E <: FeatureType : WeakTypeTag, A, O <: FeatureType : WeakTypeTag]
(
  extractFn: I => E,
  aggregator: MonoidAggregator[Event[E], A, O],
  isResponse: Boolean,
  specialTimeWindow: Option[Duration]
) extends FeatureAggregator[I, E, A, O] {

  final override def filterByDateWithCutoff(
    date: Long,
    cutOffTime: CutOffTime,
    isResponse: Boolean,
    timeWindow: Option[Duration]
  ): Boolean =
    cutOffTime.timeMs match {
      case None => true
      case Some(cutoff) =>
        specialTimeWindow.orElse(timeWindow) match {
          case None =>
            if (isResponse) date >= cutoff else date < cutoff
          case Some(window) =>
            if (isResponse) date >= cutoff && date <= (cutoff + window.getMillis)
            else date < cutoff && date >= (cutoff - window.getMillis)
        }
    }
}

/**
 * Reasonable defaults values for most aggregators
 *
 * @tparam T type of Feature computed from aggregator
 */
trait AggregatorDefaults[T <: FeatureType] {
  val ttag: WeakTypeTag[T]
  val ftFactory: FeatureTypeFactory[T]

  def prepare(input: Event[T]): T#Value = input.value.value
  def present(reduction: T#Value): T = ftFactory.newInstance(reduction)
}
