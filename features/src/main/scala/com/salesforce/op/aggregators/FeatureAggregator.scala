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
