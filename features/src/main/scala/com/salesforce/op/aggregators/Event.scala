/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types.FeatureType

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Used by feature aggregators to do time based filtering
 *
 * @param date  date associated with feature
 * @param value value associated with feature
 * @tparam T type of feature
 */
case class Event[T <: FeatureType: WeakTypeTag](date: Long, value: T, isResponse: Boolean)
  extends Ordered[Event[T]] with Serializable {

  override def compare(that: Event[T]): Int = java.lang.Long.signum(date - that.date)

}

object Event {
  def apply[T <: FeatureType: WeakTypeTag](date: Long, value: T): Event[T] =
    Event[T](date, value, isResponse = false)
}
