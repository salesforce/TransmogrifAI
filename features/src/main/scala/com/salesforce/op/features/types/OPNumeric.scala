/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types


/**
 * A base class for all the numeric feature types
 * @tparam N number type (Long, Double etc)
 */
abstract class OPNumeric[N] extends FeatureType {
  type Value = Option[N]
  def toDouble: Option[Double]
  final def isEmpty: Boolean = value.isEmpty
}
