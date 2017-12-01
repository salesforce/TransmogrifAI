/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types


class MultiPickList(val value: Set[String]) extends OPSet[String] {
  type Value = Set[String]
}
object MultiPickList {
  def apply(value: Set[String]): MultiPickList = new MultiPickList(value)
  def empty: MultiPickList = FeatureTypeDefaults.MultiPickList
}
