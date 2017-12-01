/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

/**
 * A base class for all the map feature types
 * @tparam A item type
 */
abstract class OPMap[A] extends OPCollection {
  type Element = A
  override type Value = Map[String, A]
  final def isEmpty: Boolean = value.isEmpty
}
