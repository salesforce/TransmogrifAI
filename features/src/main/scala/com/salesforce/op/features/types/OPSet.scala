/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import scala.reflect.ClassTag


/**
 * A base class for all the set feature types
 */
abstract class OPSet[A](implicit val cta: ClassTag[A]) extends OPCollection with MultiResponse {
  type Value <: scala.collection.Set[A]
  final def isEmpty: Boolean = value.isEmpty
  final def toArray: Array[A] = value.toArray(cta)
}
