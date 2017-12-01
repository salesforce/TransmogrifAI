/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import scala.reflect.ClassTag

/**
 * A base class for all the list feature types
 * @tparam A item type
 */
abstract class OPList[A](implicit val cta: ClassTag[A]) extends OPCollection {
  override type Value = Seq[A]
  final def isEmpty: Boolean = value.isEmpty
  final def toArray: Array[A] = value.toArray(cta)
}
