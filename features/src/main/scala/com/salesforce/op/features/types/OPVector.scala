/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types

import org.apache.spark.ml.linalg.Vector


class OPVector(val value: Vector) extends OPCollection {
  type Value = Vector
  final def isEmpty: Boolean = value.size == 0
}
object OPVector {
  def apply(value: Vector): OPVector = new OPVector(value)
  def empty: OPVector = FeatureTypeDefaults.OPVector
}
