/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._
import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives the union of Vector data
 */
case object UnionVector
  extends MonoidAggregator[Event[OPVector], Vector, OPVector]
    with AggregatorDefaults[OPVector] {
  implicit val ttag = weakTypeTag[OPVector]
  val ftFactory = FeatureTypeFactory[OPVector]()
  val monoid: Monoid[Vector] = Monoid.from(Vectors.zeros(0))((v1: Vector, v2: Vector) =>
    Vectors.dense(v1.toArray ++ v2.toArray)
  )
}
