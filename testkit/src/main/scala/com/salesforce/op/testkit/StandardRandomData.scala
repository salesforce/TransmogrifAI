/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.FeatureType

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Data generator that uses standard Random
 */
abstract class StandardRandomData[FT <: FeatureType : WeakTypeTag]
(
  sourceOfData: RandomStream[FT#Value]
) extends FeatureFactoryOwner[FT] with RandomData[FT] {
  /**
   * Infinite stream of values produced by sourceOfData when given the rng
   *
   * @return a stream of random data
   */
  def streamOfValues: InfiniteStream[FT#Value] = sourceOfData(rng)

  override def reset(seed: Long): Unit = {
    sourceOfData.reset(seed)
    super.reset(seed)
  }
}
