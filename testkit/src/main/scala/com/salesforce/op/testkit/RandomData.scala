/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.{FeatureType}

import scala.language.postfixOps
import scala.util.Random

/**
 * Generates random data of given feature type; it never ends, and it's not resettable.
 * If you want to generate the same data again, instantiate a new generator with the same seed.
 *
 * @tparam FT specific feature type
 */
trait RandomData[FT <: FeatureType] extends InfiniteStream[FT] with PartiallyDefined {
  self: FeatureFactoryOwner[FT] =>
  /**
   * Infinite stream of values produced
   *
   * @return a stream of random data
   */
  private[testkit] def streamOfValues: InfiniteStream[FT#Value]

  def next: FT = {
    ftFactory.newInstance {
      if (skip) null else streamOfValues.next
    }
  }

  /**
   * Reset this generator, by, say, seeding your rng(s)
   *
   * @param seed the seed
   * @return this instance
   */
  def reset(seed: Long): Unit = {
    rng.setSeed(seed)
  }


  /**
   * Random numbers generator used for a variety of purposes; not necessarily the values
   *
   * @return the generator, standard scala.util.Random
   */
  protected val rng: Random = new Random
}
