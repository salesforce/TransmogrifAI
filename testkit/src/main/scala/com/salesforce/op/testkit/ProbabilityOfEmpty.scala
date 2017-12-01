/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.FeatureType

import scala.util.Random

/**
 * This trait allows the value to be empty
 */
trait ProbabilityOfEmpty extends PartiallyDefined {
  self: RandomData[_] =>

  /**
   * Probability that the stream returns a None, instead of Some(value)
   */
  private var _probabilityOfEmpty: Double = 0.0

  def probabilityOfEmpty: Double = _probabilityOfEmpty

  /**
   * Here we specify the probability of value returned being empty
   *
   * @param p the probability; this is the ratio of empty values in putput stream
   */
  def withProbabilityOfEmpty(p: Double): this.type = {
    require(p >= 0.0 && p <= 1.0, s"Require probability 0<=$p<=1")
    _probabilityOfEmpty = p
    this
  }

  /**
   * Infinite stream of booleans specifying that next value is None
   *
   * @return the stream of booleans, which are true with the tiven probability
   */
  private def streamOfEmpties: InfiniteStream[Boolean] =
    RandomStream.ofBits(probabilityOfEmpty)(rng)

  override def skip: Boolean = streamOfEmpties.next

}
