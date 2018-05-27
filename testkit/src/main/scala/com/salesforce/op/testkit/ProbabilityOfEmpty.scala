/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
