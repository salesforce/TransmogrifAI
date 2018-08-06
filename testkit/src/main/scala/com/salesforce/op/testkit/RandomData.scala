/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
