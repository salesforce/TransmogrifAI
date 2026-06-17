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

import com.salesforce.op.features.types._

import scala.reflect.runtime.universe.WeakTypeTag

/**
 * Generator of data as integer numbers
 *
 * @param numbers the stream of longs used as the source
 * @tparam DataType the feature type of the data generated
 */
case class RandomInteger[DataType <: Integer : WeakTypeTag]
(
  numbers: RandomStream[Int]
) extends StandardRandomData[DataType](
  numbers map (Option(_))
) with ProbabilityOfEmpty

/**
 * Generator of data as integral numbers
 */
object RandomInteger {

  /**
   * Generator of random integral values
   *
   * @return generator of integrals
   */
  def integers: RandomInteger[Integer] =
    RandomInteger[Integer](RandomStream.ofInts)

  /**
   * Generator of random integral values in a given range
   *
   * @param from minimum value to produce (inclusive)
   * @param to   maximum value to produce (exclusive)
   * @return the generator of integrals
   */
  def integers(from: Int, to: Int): RandomInteger[Integer] =
    RandomInteger[Integer](RandomStream.ofInts(from, to))

}


