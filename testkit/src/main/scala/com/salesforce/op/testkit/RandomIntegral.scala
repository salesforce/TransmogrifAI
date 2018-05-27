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

import java.util.{Date => JDate}

import com.salesforce.op.features.types._

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of data as integral numbers
 *
 * @param numbers            the stream of longs used as the source
 * @tparam DataType the feature type of the data generated
 */
case class RandomIntegral[DataType <: Integral : WeakTypeTag]
(
  numbers: RandomStream[Long]
) extends StandardRandomData[DataType](
  numbers map (Option(_))
) with ProbabilityOfEmpty

/**
 * Generator of data as integral numbers
 */
object RandomIntegral {

  /**
   * Generator of random integral values
   *
   * @return generator of integrals
   */
  def integrals: RandomIntegral[Integral] =
    RandomIntegral[Integral](RandomStream.ofLongs)

  /**
   * Generator of random integral values in a given range
   *
   * @param from minimum value to produce (inclusive)
   * @param to   maximum value to produce (exclusive)
   * @return the generator of integrals
   */
  def integrals(from: Long, to: Long): RandomIntegral[Integral] =
    RandomIntegral[Integral](RandomStream.ofLongs(from, to))

  /**
   * Incremental generator of dates.
   * Dates are generated incrementally, with the step in millis that is
   * between minStep and maxStep.
   *
   * @param init    initial date starting from which to generate
   * @param minStep next date will be at least this many millis after the previous
   * @param maxStep next date will be at most this many millis after the previous (-1 if constant
   * @return the generator of dates
   */
  def dates(init: JDate, minStep: Int, maxStep: Int): RandomIntegral[Date] = {
    RandomIntegral[Date](incrementingStream(init, minStep, maxStep))
  }

  /**
   * Incremental generator of datetimes
   *
   * @param init    initial date starting from which to generate
   * @param minStep next datetime will be at least this many millis after the previous
   * @param maxStep next datetime will be at most this many millis after the previous
   * @return the generator of datetimes
   */
  def datetimes(init: JDate, minStep: Int, maxStep: Int): RandomIntegral[DateTime] = {
    RandomIntegral[DateTime](incrementingStream(init, minStep, maxStep))
  }

  private def incrementingStream(init: JDate, minStep: Long, maxStep: Long): RandomStream[Long] =
    RandomStream.incrementing(init.getTime, minStep, maxStep)
}


