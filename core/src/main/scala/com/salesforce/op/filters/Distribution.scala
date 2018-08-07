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

package com.salesforce.op.filters

import com.salesforce.op.features.types.FeatureType

import scala.util.Random

/**
 * Assigns a real-valued distribution to a given feature type.
 *
 * @param T Input FeatureType instance
 */
trait Distribution[T <: FeatureType] {

  /**
   * Total number of observations
   */
  def count: Double

  /**
   * Total number of missing observations
   */
  def nullCount: Double

  /**
   * Maximum real-value observed
   */
  def maximum: Double

  /**
   * Minimum real-value observed
   */
  def minimum: Double

  /**
   * Evaluates CDF at input point for the associated R.V. distribution
   *
   * @param x input value
   */
  def cdf(x: Double): Double

  /**
   * Returns a distribution updated with event-value of type T
   *
   * @param value input value
   */
  def update(value: T): Distribution[T]

  /**
   * CJS divergence
   *
   * @param dist distribution to compare
   * @param n Monte Carlo integration sample size
   */
  final def cjsDivergence(dist: Distribution[T], n: Int): Double =
    jsFunc(dist, d => d.cdf(_))(mcSample(dist, n))

  /**
   * Fill rate of distribution
   */
  final def fillRate: Double = if (count == 0.0) 0.0 else (count - nullCount) / count

  /**
   * Yields absolute difference between the fill rates of the two distributions
   *
   * @param dist Distribution to compare
   */
  final def relativeFillRate(dist: Distribution[T]): Double =
    math.abs(fillRate - dist.fillRate)

  /**
   * Yields max(this.fillRate / dist.fillRate, dist.fillRate / this.fillRate)
   *
   * @param dist Distribution to compare
   */
  final def relativeFillRatio(dist: Distribution[T]): Double = {
    val (thisFill, thatFill) = (this.fillRate, dist.fillRate)
    val (small, large) = if (thisFill < thatFill) (thisFill, thatFill) else (thatFill, thisFill)

    if (small == 0.0) Double.PositiveInfinity else large / small
  }

  protected def jsFunc(dist: Distribution[T], f: Distribution[T] => Double => Double): Set[Double] => Double =
    (event: Set[Double]) => {
      val (thisF, thatF) = (f(this), f(dist))

      0.5 * event.map { outcome =>
        val (thisVal, thatVal) = (thisF(outcome), thatF(outcome))
        val mainSum =
          if (thisVal == 0.0 || thatVal == 0.0) 0.0 else thisVal * log2(thisVal) + thatVal * log2(thatVal)
        val fSum = thisVal + thatVal

        mainSum - fSum * (log2(0.5) + log2(fSum))
      }.sum
    }

  protected def mcSample(dist: Distribution[T], n: Int): Set[Double] = {
    val a = math.min(minimum, dist.minimum)
    val b = math.max(maximum, dist.maximum)

    (0 until n).map(_ => ((b - a) * Random.nextDouble + a) / n).toSet
  }

  private def log2(x: Double): Double = math.log(x) / math.log(2)
}
