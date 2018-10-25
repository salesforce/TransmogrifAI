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

package com.salesforce.op.utils.stats

import scala.collection.JavaConverters._

/**
 * Provides implicit methods for [[StreamingHistogram]] to be used for distribution estimation.
 */
object RichStreamingHistogram {
  final implicit class RichStreamingHistogramImpl(val hist: StreamingHistogram) extends AnyVal {

    def getBins(): Array[(Double, Double)] =
      hist.getAsMap.asScala.toArray.flatMap {
        case (a, b) => b.headOption.map(d => (a.doubleValue, d.toDouble))
      }


    /**
     * Yields set of bins describing streaming histogram. The size of the resulting array will
     * be [[(_: StreamingHistogram).getAsMap.size + 2]] (if it is non-empty, otherwise 0) where
     * the boundary points will be [[padding]] values away from the minimum and maximum points, respectively.
     *
     * @param padding boundary padding to add to histogram bins
     * @return padded bins
     */
    def getPaddedBins(padding: Double = 0.1): Array[(Double, Double)] =
      RichStreamingHistogram.paddedBins(getBins, padding)

    /**
     * Produces standard histogram density estimator for this streaming histogram where the bins are constructed
     * given the input padding.
     *
     * @param padding boundary padding to add to histogram bins
     * @return standard histogram density estimator
     */
    def density(padding: Double = 0.1): Double => Double =
      RichStreamingHistogram.density(getPaddedBins(padding))
  }

  // The following are exposed for test comparisons
  private[op] def paddedBins(bins: Array[(Double, Double)], padding: Double): Array[(Double, Double)] =
    if (bins.isEmpty) bins
    else {
      val points = bins.map(_._1)
      val (min, max) = (points.min, points.max)

      Array((points.min - padding) -> 0.0) ++ bins ++ Array((points.max + padding) -> 0.0)
    }

  private[op] def density(bins: Array[(Double, Double)]): Double => Double =
    (x: Double) =>
      bins.sliding(2).foldLeft((0.0, 0.0)) { case ((prob, sum), arr) =>
        arr match {
          case Array((p, m)) => (prob + m, sum + m)
          case Array((p1, m1), (p2, m2)) =>
            val sumTerm = (m1 + m2) / 2
            val newProb = prob + { if (x >= p1 && x < p2) sumTerm else 0.0 }

            (newProb, sum + sumTerm)
        }
      } match {
        case (_, 0.0) => 0.0
        case (p, s) => p / s
      }
}
