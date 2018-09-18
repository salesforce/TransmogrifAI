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

import java.util.TreeMap

import scala.collection.JavaConverters._

/**
 * By default this provides a dynamic histogram representation of size no larger
 * than maxBins, per:
 *
 * http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
 *
 * @param maxBins maximum number of allowed bins in histogram
 */
class AdaptiveHistogram(val maxBins: Int) {

  protected[this] val points: TreeMap[Double, Long] =
    AdaptiveHistogramUtils.getTreeMap.asInstanceOf[TreeMap[Double, Long]]

  /**
   * @return histogram bins
   */
  final def getBins(): Map[Double, Long] = points.asScala.toMap

  /**
   * Merges this distribution with the input distribution.
   *
   * @param dist input value
   * @return merged distribution
   */
  final def merge(dist: AdaptiveHistogram): this.type = {
    updatePoints(dist.getBins.toSeq: _*)

    this
  }

  /**
   * Returns a histogram updated with new observations
   *
   * @param values input values
   * @return updated feature distribution
   */
  final def update(values: Double*): this.type = {
    updatePoints(values.map(_ -> 1L): _*)

    this
  }

  private def mergePoints(): Unit = if (points.size > math.max(maxBins, 2)) {
    val (q1, q2) = points.descendingKeySet.descendingIterator.asScala.sliding(2).map(_ match {
      case List(a, b) => (a, b)
    }).reduce { (p1, p2) =>
      // Please note that default TreeMap comparator will actually produce ascending iterator
      // i.e p2._2 > p2._1 = p1._2 > p1._1
      if (p1._2 - p1._1 <= p2._2 - p2._1) p1 else p2
    }
    val (k1, k2) = (points.get(q1), points.get(q2)) // By construction, points are in our keyset

    // Remove the old points from keyset
    points.remove(q1)
    points.remove(q2)

    // Add new point to bins.
    points.put((q1 * k1 + q2 * k2) / (k1 + k2), k1 + k2)
  }

  private def updatePoints(updates: (Double, Long)*): Unit = {
    updates.foreach { case (point, ct) =>
      points.put(point, Option(points.get(point)).getOrElse(0L) + ct)
    }

    (0 until math.max(0, points.size - maxBins)).foreach(_ => mergePoints())
  }
}
