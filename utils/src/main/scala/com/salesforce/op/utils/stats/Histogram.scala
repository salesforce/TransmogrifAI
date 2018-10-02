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

import com.salesforce.op.utils.stats.Histogram._
import com.salesforce.op.utils.stats.HistogramBase._

import scala.collection.JavaConverters._

/**
 * A bin is defined to be a point (p, m) where p, m are real numbers such that m > 0. We call p a binning point,
 * and m a point count.
 *
 * Given a non-negative integer B, a histogram of size B is a set of bins H_B = {(p_1, m_1), ..., (p_B, m_B)}
 * such that p_1 < ... < p_B, where H_0 is defined to be the empty set. The size of B is |H_B| = B.
 *
 * The total count of H_B is defined to be m_1 + ... + m_B if B > 0, and 0 otherwise.
 *
 * If H_B is non-empty, then the minimum of H_B is p_1, i.e. its smallest binning point,
 * and the maximum, p_B, i.e. its largest binning point.
 *
 * If H_B is non-empty, then we define [[sum]] and [[uniform]]  operations on H_B per:
 *
 * http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
 *
 */
sealed trait HistogramBase {

  /**
   * @eturn an array of bins (in ascending order per binning points [if non-empty]) representing this histogram
   */
  def bins: Array[Bin]

  /**
   * @return the maximum of this histogram, if it is defined.
   */
  def maximum: Option[Double]

  /**
   * @return the minimum of this histogram, if it is defined.
   */
  def minimum: Option[Double]

  /**
   * @return the size of this histogram
   */
  def size: Int

  /**
   * @return the total count of this histogram.
   */
  def totalCount: Double

  /**
   * @param bins input histogram bins
   * @param f    non-negative functional defined on set of all possible bins
   * @param cond deciding condition for binary search algorithm
   * @returns index minimizing f given constraints set by cond
   */
  private def binarySearch(bins: Array[Bin], f: Bin => Double, cond: Double => Boolean): Int = 0

}

object HistogramBase {
  private type Bin = (Double, Double)
}

/**
 * By default this provides a dynamic histogram representation of size no larger
 * than maxBins, per:
 *
 * http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
 *
 * @param maxBins maximum number of allowed bins in histogram
 */
class Histogram(val maxBins: Int) {

  protected[this] val points: TreeMap[Double, Double] =
    getTreeMap[Double].asInstanceOf[TreeMap[Double, Double]]

  /**
   * @return histogram bins
   */
  final def getBins(): Map[Double, Double] = points.asScala.toMap

  /**
   * Merges this distribution with the input distribution.
   *
   * @param dist input value
   * @return merged distribution
   */
  final def merge(dist: Histogram): this.type = {
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
    updatePoints(values.map(_ -> 1.0): _*)

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

  private def updatePoints(updates: (Double, Double)*): Unit = {
    updates.foreach { case (point, ct) =>
      points.put(point, Option(points.get(point)).getOrElse(0.0) + ct)
    }

    (0 until math.max(0, points.size - maxBins)).foreach(_ => mergePoints())
  }
}

object Histogram {
  def getTreeMap[T](): TreeMap[Double, T] =
    HistogramJavaUtils.getTreeMap[T].asInstanceOf[TreeMap[Double, T]]
}
