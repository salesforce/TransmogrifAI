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

import com.salesforce.op.utils.stats.StreamingHistogram._

import scala.annotation.tailrec
import scala.collection.JavaConverters._

/**
 * By default this provides a dynamic histogram representation of size no larger
 * than maxBins, per:
 *
 * http://www.jmlr.org/papers/volume11/ben-haim10a/ben-haim10a.pdf
 *
 * @param maxBins maximum number of allowed bins in histogram
 */
class StreamingHistogram(val maxBins: Int) extends Serializable {

  private[this] val points: TreeMap[Double, Double] =
    getTreeMap[Double].asInstanceOf[TreeMap[Double, Double]]

  /**
   * @return histogram bins
   */
  final def getBins(): Array[Bin] = points.asScala.toArray

  /**
   * Merges this histogram with the input histogram.
   *
   * @param dist input value
   * @return merged distribution
   */
  final def merge(dist: StreamingHistogram): this.type = {
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

  private[stats] final def density(x: Double): Double = StreamingHistogram.density(getBins, x)

  /**
   * Performs sum algorithm outlined in paper evaluated at some input point, which approximates the number of
   * points less than or equal to the input point.
   *
   * @param x input point
   * @return approximate number of points that are less than or equal to the input point
   */
  final def sum(x: Double): Double = StreamingHistogram.sum(getBins, x)

  /**
   * @return CDF estimate using the histogram sum algorithm
   */
  final def sumCDF(x: Double): Double = StreamingHistogram.sumCDF(getBins, x)

  /**
   * @return empirical CDF estimate
   */
  final def empiricalCDF(x: Double): Double = StreamingHistogram.empiricalCDF(getBins, x)

  private[this] def mergePoints(): Unit = if (points.size > math.max(maxBins, 2)) {
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

  private[this] def updatePoints(updates: (Double, Double)*): Unit = {
    updates.foreach { case (point, ct) =>
      points.put(point, Option(points.get(point)).getOrElse(0.0) + ct)
    }

    (0 until math.max(0, points.size - maxBins)).foreach(_ => mergePoints())
  }
}

object StreamingHistogram {

  private[stats] type Bin = (Double, Double)
  private val EmptyBin: Bin = (0.0, 0.0)

  private[stats] def density(bins: Array[Bin], x: Double): Double = {
    val epsilon = 0.01
    val binsMin: Option[Bin] = if (bins.nonEmpty) Option((bins.map(_._1).min - epsilon) -> 0.0) else None
    val binsMax: Option[Bin] = if (bins.nonEmpty) Option((bins.map(_._1).max + epsilon) -> 0.0) else None
    val finalBins = binsMin ++ bins ++ binsMax
    finalBins.sliding(2).foldLeft(EmptyBin) { case ((prob, sum), arr) =>
      arr match {
        case List((p, m)) => (prob + m, sum + m)
        case List((p1, m1), (p2, m2)) =>
          val sumTerm = (m1 + m2) / 2
          val newProb = prob + { if (x >= p1 && x < p2) sumTerm else 0.0 }

          (newProb, sum + sumTerm)
      }
    } match {
      case (_, 0.0) => 0.0
      case (p, s) => p / s
    }
  }

  /**
   * Performs sum algorithm outlined in paper evaluated at some input point, which approximates the number of
   * points less than or equal to the input point.
   *
   * @param x input point
   * @return approximate number of points that are less than or equal to the input point
   */
  private[stats] def sum(bins: Array[Bin], x: Double): Double =
    bins.sliding(2).foldLeft((0.0, EmptyBin, false)) { case ((s, _, done), arr) =>
      arr match {
        case Array((p, m)) =>
          val newS = if (x == p) m / 2
            else if (x > m) m
            else 0.0

          (newS, EmptyBin, true)

        case Array((p1, m1), (p2, m2)) =>
          if (x >= p1 && x < p2 && !done) {
            val mx = m1 + ((m2 - m1) / (p2 - p1)) * (x - p1)
            val newS = s + (m1 / 2) + ((m1 + mx) / 2) * ((x - p1) / (p2 - p1))

            (newS, (p2, m2), true)
          } else if (done) {
            (s, (p2, m2), done)
          // It must be the case that (x < p1 || x >= p2)
          } else if (x >= p2) {
            (s + m1, (p2, m2), false)
          } else {
            (s, (p2, m2), true)
          }
      }
    } match {
      case (s, (p, m), true) => s
      case (s, (p, m), false) => s + { if (x == p) m / 2 else m }
    }

  /**
   * @return CDF estimate using the histogram sum algorithm
   */
  private[stats] def sumCDF(bins: Array[Bin], x: Double): Double =
    sum(bins, x) / bins.map(_._2).sum

  /**
   * @return empirical CDF estimate
   */
  private[stats] def empiricalCDF(bins: Array[Bin], x: Double): Double =
    bins.collect { case (p, m) if p <= x => m }.sum / bins.map(_._2).sum

  private def getTreeMap[T](): TreeMap[Double, T] =
    HistogramJavaUtils.getTreeMap[T].asInstanceOf[TreeMap[Double, T]]
}
