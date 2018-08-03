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

package com.salesforce.op.aggregators

import language.postfixOps
import math._
import com.salesforce.op.features.types.{Geolocation, GeolocationAccuracy}
import com.twitter.algebird.{Monoid, MonoidAggregator}

/**
 * Geolocation doesn't support concatenation since each list really represents just one object,
 * instead its default will be the geographic midpoint (found by averaging corresponding
 * x,y,z coordinates and then projecting that point onto the surface of the Earth)
 */
case object GeolocationMidpoint
  extends MonoidAggregator[Event[Geolocation], Array[Double], Geolocation]
    with GeolocationFunctions {

  override def prepare(input: Event[Geolocation]): Array[Double] = prepare(input.value)

  val monoid: Monoid[Array[Double]] = new Monoid[Array[Double]] {
    override def zero: Array[Double] = Zero

    override def plus(p1: Array[Double], p2: Array[Double]): Array[Double] = {
      if (isNone(p1)) p2
      else if (isNone(p2)) p1
      else {
        val weight1 = p1(3) // weight of the left point
        val weight2 = p2(3) // weight of the right point
        val weight = weight2 + weight1 // summary weight

        val (xmin, ymin, zmin) = (min(p1(4), p2(4)), min(p1(5), p2(5)), min(p1(6), p2(6)))
        val (xmax, ymax, zmax) = (max(p1(7), p2(7)), max(p1(8), p2(8)), max(p1(9), p2(9)))

        val (x, y, z) = ( // weighted coordinates
          (p1(0) * weight1 + p2(0) * weight2) / weight,
          (p1(1) * weight1 + p2(1) * weight2) / weight,
          (p1(2) * weight1 + p2(2) * weight2) / weight)

        val res = Array(
          x, y, z, weight,
          xmin, ymin, zmin,
          xmax, ymax, zmax
        )
        res
      }
    }
  }
}

trait GeolocationFunctions {

  val Zero: Array[Double] = Array.fill[Double](4)(0.0)

  def isNone(data: Array[Double]): Boolean = data(3) == 0

  /**
   * Prepare method to be used in the MonoidAggregator for Geolocation objects
   *
   * @param input Event-wrapped Geolocation object
   * @return Array of (x,y,z,acc,count) to be used by the arrayMonoid during aggregation
   */
  private[op] def prepare(input: Geolocation): Array[Double] = {
    // Convert the geolocation objects into arrays with (x, y, z, acc, count) for aggregation
    if (input.isEmpty) Zero
    else {
      val g = input.toGeoPoint
      val d = input.accuracy.rangeInUnits / 2
      Array[Double](
        g.x, g.y, g.z,
        1.0,
        g.x - d, g.y - d, g.z - d,
        g.x + d, g.y + d, g.z - d
      )
    }
  }

  /**
   * The width of an area described by this given entry.
   * We have minx, miny, minz in xs(4), xs(5), xs(6), and maxx, maxy, maxz in xs(7), xs(8), xs(9).
   * Their differences are the sizes of the rectangular prism defined by these point.
   * We take the maximum of these dimensions, so that we could figure out the accuracy later on.
   * @param xs an array with geolocation data.
   * @return the biggest dimension of the prism.
   */
  private def width(xs: Array[Double]): Double = max(max(xs(7) - xs(4), xs(8) - xs(5)), xs(9) - xs(6))

  /**
   * Present method to be used in the MonoidAggregator for Geolocation objects
   *
   * @param data Array of (x,y,z,acc,count) to be used by the arrayMonoid during aggregation
   * @return Geolocation object corresponding to aggregated x,y,z coordinates
   */
  def present(data: Array[Double]): Geolocation = {
    if (isNone(data)) Geolocation.empty
    else {
      val lat = toDegrees(
        atan2(data(2), sqrt(data(0) * data(0) + data(1) * data(1)))
      )
      val lon = toDegrees(atan2(data(1), data(0)))
      val range = width(data) // widths max

      Geolocation(lat, lon, GeolocationAccuracy.forRangeInUnits(range))
    }
  }
}
