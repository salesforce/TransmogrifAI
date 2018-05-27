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

import com.salesforce.op.features.types._
import com.salesforce.op.testkit.RandomReal.{NormalDistribution, UniformDistribution}

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of lists
 *
 * @param values the stream of values used as the source
 * @tparam DataType the type of data in lists
 * @tparam ListType the feature type of the data generated
 */
case class RandomList[DataType, ListType <: OPList[DataType] : WeakTypeTag]
(
  values: RandomStream[ListType#Value]
) extends StandardRandomData[ListType](sourceOfData = values)

object RandomList {

  /**
   * Produces random lists of texts
   *
   * @param texts  generator of random texts to be stored in the generated lists
   * @param minLen minimum length of the list; 0 if missing
   * @param maxLen maximum length of the list; if missing, all lists are of the same size
   * @return a generator of lists of texts
   */
  def ofTexts(texts: RandomText[_], minLen: Int = 0, maxLen: Int = -1): RandomList[String, TextList] =
    listsOf[String, TextList](texts.stream, minLen, maxLen)

  /**
   * Produces random lists of dates
   *
   * @param minLen minimum length of the list; 0 if missing
   * @param maxLen maximum length of the list; if missing, all lists are of the same size
   * @param dates  generator of random dates to be stored in the generated lists
   * @return a generator of lists of dates
   */
  def ofDates(dates: RandomIntegral[Date], minLen: Int = 0, maxLen: Int = -1): RandomList[Long, DateList] =
    listsOf[Long, DateList](dates.numbers.producer, minLen, maxLen)

  /**
   * Produces random lists of datetimes
   *
   * @param minLen    minimum length of the list; 0 if missing
   * @param maxLen    maximum length of the list; if missing, all lists are of the same size
   * @param datetimes generator of random dates to be stored in the generated lists
   * @return a generator of lists of datetimes
   */
  def ofDateTimes(datetimes: RandomIntegral[DateTime], minLen: Int = 0, maxLen: Int = -1):
  RandomList[Long, DateTimeList] = listsOf[Long, DateTimeList](datetimes.numbers.producer, minLen, maxLen)

  /**
   * Generator of random geolocations
   * latitude and longitude are uniformly distributed on the Earth's surface;
   * accuracy is uniformly distributed between all allowed values (0 to 10, including)
   *
   * @return the generator
   */
  def ofGeolocations: UniformGeolocation = new UniformGeolocation

  /**
   * Generator of random geolocations near a location with provided coordinates
   *
   * @param lat      latitude
   * @param lon      longitude
   * @param accuracy geolocation accuracy
   * @return the generator
   */
  def ofGeolocationsNear(lat: Double, lon: Double, accuracy: GeolocationAccuracy): NormalGeolocation =
    NormalGeolocation((lat, lon), accuracy)

  // scalastyle:off
  // TODO(vlad): implement locations on the ground using https://api.onwater.io/api/v1/results/60.085332511000786,150.01843974166854
  // scalastyle:on

  private def listsOf[D, T <: OPList[D] : WeakTypeTag](producer: Random => D, minLen: Int, maxLen: Int) =
    RandomList[D, T](RandomStream.ofChunks[D](minLen, maxLen)(producer))

  private def listsOf[D, T <: OPList[D] : WeakTypeTag](singles: RandomStream[D], minLen: Int, maxLen: Int) =
    RandomList[D, T](RandomStream.groupInChunks[D](minLen, maxLen)(singles))

  /**
   * Random geolocation class. @see #ofGeolocations for usage.
   */
  class UniformGeolocation extends
    FeatureFactoryOwner[Geolocation] with RandomData[Geolocation] {

    private val randomLongitude = new UniformDistribution(-180.0, 180.0)
    private val randomLatitude = new UniformDistribution(0.0, 2.0)
    private val randomAccuracy = new UniformDistribution(
      min = GeolocationAccuracy.Address.value,
      max = GeolocationAccuracy.State.value
    )

    override def reset(seed: Long): Unit = {
      randomLongitude.setSeed(seed)
      randomLatitude.setSeed(seed)
      randomAccuracy.setSeed(seed)
    }

    def streamOfValues: InfiniteStream[Seq[Double]] = new InfiniteStream[Seq[Double]] {
      override def next(): Seq[Double] = {
        val lon = randomLongitude.nextValue
        val ulat = randomLatitude.nextValue
        val lat = math.toDegrees(math.acos(1 - ulat))
        val acc = GeolocationAccuracy.withValue(randomAccuracy.nextValue.toInt)
        Geolocation(lat = lat - 90, lon = lon, accuracy = acc).value
      }
    }
  }

  /**
   * Random geolocation class. @see #ofGeolocations for usage.
   */
  case class NormalGeolocation(center: (Double, Double), accuracy: GeolocationAccuracy)
    extends FeatureFactoryOwner[Geolocation]
      with RandomData[Geolocation] {
    val randomDirection = new UniformDistribution(0.0, math.Pi * 2)
    val randomDistance = new NormalDistribution(0.0, accuracyOf(accuracy))

    override def reset(seed: Long): Unit = {
      randomDirection.setSeed(seed)
      randomDistance.setSeed(seed)
    }

    def streamOfValues: InfiniteStream[Seq[Double]] = new InfiniteStream[Seq[Double]] {
      override def next(): Seq[Double] = {
        val dist = randomDistance.nextValue
        val dir = randomDirection.nextValue
        val lat0 = center._1 + dist * math.cos(dir)
        val lon0 = center._2 + dist * math.sin(dir)
        val lat = math.max(-90.0, math.min(90.0, lat0))
        val lon = math.max(-180.0, math.min(180.0, lon0))
        Geolocation(lat = lat, lon = lon, accuracy = accuracy).value
      }
    }
  }

  private val accuracyOf = {
    import GeolocationAccuracy._
    Map[GeolocationAccuracy, Double](
      Unknown -> 40.0, // 40 degrees Fahrenheit
      Address -> 0.0001, // about 30 ft
      NearAddress -> 0.0003, // about 100 ft
      Block -> 0.0008, // about 300 ft
      Street -> 0.002, // about 800 ft
      ExtendedZip -> 0.005, // about 0.4 miles
      Zip -> 0.015, // about 1.2 miles
      Neighborhood -> 0.04, // about 2.5 miles
      City -> 0.25, // about 16 miles
      County -> 0.75, // about 50 miles
      State -> 2.0 // about 150 miles
    )
  }


}
