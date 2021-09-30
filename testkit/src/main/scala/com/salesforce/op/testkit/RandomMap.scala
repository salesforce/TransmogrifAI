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

import language.postfixOps
import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random
import RandomMap._
import com.salesforce.op.features.types._
import com.salesforce.op.testkit.DataSources.{RandomFirstName, RandomLastName}

/**
 * Generator of maps
 *
 * @param values the stream of values used as the source
 * @tparam DataType the type of data in lists
 * @tparam MapType the feature type of the data generated
 */
case class RandomMap[DataType, MapType <: OPMap[DataType] : WeakTypeTag]
(
  private val values: RandomStream[Seq[MapType#Element]],
  private val keys: Int => String = "k" +,
  private val sources: Iterable[RandomData[_]] = None
) extends StandardRandomData[MapType](values map asMap(keys)) {

  override def reset(seed: Long): Unit = {
    for { random <- sources } {
      random.reset(seed)
    }
    super.reset(seed)
  }

  def withKeys(keys: Int => String): RandomMap[MapType#Element, MapType] = copy(keys = keys)

  def withPrefix(prefix: String): RandomMap[MapType#Element, MapType] = withKeys(prefix +)
}

object RandomMap {
  class Compatibility[DataType, MapType]
  implicit val cBase64 = new Compatibility[Base64, Base64Map]
  implicit val cCity = new Compatibility[City, CityMap]
  implicit val cComboBox = new Compatibility[ComboBox, ComboBoxMap]
  implicit val cCountry = new Compatibility[Country, CountryMap]
  implicit val cCurrency = new Compatibility[Currency, CurrencyMap]
  implicit val cDate = new Compatibility[Date, DateMap]
  implicit val cDateTime = new Compatibility[DateTime, DateTimeMap]
  implicit val cEmail = new Compatibility[Email, EmailMap]
  implicit val cGeolocation = new Compatibility[Geolocation, GeolocationMap]
  implicit val cID = new Compatibility[ID, IDMap]
  implicit val cInteger = new Compatibility[Integer, IntegerMap]
  implicit val cIntegral = new Compatibility[Integral, IntegralMap]
  implicit val cMultiPickList = new Compatibility[MultiPickList, MultiPickListMap]
  implicit val cPhone = new Compatibility[Phone, PhoneMap]
  implicit val cPickList = new Compatibility[PickList, PickListMap]
  implicit val cPostalCode = new Compatibility[PostalCode, PostalCodeMap]
  implicit val cPercent = new Compatibility[Percent, PercentMap]
  implicit val cReal = new Compatibility[Real, RealMap]
  implicit val cState = new Compatibility[State, StateMap]
  implicit val cStreet = new Compatibility[Street, StreetMap]
  implicit val cText = new Compatibility[Text, TextMap]
  implicit val cTextArea = new Compatibility[TextArea, TextAreaMap]
  implicit val cURL = new Compatibility[URL, URLMap]

  private def asMap[T](keys: Int => String)(values: Seq[T]): Map[String, T] = {
    (for {
      (v, i) <- values.zipWithIndex
    } yield keys(i) -> v) toMap
  }

  private def mapsOf[T, M <: OPMap[T] : WeakTypeTag](
    producer: Random => T,
    minLen: Int, maxLen: Int,
    sources: Iterable[RandomData[_]] = None
  ) = {
    val values: RandomStream[Seq[T]] = RandomStream.ofChunks[T](minLen, maxLen)(producer)
    RandomMap[T, M](values, sources = sources)
  }

  private def mapsFromStream[T, M <: OPMap[T] : WeakTypeTag](
    stream: RandomStream[T],
    minLen: Int, maxLen: Int,
    sources: Iterable[RandomData[_]] = None
  ) = {
    val values: RandomStream[Seq[T]] = RandomStream.groupInChunks[T](minLen, maxLen)(stream)
    RandomMap[T, M](values, sources = sources)
  }

  /**
   * Produces random maps of binaries
   *
   * @param probabilityOfSuccess the probability at which the values are true
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @return a generator of maps of texts; the keys by default have the form "k0", "k1", etc
   */
  def ofBinaries(probabilityOfSuccess: Double, minSize: Int, maxSize: Int):
  RandomMap[Boolean, BinaryMap] =
    mapsOf[Boolean, BinaryMap](
      RandomStream.ofBits(probabilityOfSuccess).producer,
      minSize: Int, maxSize: Int)

  /**
   * Produces random maps of texts
   *
   * @param textGenerator - a generator of single values, e.g. emails(domain), or countries
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @tparam T the type of single-value data to be generated
   * @tparam M the type of map
   * @return a generator of maps of texts; the keys by default have the form "k0", "k1", etc
   */
  def of[T <: Text, M <: OPMap[String] : WeakTypeTag](
    textGenerator: RandomText[T], minSize: Int, maxSize: Int)
    (implicit compatibilityOfBaseTypeAndMapType: Compatibility[T, M]):
  RandomMap[String, M] =
    mapsFromStream[String, M](textGenerator.stream, minSize, maxSize, sources = Some(textGenerator))

  /**
   * Produces random maps of integer
   *
   * @param valueGenerator - a generator of single (int) values
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @tparam T the type of single-value data to be generated
   * @tparam M the type of map
   * @return a generator of maps of integrals; the keys by default have the form "k0", "k1", etc
   */
  def ofInt[T <: Integer, M <: OPMap[Int] : WeakTypeTag](valueGenerator: RandomInteger[T], minSize: Int, maxSize: Int)
  (implicit compatibilityOfBaseTypeAndMapType: Compatibility[T, M]): RandomMap[Int, M] =
    mapsOf[Int, M](valueGenerator.numbers.producer, minSize, maxSize, sources = Some(valueGenerator))

  /**
   * Produces random maps of integrals
   *
   * @param valueGenerator - a generator of single (long) values
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @tparam T the type of single-value data to be generated
   * @tparam M the type of map
   * @return a generator of maps of integrals; the keys by default have the form "k0", "k1", etc
   */
  def of[T <: Integral, M <: OPMap[Long] : WeakTypeTag](
    valueGenerator: RandomIntegral[T], minSize: Int, maxSize: Int)
    (implicit compatibilityOfBaseTypeAndMapType: Compatibility[T, M]):
  RandomMap[Long, M] =
    mapsOf[Long, M](valueGenerator.numbers.producer, minSize, maxSize, sources = Some(valueGenerator))

  /**
   * Produces random NameStats maps (custom maps with pre-defined keys/values with names and demographic info)
   *
   * @return a generator of NameStats maps; the keys will be limted to the set defined by NameStats.Keys
   */
  def ofNameStats(): RandomMap[String, NameStats] = {
    def getRandomElement(list: Seq[String], random: Random): String =
      list(random.nextInt(list.length))
    val producer: Random => Seq[String] = rng => {
      val firstName = RandomFirstName(rng)
      val lastName = RandomLastName(rng)
      Seq(
        firstName + " " + lastName,
        getRandomElement(Seq(true.toString, false.toString), rng),
        firstName, lastName,
        getRandomElement(NameStats.GenderValue.values.map(_.toString), rng)
      )
    }
    val values: RandomStream[Seq[String]] = RandomStream(producer = producer)
    val keys: Int => String = NameStats.Key.values.map(_.toString)
    RandomMap[String, NameStats](values = values, keys = keys, sources = None)
  }

  /**
   * Produces random maps of geolocation
   *
   * @param valueGenerator - a generator of single Geolocation values
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @tparam Source the type of geolocation generator
   * @return a generator of maps of geolocations; the keys by default have the form "k0", "k1", etc
   */
  def ofGeolocations[Source <: RandomData[Geolocation]](
    valueGenerator: RandomData[Geolocation], minSize: Int, maxSize: Int)
    (implicit compatibilityOfBaseTypeAndMapType: Compatibility[Geolocation, GeolocationMap]):
  RandomMap[Seq[Double], GeolocationMap] =
    mapsOf[Seq[Double], GeolocationMap](rng => valueGenerator.next.value, minSize, maxSize)

  def ofMultiPickLists(picklists: RandomSet[String, MultiPickList], minSize: Int, maxSize: Int):
  RandomMap[Set[String], MultiPickListMap] =
    mapsOf[Set[String], MultiPickListMap](picklists.values.producer, minSize, maxSize)

  /**
   * Produces random maps of Reals
   *
   * @param valueGenerator - a generator of single (Double) values
   * @param minSize minimum size of the map; 0 if missing
   * @param maxSize maximum size of the map; if missing, all maps are of the same size
   * @tparam T the type of single-value data to be generated
   * @tparam M the type of map
   * @return a generator of maps of reals; the keys by default have the form "k0", "k1", etc
   */
  def ofReals[T <: Real, M <: OPMap[Double] : WeakTypeTag](
    valueGenerator: RandomReal[T], minSize: Int, maxSize: Int)
    (implicit compatibilityOfBaseTypeAndMapType: Compatibility[T, M]):
  RandomMap[Double, M] = {
    val sequences: RandomStream[Seq[Double]] = new RandomStream[Seq[Double]](
      rng => {
        val range = 0 until RandomStream.randomBetween(minSize, maxSize)(rng)
        range map (_ => valueGenerator.randomValues.nextValue)
      }
    )

    RandomMap[Double, M](sequences, sources = Some(valueGenerator))
  }
}
