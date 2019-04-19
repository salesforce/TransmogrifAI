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

package com.salesforce.op.test

import java.text.SimpleDateFormat

import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureBuilder, FeatureSparkTypes}
import com.salesforce.op.testkit.RandomList.UniformGeolocation
import com.salesforce.op.testkit._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.types.StructType

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe._


/**
 * Test Feature Builder is a factory for creating datasets and features for tests
 */
case object TestFeatureBuilder {

  case object DefaultFeatureNames {
    val (f1, f2, f3, f4, f5) = ("f1", "f2", "f3", "f4", "f5")
  }

  /**
   * Build a dataset with one feature of specified type
   *
   * @param f1name feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 feature type
   * @return dataset with one feature of specified type
   */
  def apply[F1 <: FeatureType : TypeTag](
    f1name: String, data: Seq[F1]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1]) = {
    val f1 = feature[F1](f1name)
    val schema = FeatureSparkTypes.toStructType(f1)
    (dataframe(schema, data.map(Tuple1(_))), f1)
  }

  /**
   * Build a dataset with one feature of specified type
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 feature type
   * @return dataset with one feature of specified type
   */
  def apply[F1 <: FeatureType : TypeTag](data: Seq[F1])(implicit spark: SparkSession): (DataFrame, Feature[F1]) = {
    apply[F1](f1name = DefaultFeatureNames.f1, data)
  }

  /**
   * Build a dataset with two features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @return dataset with two features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag](
    f1name: String, f2name: String, data: Seq[(F1, F2)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2]) = {
    val (f1, f2) = (feature[F1](f1name), feature[F2](f2name))
    val schema = FeatureSparkTypes.toStructType(f1, f2)
    (dataframe(schema, data), f1, f2)
  }

  /**
   * Build a dataset with two features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @return dataset with two features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag](data: Seq[(F1, F2)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2]) = {
    apply[F1, F2](f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2, data)
  }

  /**
   * Build a dataset with three features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @return dataset with three features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag, F3 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, data: Seq[(F1, F2, F3)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3]) = {
    val (f1, f2, f3) = (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3)
    (dataframe(schema, data), f1, f2, f3)
  }

  /**
   * Build a dataset with three features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @return dataset with three features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag, F2 <: FeatureType : TypeTag, F3 <: FeatureType : TypeTag](
    data: Seq[(F1, F2, F3)]
  )(implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3]) = {
    apply[F1, F2, F3](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, data)
  }

  /**
   * Build a dataset with four features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param f4name 4th feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @return dataset with four features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, f4name: String, data: Seq[(F1, F2, F3, F4)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4]) = {
    val (f1, f2, f3, f4) = (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name), feature[F4](f4name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3, f4)
    (dataframe(schema, data), f1, f2, f3, f4)
  }

  /**
   * Build a dataset with four features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @return dataset with four features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag](data: Seq[(F1, F2, F3, F4)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4]) = {
    apply[F1, F2, F3, F4](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, f4name = DefaultFeatureNames.f4, data)
  }

  /**
   * Build a dataset with five features of specified types
   *
   * @param f1name 1st feature name
   * @param f2name 2nd feature name
   * @param f3name 3rd feature name
   * @param f4name 4th feature name
   * @param f5name 5th feature name
   * @param data   data
   * @param spark  spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @tparam F5 5th feature type
   * @return dataset with five features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag,
  F5 <: FeatureType : TypeTag](
    f1name: String, f2name: String, f3name: String, f4name: String, f5name: String, data: Seq[(F1, F2, F3, F4, F5)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4], Feature[F5]) = {
    val (f1, f2, f3, f4, f5) =
      (feature[F1](f1name), feature[F2](f2name), feature[F3](f3name), feature[F4](f4name), feature[F5](f5name))
    val schema = FeatureSparkTypes.toStructType(f1, f2, f3, f4, f5)
    (dataframe(schema, data), f1, f2, f3, f4, f5)
  }

  /**
   * Build a dataset with five features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @tparam F1 1st feature type
   * @tparam F2 2nd feature type
   * @tparam F3 3rd feature type
   * @tparam F4 4th feature type
   * @tparam F5 5th feature type
   * @return dataset with five features of specified types
   */
  def apply[F1 <: FeatureType : TypeTag,
  F2 <: FeatureType : TypeTag,
  F3 <: FeatureType : TypeTag,
  F4 <: FeatureType : TypeTag,
  F5 <: FeatureType : TypeTag](data: Seq[(F1, F2, F3, F4, F5)])
    (implicit spark: SparkSession): (DataFrame, Feature[F1], Feature[F2], Feature[F3], Feature[F4], Feature[F5]) = {
    apply[F1, F2, F3, F4, F5](
      f1name = DefaultFeatureNames.f1, f2name = DefaultFeatureNames.f2,
      f3name = DefaultFeatureNames.f3, f4name = DefaultFeatureNames.f4,
      f5name = DefaultFeatureNames.f5, data)
  }

  private val InitDate = new SimpleDateFormat("dd/MM/yy").parse("18/04/19")

  /**
   * Build a dataset with random features of specified size
   *
   * @param numOfRows number of rows to generate (must be positive)
   * @param spark     spark session
   * @return dataset with random features of specified size
   */
  // scalastyle:off parameter.number
  def random
  (
    numOfRows: Int = 10
  )(
    vectors: => Seq[OPVector] = RandomVector.sparse(RandomReal.normal[Real](), 10).limit(numOfRows),
    textLists: => Seq[TextList] = RandomList.ofTexts(RandomText.strings(0, 10), maxLen = 10).limit(numOfRows),
    dateLists: => Seq[DateList] = RandomList.ofDates(
      RandomIntegral.dates(InitDate, 1000, 1000000), maxLen = 10
    ).limit(numOfRows),
    dateTimeLists: => Seq[DateList] = RandomList.ofDateTimes(
      RandomIntegral.datetimes(InitDate, 1000, 1000000), maxLen = 10
    ).limit(numOfRows),
    geoLocations: => Seq[Geolocation] = RandomList.ofGeolocations.limit(numOfRows),
    base64Maps: => Seq[Base64Map] = RandomMap.of[Base64, Base64Map](RandomText.base64(5, 10), 0, 5).limit(numOfRows),
    binaryMaps: => Seq[BinaryMap] = RandomMap.ofBinaries(0.5, 0, 5).limit(numOfRows),
    comboBoxMaps: => Seq[ComboBoxMap] = RandomMap.of[ComboBox, ComboBoxMap](
      RandomText.comboBoxes(List("choice1", "choice2", "choice3")), 0, 5
    ).limit(numOfRows),
    currencyMaps: => Seq[CurrencyMap] = RandomMap.ofReals[Currency, CurrencyMap](
      RandomReal.poisson[Currency](5.0), 0, 5
    ).limit(numOfRows),
    dateMaps: => Seq[DateMap] = RandomMap.of(
      RandomIntegral.dates(InitDate, 1000, 1000000), 0, 5
    ).limit(numOfRows),
    dateTimeMaps: => Seq[DateTimeMap] = RandomMap.of(
      RandomIntegral.datetimes(InitDate, 1000, 1000000), 0, 5
    ).limit(numOfRows),
    emailMaps: => Seq[EmailMap] = RandomMap.of(
      RandomText.emailsOn(RandomStream.of(List("example.com", "test.com"))), 0, 5
    ).limit(numOfRows),
    idMaps: => Seq[IDMap] = RandomMap.of[ID, IDMap](RandomText.ids, 0, 5).limit(numOfRows),
    integralMaps: => Seq[IntegralMap] = RandomMap.of(RandomIntegral.integrals, 0, 5).limit(numOfRows),
    multiPickListMaps: => Seq[MultiPickListMap] = RandomMap.ofMultiPickLists(
      RandomMultiPickList.of(RandomText.countries, maxLen = 5), 0, 5
    ).limit(numOfRows),
    percentMaps: => Seq[PercentMap] = RandomMap.ofReals[Percent, PercentMap](
      RandomReal.normal[Percent](50, 5), 0, 5
    ).limit(numOfRows),
    phoneMaps: => Seq[PhoneMap] = RandomMap.of[Phone, PhoneMap](RandomText.phones, 0, 5).limit(numOfRows),
    pickListMaps: => Seq[PickListMap] = RandomMap.of[PickList, PickListMap](
      RandomText.pickLists(List("pick1", "pick2", "pick3")), 0, 5
    ).limit(numOfRows),
    realMaps: => Seq[RealMap] = RandomMap.ofReals[Real, RealMap](RandomReal.normal[Real](), 0, 5).limit(numOfRows),
    textAreaMaps: => Seq[TextAreaMap] = RandomMap.of[TextArea, TextAreaMap](
      RandomText.textAreas(0, 50), 0, 5
    ).limit(numOfRows),
    textMaps: => Seq[TextMap] = RandomMap.of[Text, TextMap](RandomText.strings(0, 10), 0, 5).limit(numOfRows),
    urlMaps: => Seq[URLMap] = RandomMap.of[URL, URLMap](RandomText.urls, 0, 5).limit(numOfRows),
    countryMaps: => Seq[CountryMap] = RandomMap.of[Country, CountryMap](RandomText.countries, 0, 5).limit(numOfRows),
    stateMaps: => Seq[StateMap] = RandomMap.of[State, StateMap](RandomText.states, 0, 5).limit(numOfRows),
    cityMaps: => Seq[CityMap] = RandomMap.of[City, CityMap](RandomText.cities, 0, 5).limit(numOfRows),
    postalCodeMaps: => Seq[PostalCodeMap] = RandomMap.of[PostalCode, PostalCodeMap](
      RandomText.postalCodes, 0, 5
    ).limit(numOfRows),
    streetMaps: => Seq[StreetMap] = RandomMap.of[Street, StreetMap](RandomText.streets, 0, 5).limit(numOfRows),
    geoLocationMaps: => Seq[GeolocationMap] = RandomMap.ofGeolocations[UniformGeolocation](
      RandomList.ofGeolocations, 0, 5
    ).limit(numOfRows),
    binaries: => Seq[Binary] = RandomBinary(0.5).limit(numOfRows),
    currencies: => Seq[Currency] = RandomReal.poisson[Currency](5.0).limit(numOfRows),
    dates: => Seq[Date] = RandomIntegral.dates(InitDate, 1000, 1000000).limit(numOfRows),
    dateTimes: => Seq[DateTime] = RandomIntegral.datetimes(InitDate, 1000, 1000000).limit(numOfRows),
    integrals: => Seq[Integral] = RandomIntegral.integrals.limit(numOfRows),
    percents: => Seq[Percent] = RandomReal.normal[Percent](50, 5).limit(numOfRows),
    reals: => Seq[Real] = RandomReal.normal[Real]().limit(numOfRows),
    realNNs: => Seq[RealNN] = RandomReal.normal[RealNN]().limit(numOfRows),
    multiPickLists: => Seq[MultiPickList] = RandomMultiPickList.of(RandomText.countries, maxLen = 5).limit(numOfRows),
    base64s: => Seq[Base64] = RandomText.base64(5, 10).limit(numOfRows),
    comboBoxes: => Seq[ComboBox] = RandomText.comboBoxes(List("choice1", "choice2", "choice3")).limit(numOfRows),
    emails: => Seq[Email] = RandomText.emailsOn(RandomStream.of(List("example.com", "test.com"))).limit(numOfRows),
    ids: => Seq[ID] = RandomText.ids.limit(numOfRows),
    phones: => Seq[Phone] = RandomText.phones.limit(numOfRows),
    pickLists: => Seq[PickList] = RandomText.pickLists(List("pick1", "pick2", "pick3")).limit(numOfRows),
    texts: => Seq[Text] = RandomText.base64(5, 10).limit(numOfRows),
    textAreas: => Seq[TextArea] = RandomText.textAreas(0, 50).limit(numOfRows),
    urls: => Seq[URL] = RandomText.urls.limit(numOfRows),
    countries: => Seq[Country] = RandomText.countries.limit(numOfRows),
    states: => Seq[State] = RandomText.states.limit(numOfRows),
    cities: => Seq[City] = RandomText.cities.limit(numOfRows),
    postalCodes: => Seq[PostalCode] = RandomText.postalCodes.limit(numOfRows),
    streets: => Seq[Street] = RandomText.streets.limit(numOfRows)
  )(implicit spark: SparkSession): (DataFrame, Array[Feature[_ <: FeatureType]]) = {

    require(numOfRows > 0, "Number of rows must be positive")

    val data: Array[Seq[FeatureType]] = Array(
      vectors, textLists, dateLists, dateTimeLists, geoLocations,
      base64Maps, binaryMaps, comboBoxMaps, currencyMaps, dateMaps,
      dateTimeMaps, emailMaps, idMaps, integralMaps, multiPickListMaps,
      percentMaps, phoneMaps, pickListMaps, realMaps, textAreaMaps,
      textMaps, urlMaps, countryMaps, stateMaps, cityMaps,
      postalCodeMaps, streetMaps, geoLocationMaps, binaries, currencies,
      dates, dateTimes, integrals, percents, reals, realNNs,
      multiPickLists, base64s, comboBoxes, emails, ids, phones,
      pickLists, texts, textAreas, urls, countries, states,
      cities, postalCodes, streets)

    this.apply(data: _*)(spark)
  }

  // scalastyle:on

  /**
   * Build a dataset with arbitrary amount features of specified types
   *
   * @param data  data
   * @param spark spark session
   * @return dataset with arbitrary amount  features of specified types
   */
  def apply(data: Seq[FeatureType]*)(implicit spark: SparkSession): (DataFrame, Array[Feature[_ <: FeatureType]]) = {
    val iterators = data.map(_.iterator).toArray
    val rows = ArrayBuffer.empty[Row]
    val featureValues = ArrayBuffer.empty[Array[FeatureType]]

    while (iterators.forall(_.hasNext)) {
      val vals: Array[FeatureType] = iterators.map(_.next())
      val sparkVals = vals.map(FeatureTypeSparkConverter.toSpark)
      rows += Row.fromSeq(sparkVals)
      featureValues += vals
    }

    val features: Array[Feature[_ <: FeatureType]] = featureValues.head.zipWithIndex.map { case (f, i) =>
      val wtt = FeatureType.featureTypeTag(f.getClass.getName).asInstanceOf[WeakTypeTag[FeatureType]]
      feature[FeatureType](name = s"f${i + 1}")(wtt)
    }.toArray

    val schema = StructType(features.map(FeatureSparkTypes.toStructField(_)))
    dataframeOfRows(schema, rows) -> features
  }

  private def dataframe[T <: Product](schema: StructType, data: Seq[T])(implicit spark: SparkSession): DataFrame = {
    val rows = data.map(p => Row.fromSeq(
      p.productIterator.toSeq.map { case f: FeatureType => FeatureTypeSparkConverter.toSpark(f) }
    ))
    dataframeOfRows(schema, rows)
  }

  private def dataframeOfRows(schema: StructType, data: Seq[Row])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    implicit val rowEncoder = RowEncoder(schema)
    data.toDF()
  }

  private def feature[T <: FeatureType](name: String)(implicit tt: WeakTypeTag[T]) =
    FeatureBuilder.fromRow[T](name)(tt).asPredictor

}
