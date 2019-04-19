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

    val iterators = Array(
      vectors.iterator, textLists.iterator, dateLists.iterator, dateTimeLists.iterator, geoLocations.iterator,
      base64Maps.iterator, binaryMaps.iterator, comboBoxMaps.iterator, currencyMaps.iterator, dateMaps.iterator,
      dateTimeMaps.iterator, emailMaps.iterator, idMaps.iterator, integralMaps.iterator, multiPickListMaps.iterator,
      percentMaps.iterator, phoneMaps.iterator, pickListMaps.iterator, realMaps.iterator, textAreaMaps.iterator,
      textMaps.iterator, urlMaps.iterator, countryMaps.iterator, stateMaps.iterator, cityMaps.iterator,
      postalCodeMaps.iterator, streetMaps.iterator, geoLocationMaps.iterator, binaries.iterator, currencies.iterator,
      dates.iterator, dateTimes.iterator, integrals.iterator, percents.iterator, reals.iterator, realNNs.iterator,
      multiPickLists.iterator, base64s.iterator, comboBoxes.iterator, emails.iterator, ids.iterator, phones.iterator,
      pickLists.iterator, texts.iterator, textAreas.iterator, urls.iterator, countries.iterator, states.iterator,
      cities.iterator, postalCodes.iterator, streets.iterator)

    val data = ArrayBuffer.empty[AllFeatureTypesRow]

    while (iterators.forall(_.hasNext)) {
      val vals: Array[FeatureType] = iterators.map(_.next())
      data += AllFeatureTypesRow(
        vector = vals(0).asInstanceOf[OPVector],
        textList = vals(1).asInstanceOf[TextList],
        dateList = vals(2).asInstanceOf[DateList],
        dateTimeList = vals(3).asInstanceOf[DateTimeList],
        geoLocation = vals(4).asInstanceOf[Geolocation],
        base64Map = vals(5).asInstanceOf[Base64Map],
        binaryMap = vals(6).asInstanceOf[BinaryMap],
        comboBoxMap = vals(7).asInstanceOf[ComboBoxMap],
        currencyMap = vals(8).asInstanceOf[CurrencyMap],
        dateMap = vals(9).asInstanceOf[DateMap],
        dateTimeMap = vals(10).asInstanceOf[DateTimeMap],
        emailMap = vals(11).asInstanceOf[EmailMap],
        idMap = vals(12).asInstanceOf[IDMap],
        integralMap = vals(13).asInstanceOf[IntegralMap],
        multiPickListMap = vals(14).asInstanceOf[MultiPickListMap],
        percentMap = vals(15).asInstanceOf[PercentMap],
        phoneMap = vals(16).asInstanceOf[PhoneMap],
        pickListMap = vals(17).asInstanceOf[PickListMap],
        realMap = vals(18).asInstanceOf[RealMap],
        textAreaMap = vals(19).asInstanceOf[TextAreaMap],
        textMap = vals(20).asInstanceOf[TextMap],
        urlMap = vals(21).asInstanceOf[URLMap],
        countryMap = vals(22).asInstanceOf[CountryMap],
        stateMap = vals(23).asInstanceOf[StateMap],
        cityMap = vals(24).asInstanceOf[CityMap],
        postalCodeMap = vals(25).asInstanceOf[PostalCodeMap],
        streetMap = vals(26).asInstanceOf[StreetMap],
        geoLocationMap = vals(27).asInstanceOf[GeolocationMap],
        binary = vals(28).asInstanceOf[Binary],
        currency = vals(29).asInstanceOf[Currency],
        date = vals(30).asInstanceOf[Date],
        dateTime = vals(31).asInstanceOf[DateTime],
        integral = vals(32).asInstanceOf[Integral],
        percent = vals(33).asInstanceOf[Percent],
        real = vals(34).asInstanceOf[Real],
        realNN = vals(35).asInstanceOf[RealNN],
        multiPickList = vals(36).asInstanceOf[MultiPickList],
        base64 = vals(37).asInstanceOf[Base64],
        comboBox = vals(38).asInstanceOf[ComboBox],
        email = vals(39).asInstanceOf[Email],
        id = vals(40).asInstanceOf[ID],
        phone = vals(41).asInstanceOf[Phone],
        pickList = vals(42).asInstanceOf[PickList],
        text = vals(43).asInstanceOf[Text],
        textArea = vals(44).asInstanceOf[TextArea],
        url = vals(45).asInstanceOf[URL],
        country = vals(46).asInstanceOf[Country],
        state = vals(47).asInstanceOf[State],
        city = vals(48).asInstanceOf[City],
        postalCode = vals(49).asInstanceOf[PostalCode],
        street = vals(50).asInstanceOf[Street]
      )
    }
    val features = Array[Feature[_ <: FeatureType]](
      feature[OPVector]("vector"),
      feature[TextList]("textList"),
      feature[DateList]("dateList"),
      feature[DateList]("dateTimeList"),
      feature[Geolocation]("geoLocation"),
      feature[Base64Map]("base64Map"),
      feature[BinaryMap]("binaryMap"),
      feature[ComboBoxMap]("comboBoxMap"),
      feature[CurrencyMap]("currencyMap"),
      feature[DateMap]("dateMap"),
      feature[DateTimeMap]("dateTimeMap"),
      feature[EmailMap]("emailMap"),
      feature[IDMap]("idMap"),
      feature[IntegralMap]("integralMap"),
      feature[MultiPickListMap]("multiPickListMap"),
      feature[PercentMap]("percentMap"),
      feature[PhoneMap]("phoneMap"),
      feature[PickListMap]("pickListMap"),
      feature[RealMap]("realMap"),
      feature[TextAreaMap]("textAreaMap"),
      feature[TextMap]("textMap"),
      feature[URLMap]("urlMap"),
      feature[CountryMap]("countryMap"),
      feature[StateMap]("stateMap"),
      feature[CityMap]("cityMap"),
      feature[PostalCodeMap]("postalCodeMap"),
      feature[StreetMap]("streetMap"),
      feature[GeolocationMap]("geoLocationMap"),
      feature[Binary]("binary"),
      feature[Currency]("currency"),
      feature[Date]("date"),
      feature[DateTime]("dateTime"),
      feature[Integral]("integral"),
      feature[Percent]("percent"),
      feature[Real]("real"),
      feature[RealNN]("realNN"),
      feature[MultiPickList]("multiPickList"),
      feature[Base64]("base64"),
      feature[ComboBox]("comboBox"),
      feature[Email]("email"),
      feature[ID]("id"),
      feature[Phone]("phone"),
      feature[PickList]("pickList"),
      feature[Text]("text"),
      feature[TextArea]("textArea"),
      feature[URL]("url"),
      feature[Country]("country"),
      feature[State]("state"),
      feature[City]("city"),
      feature[PostalCode]("postalCode"),
      feature[Street]("street"))

    val schema = StructType(features.map(FeatureSparkTypes.toStructField(_)))
    dataframe(schema, data) -> features
  }
  // scalastyle:on


  private def dataframe[T <: Product](schema: StructType, data: Seq[T])(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    implicit val rowEncoder = RowEncoder(schema)

    data.map(p => Row.fromSeq(
      p.productIterator.toSeq.map { case f: FeatureType => FeatureTypeSparkConverter.toSpark(f) }
    )).toDF()
  }

  private def feature[T <: FeatureType](name: String)(implicit tt: WeakTypeTag[T]) =
    FeatureBuilder.fromRow[T](name)(tt).asPredictor

  private case class AllFeatureTypesRow
  (
    vector: OPVector,
    textList: TextList,
    dateList: DateList,
    dateTimeList: DateList,
    geoLocation: Geolocation,
    base64Map: Base64Map,
    binaryMap: BinaryMap,
    comboBoxMap: ComboBoxMap,
    currencyMap: CurrencyMap,
    dateMap: DateMap,
    dateTimeMap: DateTimeMap,
    emailMap: EmailMap,
    idMap: IDMap,
    integralMap: IntegralMap,
    multiPickListMap: MultiPickListMap,
    percentMap: PercentMap,
    phoneMap: PhoneMap,
    pickListMap: PickListMap,
    realMap: RealMap,
    textAreaMap: TextAreaMap,
    textMap: TextMap,
    urlMap: URLMap,
    countryMap: CountryMap,
    stateMap: StateMap,
    cityMap: CityMap,
    postalCodeMap: PostalCodeMap,
    streetMap: StreetMap,
    geoLocationMap: GeolocationMap,
    binary: Binary,
    currency: Currency,
    date: Date,
    dateTime: DateTime,
    integral: Integral,
    percent: Percent,
    real: Real,
    realNN: RealNN,
    multiPickList: MultiPickList,
    base64: Base64,
    comboBox: ComboBox,
    email: Email,
    id: ID,
    phone: Phone,
    pickList: PickList,
    text: Text,
    textArea: TextArea,
    url: URL,
    country: Country,
    state: State,
    city: City,
    postalCode: PostalCode,
    street: Street
  )
}
