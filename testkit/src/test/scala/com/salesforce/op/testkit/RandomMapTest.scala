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

import java.text.SimpleDateFormat

import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import com.salesforce.op.testkit.RandomList.{NormalGeolocation, UniformGeolocation}
import com.salesforce.op.testkit.RandomMap._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec}

import scala.language.postfixOps


@RunWith(classOf[JUnitRunner])
class RandomMapTest extends FlatSpec with TestCommon with Assertions {
  private val numTries = 10000
  private val rngSeed = 314159271828L

  private def check[D, T <: OPMap[D]](
    sut: RandomMap[D, T],
    minLen: Int, maxLen: Int,
    predicate: (D => Boolean) = (_: D) => true,
    samples: List[Map[String, D]] = Nil
  ) = {
    sut reset rngSeed

    val found = sut.next
    sut reset rngSeed
    val foundAfterReseed = sut.next
    withClue(s"generator reset did not work for $sut") {
      foundAfterReseed shouldBe found
    }

    sut reset rngSeed

    def segment = sut take numTries

    segment count (_.value.size < minLen) shouldBe 0
    segment count (_.value.size > maxLen) shouldBe 0

    def number(key: String) = try {
      key dropWhile (!Character.isDigit(_)) toInt
    } catch {
      case _: Exception => 0
    }

    segment foreach (map => map.value.toList.sortBy(kv => number(kv._1)) foreach {
      case (k, x) =>
        predicate(x) shouldBe true
    })
    sut reset rngSeed + 1
    (sut take samples.length map (_.value.toMap) toList) shouldBe samples
    sut reset rngSeed + 1
    (sut take samples.length map (_.value.toMap) toList) shouldBe samples
  }

  private def checkWithMapPredicate[D, T <: OPMap[D]](
    sut: RandomMap[D, T],
    minLen: Int, maxLen: Int,
    predicate: (T => Boolean) = (_: T) => true,
    samples: List[Map[String, D]] = Nil
  ) = {
    sut reset rngSeed

    val found = sut.next
    sut reset rngSeed
    val foundAfterReseed = sut.next
    withClue(s"generator reset did not work for $sut") {
      foundAfterReseed shouldBe found
    }

    sut reset rngSeed

    def segment = sut take numTries

    segment count (_.value.size < minLen) shouldBe 0
    segment count (_.value.size > maxLen) shouldBe 0

    def number(key: String) = try {
      key dropWhile (!Character.isDigit(_)) toInt
    } catch {
      case _: Exception => 0
    }

    segment foreach (map => predicate(map) shouldBe true)
    sut reset rngSeed + 1
    (sut take samples.length map (_.value.toMap) toList) shouldBe samples
    sut reset rngSeed + 1
    (sut take samples.length map (_.value.toMap) toList) shouldBe samples
  }

  Spec[Text, RandomMap[String, TextMap]] should "generate maps of texts" in {
    val sut = RandomMap.of[Text, TextMap](RandomText.strings(2, 5), 0, 4)
    check[String, TextMap](sut, 0, 4, s => s.length >= 2 && s.length < 5)
  }

  Spec[TextArea, RandomMap[String, TextAreaMap]] should "generate maps of texts" in {
    val sut = RandomMap.of[TextArea, TextAreaMap](RandomText.textAreas(3, 666), 0, 4)
    check[String, TextAreaMap](sut, 0, 4, s => s.length >= 3 && s.length < 666)
  }

  // scalastyle:off
  Spec[Email, RandomMap[String, EmailMap]] should "generate maps of emails" in {
    val sut = RandomMap.of[Email, EmailMap](RandomText.emails("kim.kr"), 0, 5) withKeys (i => "@" + (i + 1))
    check[String, EmailMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("@1" -> "kakins@kim.kr", "@2" -> "rheath@kim.kr"),
        Map(
          "@1" -> "rheffner@kim.kr",
          "@2" -> "Swamy.Hobbs@kim.kr",
          "@3" -> "Yvonne.Nelms@kim.kr"), // see http://www.legacy.com/obituaries/name/yvonne-nelms-obituary?pid=1000000185772062&view=guestbook
        Map(), Map(), Map(),
        Map("@1" -> "srickard@kim.kr"),
        Map(
          "@1" -> "Barbara.Amundson@kim.kr", // see http://www.karvonenfuneralhome.com/obituary/Barbara-Ann-Amundson/Rochester-MN/1640311
          "@2" -> "hbatchelor@kim.kr"),
        Map(
          "@1" -> "ttreadway@kim.kr",
          "@2" -> "Oscar.Tennant@kim.kr", // @see http://www.tyreefuneralhome.com/obituaries/Oscar-Tennant-35986/
          "@3" -> "Pria.Talbott@kim.kr",
          "@4" -> "lyee542@kim.kr")
      ))
  }
  // scalastyle:on

  Spec[Base64, RandomMap[String, Base64Map]] should "generate maps of base64" in {
    val sut = RandomMap.of[Base64, Base64Map](RandomText.base64(5, 10), 0, 4) withKeys ("B" +)

    check[String, Base64Map](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("B0" -> "c5PvXF2KmEQ=", "B1" -> "qpgrBYE="),
        Map("B0" -> "N4QTnOOTpw=="),
        Map("B0" -> "gXI9oSaKyUc="),
        Map("B0" -> "/wLPVC0="),
        Map("B0" -> "QnoMX4k=", "B1" -> "iDijm10=", "B2" -> "BiOXpFz2ZDM="),
        Map(),
        Map("B0" -> "qrZF/OZJBg==", "B1" -> "a6Hfijq5og==")
      ))
  }

  Spec[Phone, RandomMap[String, PhoneMap]] should "generate maps of phones" in {
    val sut = RandomMap.of[Phone, PhoneMap](RandomText.phones, 0, 4) withPrefix "ph#"

    check[String, PhoneMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("ph#0" -> "8585552361", "ph#1" -> "8175552810"),
        Map(),
        Map("ph#0" -> "7755550770"),
        Map("ph#0" -> "8045555039", "ph#1" -> "2765559392"),
        Map("ph#0" -> "2095554374"),
        Map("ph#0" -> "9715559825"),
        Map("ph#0" -> "5625558100", "ph#1" -> "9255552660", "ph#2" -> "6195551959")
      )
    )
  }

  Spec[ID, RandomMap[String, IDMap]] should "generate maps of IDs" in {
    val sut = RandomMap.of[ID, IDMap](RandomText.ids, 0, 4) withPrefix "_"

    check[String, IDMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("_0" -> "o7pKTgiQyojJQhvBxGyK", "_1" -> "yRiLnIJuT3v9WHfLuxbXa"),
        Map()
      )
    )
  }

  Spec[URL, RandomMap[String, URLMap]] should "generate maps of URLs" in {
    val sut = RandomMap.of[URL, URLMap](RandomText.urls, 0, 4) withPrefix "//"

    check[String, URLMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map(
          "//0" -> "https://guo0?Raf=%E0%A2%93%E2%89%9F&Car=%E7%BE%9E%C3%B4&Har=%CF%82%E9%9C%8D",
          "//1" -> "http://q5d?Ben=%EA%9B%B5%E3%96%98&Nin=%E4%A3%95%D3%B8"),
        Map(
          "//0" -> "https://4ty.q80d.lmi?Mys=%EB%A0%BC%E1%9A%81",
          "//1" -> "https://cupq.k29.y6cf?Suu=%E9%B6%86%E4%A9%AB&Vic=%E4%90%9A%E4%91%94&Son=%E8%BB%80%EB%A9%95"),
        Map()
      )
    )
  }

  Spec[Country, RandomMap[String, CountryMap]] should "generate maps of countries" in {
    val sut = RandomMap.of[Country, CountryMap](RandomText.countries, 0, 4)
    check[String, CountryMap](sut, 0, 4,
      predicate = _.length > 0,
      // scalastyle:off
      samples = List(
        Map("k0" -> "Slovenia", "k1" -> "Laurania"),
        Map("k0" -> "Gabon"),
        Map(),
        Map("k0" -> "Rhelasia"),
        Map(
          "k0" -> "Syldavia",
          "k1" -> "United States Of South Africa",
          "k2" -> "Chinese Federation"),
        Map("k0" -> "Westeros"),
        Map("k0" -> "Bolumbia"),
        Map("k0" -> "Groland", "k1" -> "Slovakia"),
        Map("k0" -> "Transia", "k1" -> "Unaudited Arab Emirates", "k2" -> "Dothraki")
      )
      // scalastyle:on
    )
  }

  Spec[State, RandomMap[String, StateMap]] should "generate maps of US states" in {
    val sut = RandomMap.of[State, StateMap](RandomText.states, 0, 4)
    check[String, StateMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "Georgia", "k1" -> "Idaho"),
        Map("k0" -> "Hawaii"),
        Map(),
        Map("k0" -> "Ohio"),
        Map("k0" -> "Indiana", "k1" -> "New Jersey", "k2" -> "South Carolina"),
        Map("k0" -> "Texas"),
        Map("k0" -> "Mississippi")
      )
    )
  }

  Spec[City, RandomMap[String, CityMap]] should "generate maps of CA cities" in {
    val sut = RandomMap.of[City, CityMap](RandomText.cities, 0, 4)
    check[String, CityMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "Baldwin Park", "k1" -> "Farmersville"),
        Map("k0" -> "Walnut Creek"),
        Map(),
        Map("k0" -> "Fontana"),
        Map(
          "k0" -> "West Sacramento",
          "k1" -> "Anaheim",
          "k2" -> "Bishop"),
        Map("k0" -> "Fresno"),
        Map("k0" -> "Blythe")
      )
    )
  }

  Spec[PostalCode, RandomMap[String, PostalCodeMap]] should "generate maps of US postal codes" in {
    val sut = RandomMap.of[PostalCode, PostalCodeMap](RandomText.postalCodes, 0, 4)
    check[String, PostalCodeMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "05659", "k1" -> "77361"),
        Map("k0" -> "34810"),
        Map(),
        Map("k0" -> "50884")
      )
    )
  }

  Spec[Street, RandomMap[String, StreetMap]] should "generate maps of San Jose Streets" in {
    val sut = RandomMap.of[Street, StreetMap](RandomText.streets, 0, 4)
    check[String, StreetMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "Cory Avenue", "k1" -> "Montgomery Street"),
        Map("k0" -> "Sunol Street"),
        Map(),
        Map("k0" -> "Harding Avenue")
      )
    )
  }

  Spec[Text, RandomMap[String, NameStats]] should "generate NameStats maps correctly" in {
    val sut = RandomMap.ofNameStats()
    checkWithMapPredicate[String, NameStats](sut,
      minLen = NameStats.Key.values.length,
      maxLen = NameStats.Key.values.length,
      predicate = { nameStats =>
        val allKeysPresent = NameStats.Key.values map { nameStats.value contains _.toString } forall identity
        val validNameIndicatorEntries = Seq(true, false)
          .map(bool => Some(bool.toString))
          .contains(nameStats.value.get(NameStats.Key.IsName.toString))
        val validGenderEntries = NameStats.GenderValue.values
          .map(enum => Some(enum.toString))
          .contains(nameStats.value.get(NameStats.Key.Gender.toString))
        allKeysPresent & validNameIndicatorEntries & validGenderEntries
      }
    )
  }

  Spec[PickList, RandomMap[String, PickListMap]] should "generate maps of picklists" in {
    val domain = "Bourbon"::"Cabernet"::"Beer"::Nil
    val sut = RandomMap.of[PickList, PickListMap](RandomText.pickLists(domain), 1, 4)
    check[String, PickListMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "Bourbon"),
        Map("k0" -> "Bourbon", "k1" -> "Bourbon", "k2" -> "Beer"),
        Map("k0" -> "Bourbon", "k1" -> "Beer", "k2" -> "Beer"),
        Map("k0" -> "Beer"),
        Map("k0" -> "Cabernet"),
        Map("k0" -> "Cabernet",
          "k1" -> "Bourbon",
          "k2" -> "Beer")
      )
    )
  }

  Spec[ComboBox, RandomMap[String, ComboBoxMap]] should "generate maps of comboboxes" in {
    val domain = "Bourbon"::"Cabernet"::"Beer"::Nil
    val sut = RandomMap.of[ComboBox, ComboBoxMap](RandomText.comboBoxes(domain), 1, 4)
    check[String, ComboBoxMap](sut, 0, 4,
      predicate = _.length > 0,
      samples = List(
        Map("k0" -> "Bourbon"),
        Map("k0" -> "Bourbon", "k1" -> "Bourbon", "k2" -> "Beer"),
        Map("k0" -> "Bourbon", "k1" -> "Beer", "k2" -> "Beer"),
        Map("k0" -> "Beer"),
        Map("k0" -> "Cabernet"),
        Map("k0" -> "Cabernet",
          "k1" -> "Bourbon",
          "k2" -> "Beer")
      )
    )
  }

  Spec[Binary, RandomMap[Boolean, BinaryMap]] should "generate maps of binaries" in {
    val sut = RandomMap.ofBinaries(0.25, 11, 22)
    check[BinaryMap#Element, BinaryMap](sut, 11, 22)
  }

  Spec[Long, RandomMap[Long, IntegralMap]] should "generate maps of longs" in {
    val sut = RandomMap.of(RandomIntegral.integrals, 1, 6) withPrefix "N"
    check[Long, IntegralMap](sut, 1, 6,
      samples = List(
        Map(
          "N0" -> -8519943704600931469L,
          "N1" -> 4942852721459949300L,
          "N2" -> 372559252054139265L),
        Map(
          "N0" -> -2069254063080111049L,
          "N1" -> 1344205618275271743L,
          "N2" -> 5799111682382721665L,
          "N3" -> 5172817544017718694L,
          "N4" -> -7848874697032793345L),
        Map(
          "N0" -> -4285035807136886576L,
          "N1" -> 6848983556037760905L,
          "N2" -> 7891868457571137672L,
          "N3" -> 417376114746839822L),
        Map(
          "N0" -> 3703355669373048704L,
          "N1" -> -8746575821146834884L,
          "N2" -> -268607759685629466L,
          "N3" -> 36168810987757931L)
      )
    )
  }

  private val df = new SimpleDateFormat("dd/MM/yy")

  Spec[Date, RandomMap[Long, DateMap]] should "generate maps of dates" in {
    val dates = RandomIntegral.dates(df.parse("01/01/2017"), 1000, 1000000)
    val sut = RandomMap.of(dates, 11, 22)
    var d0 = 0L
    check[Long, DateMap](sut, 11, 22, d => {
      val d1 = d0
      d0 = d
      d > d1
    })
  }

  Spec[DateTimeMap, RandomMap[Long, DateTimeMap]] should "generate maps of datetimes" in {
    val datetimes = RandomIntegral.datetimes(df.parse("01/01/2017"), 1000, 1000000)
    val sut = RandomMap.of(datetimes, 11, 22)
    var d0 = 0L
    check[Long, DateTimeMap](sut, 11, 22, d => {
      val d1 = d0
      d0 = d
      d > d1
    })
  }

  Spec[UniformGeolocation, RandomMap[Seq[Double], GeolocationMap]] should
    "generate maps of uniformly distributed geolocations" in {
    val sut = RandomMap.ofGeolocations[UniformGeolocation](RandomList.ofGeolocations, 2, 6)
    val segment = sut take numTries
    segment foreach (map => map.value.values.foreach { pos =>
      Geolocation.validate(pos(0), pos(1))
      pos.length shouldBe 3
    })
  }

  Spec[NormalGeolocation, RandomMap[Seq[Double], GeolocationMap]] should
    "generate maps of geolocations around given point" in {
    for {accuracy <- GeolocationAccuracy.values} {
      val geolocation = RandomMap.ofGeolocations(
        RandomList.ofGeolocationsNear(37.444136, 122.163160, accuracy), 1, 4)
      val segment = geolocation take numTries
      segment foreach (map => map.value.values.foreach { pos =>
        Geolocation.validate(pos(0), pos(1))
        pos.length shouldBe 3
      })
    }
  }

  Spec[Real, RandomMap[Double, RealMap]] should "generate maps of reals" in {
    val normal = RandomReal.normal[Real](5.0, 2.0)
    val sut = RandomMap.ofReals[Real, RealMap](normal, 1, 4) withKeys (i => "" + ('a' + i).toChar)
    check[Double, RealMap](sut, 1, 3,
      samples = List(
        Map("a" -> 3.3573821018748577),
        Map("a" -> 6.155000792586161),
        Map("a" -> 4.006348243684868, "b" -> 5.683036228303376, "c" -> 5.832671498716051)
      )
    )
  }

  Spec[Currency, RandomMap[Double, CurrencyMap]] should "generate maps of cash values" in {
    val poisson = RandomReal.poisson[Currency](5.0)
    val sut = RandomMap.ofReals[Currency, CurrencyMap](poisson, 1, 6) withPrefix "$"
    check[Double, CurrencyMap](sut, 1, 6,
      samples = List(
        Map("$0" -> 6.0, "$1" -> 7.0, "$2" -> 3.0),
        Map("$4" -> 5.0, "$0" -> 5.0, "$3" -> 4.0, "$2" -> 5.0, "$1" -> 5.0)
      )
    )
  }

  Spec[Percent, RandomMap[Double, PercentMap]] should "generate maps of percent values" in {
    val poisson = RandomReal.poisson[Percent](5.0)
    val sut = RandomMap.ofReals[Percent, PercentMap](poisson, 1, 6) withPrefix "%%"
    check[Double, PercentMap](sut, 1, 6,
      samples = List(
        Map("%%0" -> 6.0, "%%1" -> 7.0, "%%2" -> 3.0),
        Map("%%4" -> 5.0, "%%3" -> 4.0, "%%2" -> 5.0, "%%1" -> 5.0, "%%0" -> 5.0)
      )
    )
  }

  Spec[MultiPickList, RandomMap[Set[String], MultiPickListMap]] should "generate maps of mpls" in {
    val sut = RandomMap.ofMultiPickLists(RandomMultiPickList.of(RandomText.countries, maxLen = 5), 0, 4)
    // scalastyle:off
    check[Set[String], MultiPickListMap](sut, 0, 4, s => s.size < 5,
      samples = List(
        Map(
          "k0" -> Set("Laurania", "Cobra Island", "Gabon", "Seychelles"),
          "k1" -> Set("Rhelasia", "Switzerland")),
        Map("k0" -> Set("Chinese Federation", "Senegal", "Westeros", "Glovania"),
          "k1" -> Set("Tanzania", "Groland", "Slovakia")),
        Map("k0" -> Set(), "k1" -> Set(), "k2" -> Set()),
        Map("k0" -> Set()),
        Map(),
        Map("k0" -> Set("Neutral Zone", "Phaic Tăn", "Atlantis")),
        Map(
          "k0" -> Set("Ragaan", "Sierra Gordo", "Molvanîa", "Zephyria"),
          "k1" -> Set("Nambutu"), "k2" -> Set("Vespugia", "Tazbekistan"))
      )
    )
    // scalastyle:on
  }

}
