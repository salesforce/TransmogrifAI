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
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.tuples.RichTuple._
import com.twitter.algebird._
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

import scala.reflect.runtime.universe._

@RunWith(classOf[JUnitRunner])
class MonoidAggregatorDefaultsTest extends FlatSpec with TestCommon {

  import MonoidAggregatorDefaults._

  val doubleBase = Seq(Option(-1.0), None, Option(0.25), Option(0.1), Option(0.7), Option(2.5))
  val longBase = Seq(Option(1L), None, Option(11110L), Option(250L), Option(10L), Option(1234324234L))
  val booleanBase = Seq(Option(true), None, Option(false), Option(true), None)
  val stringSetBase = Seq(Set("a", "b", "c"), Set("d", "e"), Set("d", "a"), Set.empty[String])
  val pickListBase = Seq(Some("A"), Some("B"), Some("B"), Some("A"), None, Some("A"), None, Some("C"))
  val pickListBaseBalanced = Seq(Some("C"), Some("B"), None, Some("D"))
  val textBase = Seq(
    Option("My name is Joe"),
    Option("And I work in a button factory"),
    Option("One day my boss said to me"),
    Option("Are you busy, I said no"),
    Option("Then push the button with your right hand"),
    None, None
  )
  val doubleMapBase = Seq(
    Map("a" -> 1.0, "b" -> 1.0),
    Map("b" -> 1.0, "c" -> 1.0, "e" -> 0.0),
    Map("a" -> 0.5, "d" -> 0.3, "e" -> 0.0),
    Map.empty[String, Double]
  )
  val longMapBase = Seq(
    Map("a" -> 1L, "b" -> 1L),
    Map("b" -> 1L, "c" -> 1L, "e" -> 0L),
    Map("a" -> 2L, "d" -> 2L, "e" -> 0L),
    Map.empty[String, Long]
  )
  val booleanMapBase = Seq(
    Map("a" -> true, "b" -> false, "c" -> true),
    Map("a" -> true, "b" -> false, "c" -> false),
    Map.empty[String, Boolean]
  )
  val textMapBase = Seq(
    Map("a" -> "Mo' money", "b" -> "Been spending all their lives", "d" -> ""),
    Map("a" -> "mo' problems", "b" -> "livin' in the gangsta's paradise", "c" -> "Does this take you back?"),
    Map("d" -> ""),
    Map.empty[String, String]
  )
  val stringSetMapBase = Seq(
    Map("a" -> Set("a", "b"), "b" -> Set("a", "b")),
    Map("a" -> Set("c"), "d" -> Set.empty[String]),
    Map("b" -> Set("a", "c"), "c" -> Set("a")),
    Map.empty[String, Set[String]]
  )
  val geoBase = Seq(
    Seq(32.4, -100.2, 3.0),
    Seq(38.6, -110.4, 2.0),
    Seq.empty[Double],
    Seq(40.1, -120.3, 4.0),
    Seq(42.5, -95.4, 4.0),
    Seq.empty[Double],
    Seq(45.0, -105.5, 4.0)
  )
  val geoBaseEmpty = Seq(Seq.empty[Double], Seq.empty[Double], Seq.empty[Double])

  private val realTestSeq = doubleBase.map(Real(_))
  private val realNNTestSeq = doubleBase.map(_.toRealNN(0.0))
  private val binaryTestSeq = booleanBase.map(Binary(_))
  private val integralTestSeq = longBase.map(Integral(_))
  private val percentTestSeq = doubleBase.map(new Percent(_))
  private val currencyTestSeq = doubleBase.map(new Currency(_))
  private val dateTestSeq = longBase.map(new Date(_))
  private val dateTimeTestSeq = longBase.map(new DateTime(_))

  private val multiPickListTestSeq = stringSetBase.map(new MultiPickList(_))

  private val base64TestSeq = textBase.map(new Base64(_))
  private val comboBoxTestSeq = textBase.map(new ComboBox(_))
  private val emailTestSeq = textBase.map(new Email(_))
  private val idTestSeq = textBase.map(new ID(_))
  private val phoneTestSeq = textBase.map(new Phone(_))
  private val textTestSeq = textBase.map(new Text(_))
  private val textAreaTestSeq = textBase.map(new TextArea(_))
  private val urlTestSeq = textBase.map(new URL(_))
  private val countryTestSeq = textBase.map(new Country(_))
  private val stateTestSeq = textBase.map(new State(_))
  private val cityTestSeq = textBase.map(new City(_))
  private val postalCodeTestSeq = textBase.map(new PostalCode(_))
  private val streetTestSeq = textBase.map(new Street(_))

  private val pickListTestSeq = pickListBase.map(new PickList(_))
  private val pickListTestSeqBalanced = pickListBaseBalanced.map(new PickList(_))

  private val textListTestSeq = Seq("a", "b", "c").map { s => new TextList(Seq(s)) }
  private val dateListTestSeq = longBase.flatten.map { l => new DateList(Seq(l)) }
  private val dateTimeListTestSeq = longBase.flatten.map { l => new DateTimeList(Seq(l)) }

  private val realMapTestSeq = doubleMapBase.map(new RealMap(_))
  private val binaryMapTestSeq = booleanMapBase.map(new BinaryMap(_))
  private val integralMapTestSeq = longMapBase.map(new IntegralMap(_))
  private val percentMapTestSeq = doubleMapBase.map(new PercentMap(_))
  private val currencyMapTestSeq = doubleMapBase.map(new CurrencyMap(_))
  private val dateMapTestSeq = longMapBase.map(new DateMap(_))
  private val dateTimeMapTestSeq = longMapBase.map(new DateTimeMap(_))

  private val base64MapTestSeq = textMapBase.map(new Base64Map(_))
  private val comboBoxMapTestSeq = textMapBase.map(new ComboBoxMap(_))
  private val emailMapTestSeq = textMapBase.map(new EmailMap(_))
  private val idMapTestSeq = textMapBase.map(new IDMap(_))
  private val phoneMapTestSeq = textMapBase.map(new PhoneMap(_))
  private val pickListMapTestSeq = textMapBase.map(new PickListMap(_))
  private val textMapTestSeq = textMapBase.map(new TextMap(_))
  private val textAreaMapTestSeq = textMapBase.map(new TextAreaMap(_))
  private val urlMapTestSeq = textMapBase.map(new URLMap(_))
  private val countryMapTestSeq = textMapBase.map(new CountryMap(_))
  private val stateMapTestSeq = textMapBase.map(new StateMap(_))
  private val cityMapTestSeq = textMapBase.map(new CityMap(_))
  private val postalCodeMapTestSeq = textMapBase.map(new PostalCodeMap(_))
  private val streetMapTestSeq = textMapBase.map(new StreetMap(_))

  private val multiPickListMapTestSeq = stringSetMapBase.map(new MultiPickListMap(_))

  private val vectorTestSeq = Seq(Array(0.1, 0.2), Array(1.0), Array(0.2)).map(Vectors.dense(_).toOPVector)

  Spec(MonoidAggregatorDefaults.getClass) should "throw an error on unknown feature type" in {
    assertThrows[IllegalArgumentException](
      aggregatorOf[FeatureType](weakTypeTag[Double].asInstanceOf[WeakTypeTag[FeatureType]])
    )
  }

  Spec[SumNumeric[_, _]] should "work" in {
    val expectedDoubleResult = Option(doubleBase.flatten.sum)
    val expectedLongResult = Option(longBase.flatten.sum)

    // These are all defaults.
    assertDefaultAggr(realTestSeq, expectedDoubleResult)
    assertDefaultAggr(realNNTestSeq, expectedDoubleResult)
    assertDefaultAggr(currencyTestSeq, expectedDoubleResult)
    assertDefaultAggr(integralTestSeq, expectedLongResult)
  }

  Spec[MinMaxNumeric[_, _]] should "work" in {
    val maxDouble = Option(doubleBase.flatten.max)
    val minDouble = Option(doubleBase.flatten.min)
    val maxLong = Option(longBase.flatten.max)
    val minLong = Option(longBase.flatten.min)

    assertAggr(MaxReal, realTestSeq, maxDouble)
    assertAggr(MaxRealNN, realNNTestSeq, maxDouble)
    assertAggr(MaxCurrency, currencyTestSeq, maxDouble)
    assertAggr(MaxIntegral, integralTestSeq, maxLong)
    // These two are defaults.
    assertDefaultAggr(dateTestSeq, maxLong)
    assertDefaultAggr(dateTimeTestSeq, maxLong)
    assertAggr(MinReal, realTestSeq, minDouble)
    assertAggr(MinRealNN, realNNTestSeq, minDouble)
    assertAggr(MinCurrency, currencyTestSeq, minDouble)
    assertAggr(MinIntegral, integralTestSeq, minLong)
    assertAggr(MinDate, dateTestSeq, minLong)
    assertAggr(MinDateTime, dateTimeTestSeq, minLong)

  }

  Spec[MeanDouble[_]] should "work" in {
    val (meanDouble, meanPercent) = {
      val seqToSum = doubleBase.flatten
      val meanDouble = seqToSum.map(_ / seqToSum.length).sum
      val percentDouble = seqToSum.map {
        case p if p < 0.0 => 0.0
        case p if p > 1.0 => 1.0 / seqToSum.length
        case p => p / seqToSum.length
      }.sum

      (Option(meanDouble), Option("%.2f".format(percentDouble).toDouble))
    }
    val meanRealNN = Option(realNNTestSeq.flatMap(_.value).sum / realNNTestSeq.length)
    assertAggr(MeanReal, realTestSeq, meanDouble)
    assertAggr(MeanRealNN, realNNTestSeq, meanRealNN)
    assertAggr(MeanCurrency, currencyTestSeq, meanDouble)
    assertAggr(MeanPercent, percentTestSeq, meanPercent)
  }

  Spec(UnionMultiPickList.getClass) should "work" in {
    assertDefaultAggr(multiPickListTestSeq, Set("a", "b", "c", "d", "e"))
  }

  Spec[ConcatTextWithSeparator[_]] should "work" in {
    val expectedResultComma = Option(textBase.flatten.mkString(","))
    val expectedResultSpace = Option(textBase.flatten.mkString(" "))

    assertDefaultAggr(base64TestSeq, expectedResultComma)
    assertDefaultAggr(comboBoxTestSeq, expectedResultComma)
    assertDefaultAggr(emailTestSeq, expectedResultComma)
    assertDefaultAggr(idTestSeq, expectedResultComma)
    assertDefaultAggr(phoneTestSeq, expectedResultComma)
    assertDefaultAggr(textTestSeq, expectedResultSpace)
    assertDefaultAggr(textAreaTestSeq, expectedResultSpace)
    assertDefaultAggr(urlTestSeq, expectedResultComma)
    assertDefaultAggr(countryTestSeq, expectedResultComma)
    assertDefaultAggr(stateTestSeq, expectedResultComma)
    assertDefaultAggr(cityTestSeq, expectedResultComma)
    assertDefaultAggr(postalCodeTestSeq, expectedResultComma)
    assertDefaultAggr(streetTestSeq, expectedResultComma)
  }

  Spec(ModePickList.getClass) should "work" in {
    val expectedResultPickList = Some("A")
    val expectedResultPickListBalanced = Some("B")

    assertDefaultAggr(pickListTestSeq, expectedResultPickList)
    assertDefaultAggr(pickListTestSeqBalanced, expectedResultPickListBalanced)
  }

  Spec[ConcatList[_, _]] should "work" in {
    assertDefaultAggr(textListTestSeq, Seq("a", "b", "c"))
    assertDefaultAggr(dateListTestSeq, longBase.flatten)
    assertDefaultAggr(dateTimeListTestSeq, longBase.flatten)
  }

  Spec[MinMaxList[_, _]] should "work" in {
    val expectedMin: Seq[Long] = Seq(longBase.flatten.min)
    val expectedMax: Seq[Long] = Seq(longBase.flatten.max)

    assertAggr(MaxDateList, dateListTestSeq, expectedMax)
    assertAggr(MaxDateTimeList, dateTimeListTestSeq, expectedMax)
    assertAggr(MinDateList, dateListTestSeq, expectedMin)
    assertAggr(MinDateTimeList, dateTimeListTestSeq, expectedMin)
  }

  /**
   * This test looks different for a few reasons
   * 1) Custom aggregator does different things to different indices of the Geolocation list so not easy to
   * express the expected result in terms of the initial geolocation set
   * 2) Expected result comes from online calculator (http://www.geomidpoint.com/), so compare with an error tolerance
   */
  Spec(GeolocationMidpoint.getClass) should "find the geographic centroid" in {
    val eps = 0.01 // error tolerance

    def assertGeo(inputs: Seq[Geolocation], expected: Seq[Double]): Unit = {
      val aggregatedValue = aggregatorOf[Geolocation].apply(inputs.map(Event(0L, _))).value
      val errors = expected.zip(aggregatedValue).map(f => math.abs(f._2 - f._1)) zipWithIndex

      errors foreach { case (e, i) =>
        withClue(s"Failed at $i: expected about $expected, actual $aggregatedValue, err=$e") {
          e should be < eps
        }
      }
    }
    assertGeo(inputs = geoBase.map(Geolocation(_)), expected = Seq(40.04, -106.33, 0.0))
    assertGeo(inputs = Seq(geoBase.head.toGeolocation), expected = geoBase.head)
  }

  Spec(LogicalOr.getClass) should "work" in {
    assertDefaultAggr(binaryTestSeq, Option(true))
    assertAggr(LogicalOr, binaryTestSeq, Option(true))
  }

  private def s[T](x: T) = x match {
    case a: Array[_] => a mkString ","
    case b => String.valueOf(b)
  }

  private def checkMonoid[T](
    m: Monoid[T],
    values: Seq[T],
    compare: (T, T) => Boolean = (_: T) == (_: T)): Unit = {

    for {
      ia <- values.indices
      ib <- values.indices
      ic <- values.indices
    } {
      val (a, b, c) = (values(ia), values(ib), values(ic))
      m.plus(m.zero, a) shouldBe a
      m.plus(a, m.zero) shouldBe a
      val ab = m.plus(a, b)
      val bc = m.plus(b, c)
      val left = m.plus(ab, c)
      val right = m.plus(a, bc)

      withClue(s"\nExpecting ${s(left)}\n be equal ${s(right)}\n($ia, $ib, $ic)") {
        compare(left, right) shouldBe true
      }
    }
  }

  private val logicalValues = Some(true) :: Some(false) :: None :: Nil

  Spec(LogicalOr.getClass) should "be a monoid" in {
    checkMonoid(new LogicalOrMonoid {}.monoid, logicalValues)
  }

  Spec(LogicalXor.getClass) should "work" in {
    assertAggr(LogicalXor, binaryTestSeq, Option(false))
  }

  Spec(LogicalXor.getClass) should "be a monoid" in {
    checkMonoid(new LogicalXorMonoid {}.monoid, logicalValues)
  }

  Spec(LogicalAnd.getClass) should "work" in {
    assertAggr(LogicalAnd, binaryTestSeq, Option(false))
  }

  Spec(LogicalAnd.getClass) should "be a monoid" in {
    checkMonoid(new LogicalAndMonoid {}.monoid, logicalValues)
  }

  Spec[UnionSumNumericMap[_, _]] should "work" in {
    val expectedDoubleRes = Map("a" -> 1.5, "b" -> 2.0, "c" -> 1.0, "d" -> 0.3, "e" -> 0.0)
    val expectedLongRes = Map("a" -> 3L, "b" -> 2L, "c" -> 1L, "d" -> 2L, "e" -> 0L)
    assertDefaultAggr(realMapTestSeq, expectedDoubleRes)
    assertDefaultAggr(currencyMapTestSeq, expectedDoubleRes)
    assertDefaultAggr(integralMapTestSeq, expectedLongRes)
  }

  Spec[UnionMeanDoubleMap[_]] should "work" in {
    val expectedDoubleRes = Map("a" -> 0.75, "b" -> 1.0, "c" -> 1.0, "d" -> 0.3, "e" -> 0.0)
    assertAggr(UnionMeanRealMap, realMapTestSeq, expectedDoubleRes)
    assertAggr(UnionMeanCurrencyMap, currencyMapTestSeq, expectedDoubleRes)
    assertDefaultAggr(percentMapTestSeq, expectedDoubleRes)
  }

  Spec(UnionGeolocationMidpointMap.getClass) should "correctly find the geographic centroid" in {
    val geoMapBase = Seq(
      Map("a" -> Seq(38.4, -110.2, 3.0)),
      Map("a" -> Seq(38.6, -110.4, 2.0)),
      Map("a" -> Seq.empty[Double]),
      Map("a" -> Seq(39.1, -110.3, 4.0)),
      Map("a" -> Seq(38.5, -110.45, 4.0)),
      Map("a" -> Seq.empty[Double]),
      Map("a" -> Seq(39.0, -109.55, 4.0)),
      Map("b" -> Seq(43.8, -108.7, 2.0)),
      Map("b" -> Seq(43.9, -109.6, 3.0)),
      Map("b" -> Seq(43.4, -109.3, 2.0)),
      Map("b" -> Seq.empty[Double]),
      Map("c" -> Seq(40.4, -116.3, 2.0)),
      Map("c" -> Seq.empty[Double])
    )

    val geoMapTestSeq = geoMapBase.map(new GeolocationMap(_))

    val eps = 0.01 // error tolerance
    val expectedGeo = Map(
      "a" -> List(38.72, -110.18, 10.0),
      "b" -> List(43.7, -109.2, 10.0),
      "c" -> List(40.4, -116.3, 2.0)
    )

    val aggregatedValue = aggregatorOf[GeolocationMap].apply(geoMapTestSeq.map(Event(0L, _))).value
    val errorMap = expectedGeo.map {
      case (k, v) => k -> v.zip(aggregatedValue(k)).map(f => math.abs(f._2 - f._1))
    }

    for {(k, v) <- errorMap} {
      withClue(s"Failed at $k: expected ${expectedGeo(k)}, actual ${aggregatedValue(k)}, err=$v") {
        v.forall(_ < eps) shouldBe true
      }
    }
  }

  private def distance(xs: Array[Double], ys: Array[Double]): Double = {
    val xys = xs zip ys
    math.sqrt((0.0 /: xys) { case (s, (x, y)) => s + (x - y) * (x - y) })
  }

  private def prettyClose(xs: Array[Double], ys: Array[Double]) =
    distance(xs take 5, ys take 5) < 0.01

  private def checkGeolocationsOn(p1: Geolocation, p2: Geolocation, p3: Geolocation): Unit = {
    val m = GeolocationMidpoint.monoid

    val a = GeolocationMidpoint.prepare(p1)
    val b = GeolocationMidpoint.prepare(p2)
    val c = GeolocationMidpoint.prepare(p3)
    checkGeopoints(m, a, b, c)
  }

  // scalastyle:off
  private def checkGeopoints(m: Monoid[Array[Double]], a: Array[Double], b: Array[Double], c: Array[Double]) = {
    val aa = m.plus(a, a)
    val ab = m.plus(a, b)
    val bb = m.plus(b, b)
    val bc = m.plus(b, c)
    val cc = m.plus(c, c)
    val aa_b = m.plus(aa, b)
    val a_ab = m.plus(a, ab)
    prettyClose(aa_b, a_ab) shouldBe true
    val ab_c = m.plus(ab, c)
    val a_bc = m.plus(a, bc)
    withClue(s"oops,\n     ${s(ab_c)}\n  vs ${s(a_bc)}\ntried ${s(a)}\n      ${s(b)}\n      ${s(c)}") {
      prettyClose(ab_c, a_bc) shouldBe true
    }

    def checkOn(what: Seq[Array[Double]]) = checkMonoid(m, what, prettyClose)

    checkOn(a :: b :: c :: Nil)
  }

  Spec(GeolocationMidpoint.getClass) should "be a monoid" in {

    checkGeolocationsOn(
      Geolocation(0, 0, GeolocationAccuracy.Address),
      Geolocation(0, 90, GeolocationAccuracy.Address),
      Geolocation(90, 0, GeolocationAccuracy.Address)
    )

    def checkOn(what: Seq[Array[Double]]) =
      checkMonoid(GeolocationMidpoint.monoid, what, prettyClose)

    // The following code is temporarily commented out because we have a circular dependency
    //    val generator = RandomList.ofGeolocations
    //    generator.reset(4087688721L)
    //    val sampleSpots = generator limit 20
    //    val point1 = sampleSpots(0)
    //    val point2 = sampleSpots(1)
    //    val point3 = sampleSpots(6)
    //    checkGeolocationsOn(point1, point2, point3)
    //
    //    val testData = sampleSpots map (v => Geolocations.prepare(Event(0, v)))
    //
    //    checkOn(testData)
  }
  // scalastyle:on

  Spec[UnionMinMaxNumericMap[_, _]] should "work" in {
    val expectedDoubleMax = Map("a" -> 1.0, "b" -> 1.0, "c" -> 1.0, "d" -> 0.3, "e" -> 0.0)
    val expectedDoubleMin = Map("a" -> 0.5, "b" -> 1.0, "c" -> 1.0, "d" -> 0.3, "e" -> 0.0)
    val expectedLongMax = Map("a" -> 2L, "b" -> 1L, "c" -> 1L, "d" -> 2L, "e" -> 0L)
    val expectedLongMin = Map("a" -> 1L, "b" -> 1L, "c" -> 1L, "d" -> 2L, "e" -> 0L)

    assertAggr(UnionMaxRealMap, realMapTestSeq, expectedDoubleMax)
    assertAggr(UnionMaxCurrencyMap, currencyMapTestSeq, expectedDoubleMax)
    assertAggr(UnionMaxIntegralMap, integralMapTestSeq, expectedLongMax)
    assertDefaultAggr(dateMapTestSeq, expectedLongMax)
    assertDefaultAggr(dateTimeMapTestSeq, expectedLongMax)
    assertAggr(UnionMinRealMap, realMapTestSeq, expectedDoubleMin)
    assertAggr(UnionMinCurrencyMap, currencyMapTestSeq, expectedDoubleMin)
    assertAggr(UnionMinIntegralMap, integralMapTestSeq, expectedLongMin)
    assertAggr(UnionMinDateMap, dateMapTestSeq, expectedLongMin)
    assertAggr(UnionMinDateTimeMap, dateTimeMapTestSeq, expectedLongMin)
  }

  Spec[UnionConcatTextMap[_]] should "work" in {
    val expectedCommaRes = Map(
      "a" -> "Mo' money,mo' problems",
      "b" -> "Been spending all their lives,livin' in the gangsta's paradise",
      "c" -> "Does this take you back?",
      "d" -> ""
    )
    val expectedSpaceRes = Map(
      "a" -> "Mo' money mo' problems",
      "b" -> "Been spending all their lives livin' in the gangsta's paradise",
      "c" -> "Does this take you back?",
      "d" -> ""
    )
    assertDefaultAggr(base64MapTestSeq, expectedCommaRes)
    assertDefaultAggr(comboBoxMapTestSeq, expectedCommaRes)
    assertDefaultAggr(emailMapTestSeq, expectedCommaRes)
    assertDefaultAggr(idMapTestSeq, expectedCommaRes)
    assertDefaultAggr(phoneMapTestSeq, expectedCommaRes)
    assertDefaultAggr(pickListMapTestSeq, expectedCommaRes)
    assertDefaultAggr(textMapTestSeq, expectedSpaceRes)
    assertDefaultAggr(textAreaMapTestSeq, expectedSpaceRes)
    assertDefaultAggr(urlMapTestSeq, expectedCommaRes)
    assertDefaultAggr(countryMapTestSeq, expectedCommaRes)
    assertDefaultAggr(stateMapTestSeq, expectedCommaRes)
    assertDefaultAggr(cityMapTestSeq, expectedCommaRes)
    assertDefaultAggr(postalCodeMapTestSeq, expectedCommaRes)
    assertDefaultAggr(streetMapTestSeq, expectedCommaRes)
  }

  Spec(UnionBinaryMap.getClass) should "work" in {
    assertDefaultAggr(binaryMapTestSeq, Map("a" -> true, "b" -> false, "c" -> true))
  }

  Spec[UnionSetMap[_]] should "work" in {
    val expectedRes =
      Map("a" -> Set("a", "b", "c"), "b" -> Set("a", "b", "c"), "c" -> Set("a"), "d" -> Set.empty[String])

    assertDefaultAggr(multiPickListMapTestSeq, expectedRes)
  }

  Spec(CombineVector.getClass) should "work" in {
    assertDefaultAggr(vectorTestSeq, Vectors.dense(Array(0.1, 0.2, 1.0, 0.2)))
  }

  Spec[CustomMonoidAggregator[_]] should "work" in {
    val customAgg = new CustomMonoidAggregator[Real](zero = None, associativeFn = (r1, r2) => (r1 -> r2).map(_ + _))
    assertAggr(customAgg, realTestSeq, Option(doubleBase.flatten.sum))
  }


  /**
   * Helper method for asserting default aggregators.
   */
  private def assertDefaultAggr[A <: FeatureType : WeakTypeTag, B](testSeq: Seq[A], expectedResult: B): Unit =
    assertAggr(aggregatorOf[A], testSeq, expectedResult)

  /**
   * Helper method for asserting implemented aggregators
   */
  private def assertAggr[A <: FeatureType : WeakTypeTag, B](
    aggregator: MonoidAggregator[Event[A], _, A],
    testSeq: Seq[A],
    expectedResult: B
  ): Unit = {
    val events = testSeq.map(Event(0L, _))
    val aggregatedValue = aggregator(events).value.asInstanceOf[B]
    aggregatedValue shouldEqual expectedResult
  }


}
