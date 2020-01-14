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

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.twitter.algebird.Monoid
import com.twitter.algebird.Operators._
import org.apache.lucene.geo.GeoUtils
import org.apache.spark.ml.linalg.DenseVector
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Assertion, PropSpec}

import scala.reflect.runtime.universe.TypeTag
import scala.util.Try


@RunWith(classOf[JUnitRunner])
class FeatureTypeValueTest extends PropSpec with PropertyChecks with TestCommon {

  // Value generation
  private final val binaryGen = Gen.oneOf(false, true)
  private final val doubleGen = Gen.choose(Double.MinValue, Double.MaxValue)
  private final val longGen = Gen.choose(Long.MinValue, Long.MaxValue)
  private final val textGen = Gen.alphaNumStr

  // Vector generation
  private final val vectorGen = Gen.nonEmptyContainerOf[Array, Double](doubleGen).map(v => new DenseVector(v))

  // Set generation
  private final val setGen = Gen.nonEmptyContainerOf[Set, String](textGen)

  // List generation
  private final val longListGen = Gen.nonEmptyListOf(longGen)
  private final val textListGen = Gen.nonEmptyListOf(textGen)

  // Geolocation generation
  private final val maxLat = GeoUtils.MAX_LAT_INCL
  private final val maxLon = GeoUtils.MAX_LON_INCL
  private final val geoGen = for {
    lat <- Gen.choose(-maxLat, maxLat)
    lon <- Gen.choose(-maxLon, maxLon)
    acc <- Gen.choose(0, GeolocationAccuracy.values.length - 1)
  } yield List[Double](lat, lon, acc.toDouble)

  // Map generation
  private def mapGen[T](valGen: Gen[T], keyGen: Gen[String] = textGen): Gen[Map[String, T]] = {
    Gen.nonEmptyMap(for {k <- keyGen; v <- valGen} yield (k, v))
  }
  private final val binaryMapGen = mapGen(binaryGen)
  private final val doubleMapGen = mapGen(doubleGen)
  private final val longMapGen = mapGen(longGen)
  private final val textMapGen = mapGen(textGen)
  private final val setMapGen = mapGen(setGen)
  private final val geoMapGen = mapGen(geoGen)
  private final val longListMapGen = mapGen(longListGen) // Currently unused?
  private final val textListMapGen = mapGen(textListGen) // Currently unused?
  private final val predictionGen = mapGen(keyGen = Gen.oneOf(Seq(Prediction.Keys.PredictionName)), valGen = doubleGen)

  property("OPNumeric types should correctly wrap their corresponding Option types") {
    forAll { (x: Option[Double]) =>
      checkOptionVals(Real(x), x)
      checkOptionVals(RealNN(x.getOrElse(0.0)), if (x.isEmpty) Some(0.0) else x)
      checkOptionVals(Percent(x), x)
      checkOptionVals(Currency(x), x)
    }
    forAll { (x: Option[Long]) =>
      checkOptionVals(Integral(x), x)
      checkOptionVals(Date(x), x)
      checkOptionVals(DateTime(x), x)
    }
    forAll { (x: Option[Boolean]) => checkOptionVals(Binary(x), x) }
  }

  property("Text types should correctly wrap their corresponding Option types") {
    forAll { (x: Option[String]) =>
      checkOptionVals(Text(x), x)
      checkOptionVals(Email(x), x)
      checkOptionVals(Base64(x), x)
      checkOptionVals(Phone(x), x)
      checkOptionVals(ID(x), x)
      checkOptionVals(URL(x), x)
      checkOptionVals(TextArea(x), x)
      checkOptionVals(PickList(x), x)
      checkOptionVals(ComboBox(x), x)
      checkOptionVals(State(x), x)
      checkOptionVals(Country(x), x)
      checkOptionVals(City(x), x)
      checkOptionVals(PostalCode(x), x)
      checkOptionVals(Street(x), x)
    }
  }

  property("OPVector types should correctly wrap their corresponding types") {
    forAll(vectorGen) { x => checkVals(OPVector(x), x) }
  }

  property("OPList types should correctly wrap their corresponding types") {
    implicit val textListMonoid: Monoid[TextList] = TextList.monoid

    forAll(geoGen) { x => checkVals(Geolocation(x), x) }
    forAll(textListGen) { x =>
      checkVals(TextList(x), x)
      val tl = TextList(x)
      tl + tl shouldBe TextList(x ++ x) // Test the monoid too
    }
    forAll(longListGen) { x =>
      checkVals(DateList(x), x)
      checkVals(DateTimeList(x), x)
    }
  }

  property("OPSet types should correctly wrap their corresponding types") {
    forAll(setGen) { x => checkVals(MultiPickList(x), x) }
  }

  property("OPMap types should correctly wrap their corresponding types") {
    forAll(binaryMapGen) { x => checkVals(BinaryMap(x), x) }
    forAll(doubleMapGen) { x =>
      checkVals(RealMap(x), x)
      checkVals(CurrencyMap(x), x)
      checkVals(PercentMap(x), x)
      checkVals(RealMap(x), x)
    }
    forAll(longMapGen) { x =>
      checkVals(IntegralMap(x), x)
      checkVals(DateMap(x), x)
      checkVals(DateTimeMap(x), x)
    }
    forAll(textMapGen) { x =>
      checkVals(TextMap(x), x)
      checkVals(EmailMap(x), x)
      checkVals(Base64Map(x), x)
      checkVals(PhoneMap(x), x)
      checkVals(IDMap(x), x)
      checkVals(URLMap(x), x)
      checkVals(TextAreaMap(x), x)
      checkVals(PickListMap(x), x)
      checkVals(ComboBoxMap(x), x)
      checkVals(StateMap(x), x)
      checkVals(CountryMap(x), x)
      checkVals(CityMap(x), x)
      checkVals(PostalCodeMap(x), x)
      checkVals(StreetMap(x), x)
      checkVals(NameStats(x), x)
    }
    forAll(setMapGen) { x => checkVals(MultiPickListMap(x), x) }
    forAll(geoMapGen) { x => checkVals(GeolocationMap(x), x) }
    forAll(predictionGen) { x => checkVals(new Prediction(x), x) }
    forAll(doubleGen) { x => checkVals(Prediction(x), Map(Prediction.Keys.PredictionName -> x)) }
    forAll(doubleMapGen) { x =>
      val error = intercept[NonNullableEmptyException](new Prediction(x))
      error.getMessage shouldBe
        s"Prediction cannot be empty: value map must contain '${Prediction.Keys.PredictionName}' key"
    }
    forAll(vectorGen) { v =>
      val a = v.toArray
      val rawPred = a.zipWithIndex.map { case (v, i) => s"${Prediction.Keys.RawPredictionName}_$i" -> v }
      val prob = a.zipWithIndex.map { case (v, i) => s"${Prediction.Keys.ProbabilityName}_$i" -> v }
      val pred = Prediction.Keys.PredictionName -> a.head
      val expected = (rawPred ++ prob).toMap + pred
      checkVals(Prediction(a.head, v, v), expected)
      val fullPred = Prediction(a.head, a, a)
      checkVals(fullPred, expected)
      fullPred.prediction shouldBe a.head
      fullPred.probability shouldBe a
      fullPred.rawPrediction shouldBe a
    }

  }

  /**
   * Helper method with assertions for checking that the Option-containing OP types correctly wrap their values
   *
   * @param feature OP type object
   * @param value   Value that should be wrapped in OP type object
   * @tparam FT Feature type (OP type)
   * @tparam VT Value type (which should be wrapped in OP type)
   */
  private def checkOptionVals[FT <: FeatureType, VT <: Option[_]]
  (
    feature: FT, value: VT
  )(implicit vtt: TypeTag[FT#Value]): Assertion = {
    // Split out some specific functionality for testing the toDouble function in our OPNumeric types
    (feature, value) match {
      case (f: Binary, v: Option[_]) =>
        f.toDouble shouldBe v.map(x => if (x == true) 1.0 else 0.0)
      case (f: OPNumeric[_], v) =>
        f.toDouble shouldBe v
      case (f: Text, v) =>
        f.value shouldBe v // Nothing extra to check here yet
      case _ =>
        fail("Option types should only be passed to to OPNumeric and Text types")
    }
    feature.value shouldBe value
    feature.nonEmpty shouldBe value.nonEmpty
    checkTypeTags[FT]
  }

  /**
   * Helper method with assertions for checking that the non-Option OP types correctly wrap their values
   *
   * @param feature OP type object
   * @param value   Value that should be wrapped in OP type object
   * @param vtt     type tag of a feature type value (Real#Value, Text#Value etc
   * @tparam FT Feature type (OP type)
   * @tparam VT Value type (which should be wrapped in OP type)
   */
  private def checkVals[FT <: FeatureType, VT](feature: FT, value: VT)(implicit vtt: TypeTag[FT#Value]): Assertion = {
    feature.value shouldBe value
    feature.nonEmpty shouldBe true
    checkTypeTags[FT]
  }

  /**
   * Checks feature value type tags lookup functions
   *
   * @param vtt feature value type tag
   * @tparam FT feature type (OP type)
   */
  private def checkTypeTags[FT <: FeatureType](implicit vtt: TypeTag[FT#Value]): Assertion = {
    withClue(s"Feature value type ${vtt.tpe} (dealised: ${ReflectionUtils.dealisedTypeName(vtt.tpe)}): ") {
      val tt = Try(FeatureType.featureValueTypeTag(ReflectionUtils.dealisedTypeName(vtt.tpe)))
      if (tt.isFailure) fail(tt.failed.get)
      tt.get.tpe =:= vtt.tpe shouldBe true
      FeatureType.isFeatureValueType(vtt) shouldBe true
    }
  }
}
