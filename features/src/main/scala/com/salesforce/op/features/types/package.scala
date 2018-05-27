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

package com.salesforce.op.features

import com.daodecode.scalaj.collection.immutable._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
 * A collection of decorators that allow converting Scala and Java types into Feature Types
 */
package object types extends FeatureTypeSparkConverters {

  // String
  implicit class StringConversions(val v: String) extends AnyVal {
    def toText: Text = new Text(Option(v))
    def toEmail: Email = new Email(Option(v))
    def toBase64: Base64 = new Base64(Option(v))
    def toPhone: Phone = new Phone(Option(v))
    def toID: ID = new ID(Option(v))
    def toURL: URL = new URL(Option(v))
    def toTextArea: TextArea = new TextArea(Option(v))
    def toPickList: PickList = new PickList(Option(v))
    def toComboBox: ComboBox = new ComboBox(Option(v))
    def toCountry: Country = new Country(Option(v))
    def toState: State = new State(Option(v))
    def toPostalCode: PostalCode = new PostalCode(Option(v))
    def toCity: City = new City(Option(v))
    def toStreet: Street = new Street(Option(v))
  }
  implicit class OptStringConversions(val v: Option[String]) extends AnyVal {
    def toText: Text = new Text(v)
    def toEmail: Email = new Email(v)
    def toBase64: Base64 = new Base64(v)
    def toPhone: Phone = new Phone(v)
    def toID: ID = new ID(v)
    def toURL: URL = new URL(v)
    def toTextArea: TextArea = new TextArea(v)
    def toPickList: PickList = new PickList(v)
    def toComboBox: ComboBox = new ComboBox(v)
    def toCountry: Country = new Country(v)
    def toState: State = new State(v)
    def toPostalCode: PostalCode = new PostalCode(v)
    def toCity: City = new City(v)
    def toStreet: Street = new Street(v)
  }

  // Numerics
  implicit class JDoubleConversions(val v: java.lang.Double) extends AnyVal {
    def toReal: Real = new Real(Option(v).map(_.toDouble))
    def toCurrency: Currency = new Currency(Option(v).map(_.toDouble))
    def toPercent: Percent = new Percent(Option(v).map(_.toDouble))
  }
  implicit class JFloatConversions(val v: java.lang.Float) extends AnyVal {
    def toReal: Real = new Real(Option(v).map(_.toDouble))
    def toCurrency: Currency = new Currency(Option(v).map(_.toDouble))
    def toPercent: Percent = new Percent(Option(v).map(_.toDouble))
  }
  implicit class JIntegerConversions(val v: java.lang.Integer) extends AnyVal {
    def toReal: Real = new Real(Option(v).map(_.toDouble))
    def toIntegral: Integral = new Integral(Option(v).map(_.toLong))
    def toDate: Date = new Date(Option(v).map(_.toLong))
    def toDateTime: DateTime = new DateTime(Option(v).map(_.toLong))
  }
  implicit class JLongConversions(val v: java.lang.Long) extends AnyVal {
    def toReal: Real = new Real(Option(v).map(_.toDouble))
    def toIntegral: Integral = new Integral(Option(v).map(_.toLong))
    def toDate: Date = new Date(Option(v).map(_.toLong))
    def toDateTime: DateTime = new DateTime(Option(v).map(_.toLong))
  }
  implicit class JBooleanConversions(val v: java.lang.Boolean) extends AnyVal {
    def toBinary: Binary = new Binary(v)
  }
  implicit class OptDoubleConversions(val v: Option[Double]) extends AnyVal {
    def toReal: Real = new Real(v)
    def toRealNN(default: Double): RealNN = new RealNN(v.getOrElse(default))
    def toCurrency: Currency = new Currency(v)
    def toPercent: Percent = new Percent(v)
    def toBinary: Binary = new Binary(v.map(_ != 0.0))
  }
  implicit class OptFloatConversions(val v: Option[Float]) extends AnyVal {
    def toReal: Real = new Real(v.map(_.toDouble))
    def toRealNN(default: Float): RealNN = new RealNN(v.getOrElse(default))
    def toCurrency: Currency = new Currency(v.map(_.toDouble))
    def toPercent: Percent = new Percent(v.map(_.toDouble))
    def toBinary: Binary = new Binary(v.map(_ != 0f))
  }
  implicit class OptIntConversions(val v: Option[Int]) extends AnyVal {
    def toReal: Real = new Real(v.map(_.toDouble))
    def toIntegral: Integral = new Integral(v.map(_.toLong))
    def toDate: Date = new Date(v.map(_.toLong))
    def toDateTime: DateTime = new DateTime(v.map(_.toLong))
    def toBinary: Binary = new Binary(v.map(_ != 0))
  }
  implicit class OptLongConversions(val v: Option[Long]) extends AnyVal {
    def toReal: Real = new Real(v.map(_.toDouble))
    def toIntegral: Integral = new Integral(v)
    def toDate: Date = new Date(v)
    def toDateTime: DateTime = new DateTime(v)
    def toBinary: Binary = new Binary(v.map(_ != 0L))
  }
  implicit class OptBooleanConversions(val v: Option[Boolean]) extends AnyVal {
    def toBinary: Binary = new Binary(v)
  }
  implicit class DoubleConversions(val v: Double) extends AnyVal {
    def toReal: Real = new Real(v)
    def toRealNN: RealNN = new RealNN(v)
    def toCurrency: Currency = new Currency(v)
    def toPercent: Percent = new Percent(v)
    def toBinary: Binary = new Binary(v != 0.0)
  }
  implicit class FloatConversions(val v: Float) extends AnyVal {
    def toReal: Real = new Real(v)
    def toRealNN: RealNN = new RealNN(v)
    def toCurrency: Currency = new Currency(v)
    def toPercent: Percent = new Percent(v)
    def toBinary: Binary = new Binary(v != 0f)
  }
  implicit class IntConversions(val v: Int) extends AnyVal {
    def toReal: Real = new Real(v)
    def toRealNN: RealNN = new RealNN(v)
    def toIntegral: Integral = new Integral(v.toLong)
    def toDate: Date = new Date(v.toLong)
    def toDateTime: DateTime = new DateTime(v.toLong)
    def toBinary: Binary = new Binary(v != 0)
  }
  implicit class LongConversions(val v: Long) extends AnyVal {
    def toReal: Real = new Real(v)
    def toRealNN: RealNN = new RealNN(v)
    def toIntegral: Integral = new Integral(v)
    def toDate: Date = new Date(v)
    def toDateTime: DateTime = new DateTime(v)
    def toBinary: Binary = new Binary(v != 0L)
  }
  implicit class BooleanConversions(val v: Boolean) extends AnyVal {
    def toBinary: Binary = new Binary(v)
  }

  // Collections
  implicit class SeqDoubleConversions(val v: Seq[Double]) extends AnyVal {
    def toReal: Seq[Real] = v.map(_.toReal)
    def toRealNN: Seq[RealNN] = v.map(_.toRealNN)
    def toOPVector: OPVector = new OPVector(Vectors.dense(v.toArray))
    def toGeolocation: Geolocation = new Geolocation(v)
  }
  implicit class SeqLongConversions(val v: Seq[Long]) extends AnyVal {
    def toDateList: DateList = new DateList(v)
    def toDateTimeList: DateTimeList = new DateTimeList(v)
    def toReal: Seq[Real] = v.map(_.toReal)
    def toRealNN: Seq[RealNN] = v.map(_.toRealNN)
    def toIntegral: Seq[Integral] = v.map(_.toIntegral)
  }
  implicit class SeqStringConversions(val v: Seq[String]) extends AnyVal {
    def toTextList: TextList = new TextList(v)
    def toMultiPickList: MultiPickList = new MultiPickList(v.toSet)
    def toText: Seq[Text] = v.map(_.toText)
  }
  implicit class Tup3DoubleConversions(val v: (Double, Double, Double)) extends AnyVal {
    def toGeolocation: Geolocation = new Geolocation(v)
  }

  // Sets
  implicit class SetStringConversions(val v: Set[String]) extends AnyVal {
    def toTextList: TextList = new TextList(v.toSeq)
    def toMultiPickList: MultiPickList = new MultiPickList(v)
  }

  // Vectors
  implicit class VectorConversions(val v: Vector) extends AnyVal {
    def toOPVector: OPVector = new OPVector(v)
  }

  // Maps
  implicit class JMapStringConversions(val v: java.util.Map[String, String]) extends AnyVal {
    def toTextMap: TextMap = new TextMap(Option(v).map(_.deepAsScalaImmutable).getOrElse(Map.empty))
  }
  implicit class JMapSetConversions(val v: java.util.Map[String, java.util.HashSet[String]]) extends AnyVal {
    def toMultiPickListMap: MultiPickListMap =
      new MultiPickListMap(Option(v).map(_.deepAsScalaImmutable).getOrElse(Map.empty))
  }
  implicit class JMapLongConversions(val v: java.util.Map[String, java.lang.Long]) extends AnyVal {
    def toIntegralMap: IntegralMap = new IntegralMap(Option(v).map(_.deepAsScalaImmutable).getOrElse(Map.empty))
  }
  implicit class JMapDoubleConversions(val v: java.util.Map[String, java.lang.Double]) extends AnyVal {
    def toRealMap: RealMap = new RealMap(Option(v).map(_.deepAsScalaImmutable).getOrElse(Map.empty))
  }
  implicit class JMapBooleanConversions(val v: java.util.Map[String, java.lang.Boolean]) extends AnyVal {
    def toBinaryMap: BinaryMap = new BinaryMap(Option(v).map(_.deepAsScalaImmutable).getOrElse(Map.empty))
  }
  implicit class MapStringConversions(val v: Map[String, String]) extends AnyVal {
    def toTextMap: TextMap = new TextMap(v)
    def toEmailMap: EmailMap = new EmailMap(v)
    def toBase64Map: Base64Map = new Base64Map(v)
    def toPhoneMap: PhoneMap = new PhoneMap(v)
    def toIDMap: IDMap = new IDMap(v)
    def toURLMap: URLMap = new URLMap(v)
    def toTextAreaMap: TextAreaMap = new TextAreaMap(v)
    def toPickListMap: PickListMap = new PickListMap(v)
    def toComboBoxMap: ComboBoxMap = new ComboBoxMap(v)
    def toCountryMap: CountryMap = new CountryMap(v)
    def toStateMap: StateMap = new StateMap(v)
    def toCityMap: CityMap = new CityMap(v)
    def toPostalCodeMap: PostalCodeMap = new PostalCodeMap(v)
    def toStreetMap: StreetMap = new StreetMap(v)
  }
  implicit class MapSetConversions(val v: Map[String, Set[String]]) extends AnyVal {
    def toMultiPickListMap: MultiPickListMap = new MultiPickListMap(v)
  }
  implicit class MapLongConversions(val v: Map[String, Long]) extends AnyVal {
    def toIntegralMap: IntegralMap = new IntegralMap(v)
  }
  implicit class MapDoubleConversions(val v: Map[String, Double]) extends AnyVal {
    def toRealMap: RealMap = new RealMap(v)
    def toPrediction: Prediction = new Prediction(v)
  }
  implicit class MapBooleanConversions(val v: Map[String, Boolean]) extends AnyVal {
    def toBinaryMap: BinaryMap = new BinaryMap(v)
  }
  implicit class MapGeolocationConversions(val v: Map[String, Seq[Double]]) extends AnyVal {
    def toGeolocationMap: GeolocationMap = new GeolocationMap(v)
  }
  implicit def intMapToRealMap(m: IntegralMap#Value): RealMap#Value = m.map { case (k, v) => k -> v.toDouble }
  implicit def booleanToRealMap(m: BinaryMap#Value): RealMap#Value = m.map { case (k, b) => k -> (if (b) 1.0 else 0.0) }

}
