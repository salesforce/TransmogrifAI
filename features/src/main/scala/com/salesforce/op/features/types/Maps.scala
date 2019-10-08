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

import org.apache.spark.ml.linalg.Vector

/**
 * Map of text values
 *
 * @param value map of text values
 */
class TextMap(val value: Map[String, String]) extends OPMap[String]
object TextMap {
  def apply(value: Map[String, String]): TextMap = new TextMap(value)
  def empty: TextMap = FeatureTypeDefaults.TextMap
}

/**
 * Map of email values
 *
 * @param value map of email values
 */
class EmailMap(value: Map[String, String]) extends TextMap(value)
object EmailMap {
  def apply(value: Map[String, String]): EmailMap = new EmailMap(value)
  def empty: EmailMap = FeatureTypeDefaults.EmailMap
}

/**
 * Map of base64 binary encoded values
 *
 * @param value map of base64 binary encoded values
 */
class Base64Map(value: Map[String, String]) extends TextMap(value)
object Base64Map {
  def apply(value: Map[String, String]): Base64Map = new Base64Map(value)
  def empty: Base64Map = FeatureTypeDefaults.Base64Map
}

/**
 * Map of phone values
 *
 * @param value map of phone values
 */
class PhoneMap(value: Map[String, String]) extends TextMap(value)
object PhoneMap {
  def apply(value: Map[String, String]): PhoneMap = new PhoneMap(value)
  def empty: PhoneMap = FeatureTypeDefaults.PhoneMap
}

/**
 * Map of ID values
 *
 * @param value map of ID values
 */
class IDMap(value: Map[String, String]) extends TextMap(value)
object IDMap {
  def apply(value: Map[String, String]): IDMap = new IDMap(value)
  def empty: IDMap = FeatureTypeDefaults.IDMap
}

/**
 * Map of URL values
 *
 * @param value map of URL values
 */
class URLMap(value: Map[String, String]) extends TextMap(value)
object URLMap {
  def apply(value: Map[String, String]): URLMap = new URLMap(value)
  def empty: URLMap = FeatureTypeDefaults.URLMap
}

/**
 * Map containing information related to a particular name.
 *
 * @param value map of keys to values, where keys are one of the following:
 * - "original"
 * - "isName"
 * - "firstName"
 * - "lastName"
 * - "gender"
 */
class NameMap(value: Map[String, String]) extends TextMap(value) {
  import NameMap.Keys._

  def isName: Boolean = value.getOrElse(IsNameIndicator, "false") == "true"
  def isMale: Boolean = value.getOrElse(Gender, "") == "male"
  def isFemale: Boolean = value.getOrElse(Gender, "") == "female"
}
object NameMap {
  object Keys {
    val OriginalName = "original"
    val IsNameIndicator = "isName"
    val FirstName = "firstName"
    val LastName = "lastName"
    val Gender = "gender"
  }
  object BooleanStrings {
    val True = "true"
    val False = "false"
  }


  def apply(value: Map[String, String]): NameMap = new NameMap(value)
  def empty: NameMap = FeatureTypeDefaults.NameMap
}

/**
 * Map of text area values
 *
 * @param value map of text area values
 */
class TextAreaMap(value: Map[String, String]) extends TextMap(value)
object TextAreaMap {
  def apply(value: Map[String, String]): TextAreaMap = new TextAreaMap(value)
  def empty: TextAreaMap = FeatureTypeDefaults.TextAreaMap
}

/**
 * Map of picklist values
 *
 * @param value map of picklist values
 */
class PickListMap(value: Map[String, String]) extends TextMap(value) with SingleResponse
object PickListMap {
  def apply(value: Map[String, String]): PickListMap = new PickListMap(value)
  def empty: PickListMap = FeatureTypeDefaults.PickListMap
}

/**
 * Map of combobox values
 *
 * @param value map of combobox values
 */
class ComboBoxMap(value: Map[String, String]) extends TextMap(value)
object ComboBoxMap {
  def apply(value: Map[String, String]): ComboBoxMap = new ComboBoxMap(value)
  def empty: ComboBoxMap = FeatureTypeDefaults.ComboBoxMap
}

/**
 * Map of binary values
 *
 * @param value map of binary values
 */
class BinaryMap(val value: Map[String, Boolean]) extends OPMap[Boolean] with NumericMap with SingleResponse {
  def toDoubleMap: Map[String, Double] = value
}
object BinaryMap {
  def apply(value: Map[String, Boolean]): BinaryMap = new BinaryMap(value)
  def empty: BinaryMap = FeatureTypeDefaults.BinaryMap
}

/**
 * Map of integral values
 *
 * @param value map of integral values
 */
class IntegralMap(val value: Map[String, Long]) extends OPMap[Long] with NumericMap {
  def toDoubleMap: Map[String, Double] = value
}
object IntegralMap {
  def apply(value: Map[String, Long]): IntegralMap = new IntegralMap(value)
  def empty: IntegralMap = FeatureTypeDefaults.IntegralMap
}

/**
 * Map of real values
 *
 * @param value map of real values
 */
class RealMap(val value: Map[String, Double]) extends OPMap[Double] with NumericMap {
  def toDoubleMap: Map[String, Double] = value
}
object RealMap {
  def apply(value: Map[String, Double]): RealMap = new RealMap(value)
  def empty: RealMap = FeatureTypeDefaults.RealMap
}

/**
 * Map of percent values
 *
 * @param value map of percent values
 */
class PercentMap(value: Map[String, Double]) extends RealMap(value)
object PercentMap {
  def apply(value: Map[String, Double]): PercentMap = new PercentMap(value)
  def empty: PercentMap = FeatureTypeDefaults.PercentMap
}

/**
 * Map of currency values
 *
 * @param value map of currency values
 */
class CurrencyMap(value: Map[String, Double]) extends RealMap(value)
object CurrencyMap {
  def apply(value: Map[String, Double]): CurrencyMap = new CurrencyMap(value)
  def empty: CurrencyMap = FeatureTypeDefaults.CurrencyMap
}

/**
 * Map of date values
 *
 * @param value map of date values
 */
class DateMap(value: Map[String, Long]) extends IntegralMap(value)
object DateMap {
  def apply(value: Map[String, Long]): DateMap = new DateMap(value)
  def empty: DateMap = FeatureTypeDefaults.DateMap
}

/**
 * Map of date & time values
 *
 * @param value map of date & time values
 */
class DateTimeMap(value: Map[String, Long]) extends DateMap(value)
object DateTimeMap {
  def apply(value: Map[String, Long]): DateTimeMap = new DateTimeMap(value)
  def empty: DateTimeMap = FeatureTypeDefaults.DateTimeMap
}

/**
 * Map of multi picklist values
 *
 * @param value map of multi picklist values
 */
class MultiPickListMap(val value: Map[String, Set[String]]) extends OPMap[Set[String]] with MultiResponse
object MultiPickListMap {
  def apply(value: Map[String, Set[String]]): MultiPickListMap = new MultiPickListMap(value)
  def empty: MultiPickListMap = FeatureTypeDefaults.MultiPickListMap
}

/**
 * Map of country values
 *
 * @param value map of country values
 */
class CountryMap(value: Map[String, String]) extends TextMap(value) with Location
object CountryMap {
  def apply(value: Map[String, String]): CountryMap = new CountryMap(value)
  def empty: CountryMap = FeatureTypeDefaults.CountryMap
}

/**
 * Map of state values
 *
 * @param value map of state values
 */
class StateMap(value: Map[String, String]) extends TextMap(value) with Location
object StateMap {
  def apply(value: Map[String, String]): StateMap = new StateMap(value)
  def empty: StateMap = FeatureTypeDefaults.StateMap
}

/**
 * Map of city values
 *
 * @param value map of city values
 */
class CityMap(value: Map[String, String]) extends TextMap(value) with Location
object CityMap {
  def apply(value: Map[String, String]): CityMap = new CityMap(value)
  def empty: CityMap = FeatureTypeDefaults.CityMap
}

/**
 * Map of postal code values
 *
 * @param value map of postal code values
 */
class PostalCodeMap(value: Map[String, String]) extends TextMap(value) with Location
object PostalCodeMap {
  def apply(value: Map[String, String]): PostalCodeMap = new PostalCodeMap(value)
  def empty: PostalCodeMap = FeatureTypeDefaults.PostalCodeMap
}

/**
 * Map of street values
 *
 * @param value map of street values
 */
class StreetMap(value: Map[String, String]) extends TextMap(value) with Location
object StreetMap {
  def apply(value: Map[String, String]): StreetMap = new StreetMap(value)
  def empty: StreetMap = FeatureTypeDefaults.StreetMap
}

/**
 * Map of geolocation values
 *
 * @param value map of geolocation values
 */
class GeolocationMap(val value: Map[String, Seq[Double]]) extends OPMap[Seq[Double]] with Location
object GeolocationMap {
  def apply(value: Map[String, Seq[Double]]): GeolocationMap = new GeolocationMap(value)
  def empty: GeolocationMap = FeatureTypeDefaults.GeolocationMap
}

/**
 * Prediction representation - a map containing prediction, and optional raw prediction and probability values.
 *
 * This value can only be constructed from a non empty map containing a prediction,
 * and optional raw prediction and probability values, otherwise [[NonNullableEmptyException]] is thrown.
 *
 * @param value map containing prediction, and optional raw prediction and probability values.
 */
class Prediction private[op](value: Map[String, Double]) extends RealMap(value) with NonNullable {
  import Prediction.Keys._

  if (value == null || value.isEmpty) {
    throw new NonNullableEmptyException(classOf[Prediction])
  }
  if (!value.contains(PredictionName)) {
    throw new NonNullableEmptyException(classOf[Prediction],
      msg = Some(s"value map must contain '$PredictionName' key")
    )
  }
  require(
    value.keys.forall(k =>
      k == PredictionName || k.startsWith(RawPredictionName) || k.startsWith(ProbabilityName)
    ),
    s"value map must only contain valid keys: '$PredictionName' or " +
      s"starting with '$RawPredictionName' or '$ProbabilityName'"
  )

  // Need to make sure we sort the keys by their final index, which comes after an underscore in the apply function
  private def keysStartsWith(name: String): Array[String] = value.keys.filter(_.startsWith(name)).toArray
    .sortBy(_.split('_').last.toInt)

  /**
   * Prediction value
   */
  def prediction: Double = value(PredictionName)

  /**
   * Raw prediction values
   */
  def rawPrediction: Array[Double] = keysStartsWith(RawPredictionName).map(value)

  /**
   * Probability values
   */
  def probability: Array[Double] = keysStartsWith(ProbabilityName).map(value)

  /**
   * Score values (based of probability or prediction)
   *
   * @return prediction values or prediction
   */
  def score: Array[Double] = {
    val probKeys = keysStartsWith(ProbabilityName)
    if (probKeys.nonEmpty) probKeys.map(value) else Array(value(PredictionName))
  }

  override def toString: String = {
    val rawPred = rawPrediction.mkString("Array(", ", ", ")")
    val prob = probability.mkString("Array(", ", ", ")")
    s"${getClass.getSimpleName}(prediction = $prediction, rawPrediction = $rawPred, probability = $prob)"
  }

}
object Prediction {
  object Keys {
    val PredictionName = "prediction"
    val RawPredictionName = "rawPrediction"
    val ProbabilityName = "probability"
  }
  import Keys._

  /**
   * Creates [[Prediction]] given a prediction value
   *
   * @param prediction prediction value
   * @return [[Prediction]]
   */
  def apply(prediction: Double): Prediction = new Prediction(Map(PredictionName -> prediction))

  /**
   * Creates [[Prediction]] given a prediction value, raw prediction and probability values
   *
   * @param prediction    prediction value
   * @param rawPrediction raw prediction values
   * @return [[Prediction]]
   */
  def apply(prediction: Double, rawPrediction: Vector): Prediction = {
    val rawPred = rawPrediction.toArray.zipWithIndex.map { case (v, i) => s"${RawPredictionName}_$i" -> v }
    val pred = PredictionName -> prediction
    new Prediction(rawPred.toMap + pred)
  }

  /**
   * Creates [[Prediction]] given a prediction value, raw prediction and probability values
   *
   * @param prediction    prediction value
   * @param rawPrediction raw prediction values
   * @return [[Prediction]]
   */
  def apply(prediction: Double, rawPrediction: Double): Prediction = {
    val rawPred = s"${RawPredictionName}_0" -> rawPrediction
    val pred = PredictionName -> prediction
    new Prediction(Map(rawPred, pred))
  }

  /**
   * Creates [[Prediction]] given a prediction value, raw prediction and probability values
   *
   * @param prediction    prediction value
   * @param rawPrediction raw prediction values
   * @param probability   probability values value
   * @return [[Prediction]]
   */
  def apply(prediction: Double, rawPrediction: Vector, probability: Vector): Prediction =
    apply(prediction, rawPrediction = rawPrediction.toArray, probability = probability.toArray)

  /**
   * Creates [[Prediction]] given a prediction value, raw prediction and probability values
   *
   * @param prediction    prediction value
   * @param rawPrediction raw prediction values
   * @param probability   probability values value
   * @return [[Prediction]]
   */
  def apply(prediction: Double, rawPrediction: Array[Double], probability: Array[Double]): Prediction = {
    val rawPred = rawPrediction.zipWithIndex.map { case (v, i) => s"${RawPredictionName}_$i" -> v }
    val prob = probability.zipWithIndex.map { case (v, i) => s"${ProbabilityName}_$i" -> v }
    val pred = PredictionName -> prediction
    new Prediction((rawPred ++ prob).toMap + pred)
  }
}
