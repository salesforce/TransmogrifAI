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

package com.salesforce.op.stages.impl.feature

import java.util.{Date => JDate}

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.types._
import com.salesforce.op.features.{Feature, FeatureLike}
import com.salesforce.op.stages.base.ternary.TernaryLambdaTransformer
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorMetadata, _}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

import scala.collection.Traversable
import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class OPMapVectorizerTest extends FlatSpec with TestSparkContext {

  import OPMapVectorizerTestHelper._

  val log = LoggerFactory.getLogger(this.getClass)
  // loggingLevel(Level.INFO)

  Spec[TextMapPivotVectorizer[_]] should "return the expected vector with the default param settings" in {
    val (dataSet, top, bot) = TestFeatureBuilder("top", "bot",
      Seq(
        (Map("a" -> "d", "b" -> "d"), Map("x" -> "W")),
        (Map("a" -> "e"), Map("z" -> "w", "y" -> "v")),
        (Map("c" -> "D"), Map("x" -> "w", "y" -> "V")),
        (Map("c" -> "d", "a" -> "d"), Map("z" -> "v"))
      ).map(v => v._1.toTextMap -> v._2.toTextMap)
    )
    val vectorizer =
      new TextMapPivotVectorizer[TextMap]()
        .setCleanKeys(true).setMinSupport(0).setTopK(10).setTrackNulls(false).setInput(top, bot)

    val fitted = vectorizer.fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(vectorizer.getOutputFeatureName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)

    transformed.collect(vectorizer.getOutput()) shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  "Binary features" should "be vectorized the same whether they're in maps or not" in {
    val binaryData: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)
    val binaryData2: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)
    val binaryData3: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Binary, BinaryMap, Boolean](binaryData, binaryData2, binaryData3)
  }

  "Base64 features" should "be vectorized the same whether they're in maps or not" in {
    val base64Data: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.5).limit(1000)
    val base64Data2: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.5).limit(1000)
    val base64Data3: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Base64, Base64Map, String](base64Data, base64Data2, base64Data3)
  }

  "ComboBox features" should "be vectorized the same whether they're in maps or not" in {
    val comboBoxData: Seq[ComboBox] = RandomText.comboBoxes(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0).limit(1000)
    val comboBoxData2: Seq[ComboBox] = RandomText.comboBoxes(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val comboBoxData3: Seq[ComboBox] = RandomText.comboBoxes(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[ComboBox, ComboBoxMap, String](comboBoxData, comboBoxData2, comboBoxData3)
  }

  "Currency features" should "be vectorized the same whether they're in maps or not" in {
    val currencyData: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val currencyData2: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val currencyData3: Seq[Currency] = RandomReal.logNormal[Currency](mean = 10.0, sigma = 1.0)
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Currency, CurrencyMap, Double](currencyData, currencyData2, currencyData3)
  }

  "Date features" should "be vectorized the same whether they're in maps or not" in {
    val minSec = 1000000
    val maxSec = 1000000000
    val dateData: Seq[Date] = RandomIntegral.dates(init = new JDate(1500000000000L),
      minStep = minSec, maxStep = maxSec).withProbabilityOfEmpty(0.5).limit(1000)
    val dateData2: Seq[Date] = RandomIntegral.dates(init = new JDate(1500000000000L),
      minStep = minSec, maxStep = maxSec).withProbabilityOfEmpty(0.5).limit(1000)
    val dateData3: Seq[Date] = RandomIntegral.dates(init = new JDate(1500000000000L),
      minStep = minSec, maxStep = maxSec).withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Date, DateMap, Long](dateData, dateData2, dateData3)
  }

  "DateTime features" should "be vectorized the same whether they're in maps or not" in {
    val minSec = 1000000
    val maxSec = 1000000000
    val dateTimeData: Seq[DateTime] = RandomIntegral.datetimes(init = new JDate(), minStep = minSec, maxStep = maxSec)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val dateTimeData2: Seq[DateTime] = RandomIntegral.datetimes(init = new JDate(), minStep = minSec, maxStep = maxSec)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val dateTimeData3: Seq[DateTime] = RandomIntegral.datetimes(init = new JDate(), minStep = minSec, maxStep = maxSec)
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[DateTime, DateTimeMap, Long](dateTimeData, dateTimeData2, dateTimeData3)
  }

  "Email features" should "be vectorized the same whether they're in maps or not" in {
    val emailData: Seq[Email] = RandomText.emailsOn(RandomStream of List("gmail.com", "yahoo.com", "yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val emailData2: Seq[Email] = RandomText.emailsOn(RandomStream of List("gmail.com", "yahoo.com", "yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val emailData3: Seq[Email] = RandomText.emailsOn(RandomStream of List("gmail.com", "yahoo.com", "yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Email, EmailMap, String](emailData, emailData2, emailData3)
  }

  "ID features" should "be vectorized the same whether they're in maps or not" in {
    val idData: Seq[ID] = RandomText.uniqueIds.withProbabilityOfEmpty(0.5).limit(1000)
    val idData2: Seq[ID] = RandomText.uniqueIds.withProbabilityOfEmpty(0.5).limit(1000)
    val idData3: Seq[ID] = RandomText.uniqueIds.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[ID, IDMap, String](idData, idData2, idData3)
  }

  "Integral features" should "be vectorized the same whether they're in maps or not" in {
    val integralData: Seq[Integral] = RandomIntegral.integrals(from = -100L, to = 100L)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val integralData2: Seq[Integral] = RandomIntegral.integrals(from = -100L, to = 100L)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val integralData3: Seq[Integral] = RandomIntegral.integrals(from = -100L, to = 100L)
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Integral, IntegralMap, Long](integralData, integralData2, integralData3)
  }

  "MultiPickList features" should "be vectorized the same whether they're in maps or not" in {
    val mplData: Seq[MultiPickList] = RandomMultiPickList.of(texts =
      RandomText.pickLists(domain = List("A", "B", "C", "D", "E")), maxLen = 3).limit(1000)
    val mplData2: Seq[MultiPickList] = RandomMultiPickList.of(texts =
      RandomText.pickLists(domain = List("A", "B", "C", "D", "E")), maxLen = 3).limit(1000)
    val mplData3: Seq[MultiPickList] = RandomMultiPickList.of(texts =
      RandomText.pickLists(domain = List("A", "B", "C", "D", "E")), maxLen = 3).limit(1000)

    testFeatureToMap[MultiPickList, MultiPickListMap, Set[String]](mplData, mplData2, mplData3)
  }

  "Percent features" should "be vectorized the same whether they're in maps or not" in {
    val percentData: Seq[Percent] = RandomReal.uniform[Percent]().withProbabilityOfEmpty(0.5).limit(1000)
    val percentData2: Seq[Percent] = RandomReal.uniform[Percent]().withProbabilityOfEmpty(0.5).limit(1000)
    val percentData3: Seq[Percent] = RandomReal.uniform[Percent]().withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Percent, PercentMap, Double](percentData, percentData2, percentData3)
  }

  // TODO: Fix failing test
  "Phone features" should "be vectorized the same whether they're in maps or not" in {
    val phoneData: Seq[Phone] = RandomText.phonesWithErrors(probabilityOfError = 0.3)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val phoneData2: Seq[Phone] = RandomText.phonesWithErrors(probabilityOfError = 0.3)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val phoneData3: Seq[Phone] = RandomText.phonesWithErrors(probabilityOfError = 0.3)
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Phone, PhoneMap, String](phoneData, phoneData2, phoneData3)
  }

  "Picklist features" should "be vectorized the same whether they're in maps or not" in {
    val pickListData: Seq[PickList] = RandomText.pickLists(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val pickListData2: Seq[PickList] = RandomText.pickLists(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val pickListData3: Seq[PickList] = RandomText.pickLists(domain = List("A", "B", "C", "D", "E"))
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[PickList, PickListMap, String](pickListData, pickListData2, pickListData3)
  }

  "Real features" should "be vectorized the same whether they're in maps or not" in {
    val realData: Seq[Real] = RandomReal.uniform[Real]().withProbabilityOfEmpty(0.5).limit(1000)
    val realData2: Seq[Real] = RandomReal.uniform[Real]().withProbabilityOfEmpty(0.5).limit(1000)
    val realData3: Seq[Real] = RandomReal.uniform[Real]().withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Real, RealMap, Double](realData, realData2, realData3)
  }

  "TextArea features" should "be vectorized the same whether they're in maps or not" in {
    val textAreaData: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val textAreaData2: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(1000)
    val textAreaData3: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[TextArea, TextAreaMap, String](textAreaData, textAreaData2, textAreaData3)
  }

  "Text features" should "be vectorized the same whether they're in maps or not" in {
    val textData: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(1000)
    val textData2: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(1000)
    val textData3: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Text, TextMap, String](textData, textData2, textData3)
  }

  "URL features" should "be vectorized the same whether they're in maps or not" in {
    // Generate URLs from a list of domains where some are valid and some are not
    val urlData: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp", "ish?"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val urlData2: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp", "ish?"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val urlData3: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp", "ish?"))
      .withProbabilityOfEmpty(0.5).limit(1000)

    // The problem with doing completely random URLs is that most randomly generated URLs will be invalid and
    // therefore removed from the resulting PickListMap. If any of them map keys are entirely full of invalid URLs
    // then those map keys will not be present at all in the PickListMap, and therefore the vectorized map data
    // will have less columns than the vectorized non-map data.
    /*
    val urlData: Seq[URL] = RandomText.urls.withProbabilityOfEmpty(0.5).limit(1000)
    val urlData2: Seq[URL] = RandomText.urls.withProbabilityOfEmpty(0.5).limit(1000)
    val urlData3: Seq[URL] = RandomText.urls.withProbabilityOfEmpty(0.5).limit(1000)
     */

    testFeatureToMap[URL, URLMap, String](urlData, urlData2, urlData3)
  }

  "Country features" should "be vectorized the same whether they're in maps or not" in {
    val countryData: Seq[Country] = RandomText.countries.withProbabilityOfEmpty(0.5).limit(1000)
    val countryData2: Seq[Country] = RandomText.countries.withProbabilityOfEmpty(0.5).limit(1000)
    val countryData3: Seq[Country] = RandomText.countries.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Country, CountryMap, String](countryData, countryData2, countryData3)
  }

  "State features" should "be vectorized the same whether they're in maps or not" in {
    val stateData: Seq[State] = RandomText.states.withProbabilityOfEmpty(0.5).limit(1000)
    val stateData2: Seq[State] = RandomText.states.withProbabilityOfEmpty(0.5).limit(1000)
    val stateData3: Seq[State] = RandomText.states.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[State, StateMap, String](stateData, stateData2, stateData3)
  }

  "City features" should "be vectorized the same whether they're in maps or not" in {
    val cityData: Seq[City] = RandomText.cities.withProbabilityOfEmpty(0.5).limit(1000)
    val cityData2: Seq[City] = RandomText.cities.withProbabilityOfEmpty(0.5).limit(1000)
    val cityData3: Seq[City] = RandomText.cities.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[City, CityMap, String](cityData, cityData2, cityData3)
  }

  "PostalCode features" should "be vectorized the same whether they're in maps or not" in {
    val postalCodeData: Seq[PostalCode] = RandomText.postalCodes.withProbabilityOfEmpty(0.5).limit(1000)
    val postalCodeData2: Seq[PostalCode] = RandomText.postalCodes.withProbabilityOfEmpty(0.5).limit(1000)
    val postalCodeData3: Seq[PostalCode] = RandomText.postalCodes.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[PostalCode, PostalCodeMap, String](postalCodeData, postalCodeData2, postalCodeData3)
  }

  "Street features" should "be vectorized the same whether they're in maps or not" in {
    val streetData: Seq[Street] = RandomText.streets.withProbabilityOfEmpty(0.5).limit(1000)
    val streetData2: Seq[Street] = RandomText.streets.withProbabilityOfEmpty(0.5).limit(1000)
    val streetData3: Seq[Street] = RandomText.streets.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Street, StreetMap, String](streetData, streetData2, streetData3)
  }

  "Geolocation features" should "be vectorized the same whether they're in maps or not" in {
    val GeolocationData: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)
    val GeolocationData2: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)
    val GeolocationData3: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)

    testFeatureToMap[Geolocation, GeolocationMap, Seq[Double]](GeolocationData, GeolocationData2, GeolocationData3)
  }
}

object OPMapVectorizerTestHelper extends Matchers {

  val log = LoggerFactory.getLogger(this.getClass)

  /**
   * Constructs a single OPMap feature from three input features of the corresponding type, where each input feature
   * corresponds to its own key in the OPMap feature. This is used to test whether base feature types are vectorized
   * the same as their corresponding map types.
   *
   * @param f1Data Sequence of base feature type data (eg. from generators)
   * @param f2Data Sequence of base feature type data (eg. from generators)
   * @param f3Data Sequence of base feature type data (eg. from generators)
   * @tparam F  Base feature type (eg. ID, Text, Integer)
   * @tparam FM OPMap feature type (eg. IDMap, TextMap, IntegerMap)
   * @tparam MT Value type of map inside OPMap feature (eg. String, String, Int)
   */
  def testFeatureToMap[F <: FeatureType : TypeTag, FM <: OPMap[MT] : TypeTag, MT: TypeTag]
  (f1Data: Seq[F], f2Data: Seq[F], f3Data: Seq[F])(implicit spark: SparkSession): Unit = {

    val generatedData: Seq[(F, F, F)] = f1Data.zip(f2Data).zip(f3Data).map { case ((f1, f2), f3) => (f1, f2, f3) }
    val (rawDF, rawF1, rawF2, rawF3) = TestFeatureBuilder("f1", "f2", "f3", generatedData)

    // Turn off prepending the feature name in the tests here so that we can compare the hashes of the same data
    // in maps and base features. Otherwise because the feature type is different, they would have different feature
    // name hashes appended to all the tokens and never be equal.
    implicit val TransmogrifierTestDefaults = new TransmogrifierDefaults {
      // Also need to set ReferenceDate to prevent it from being locally different that the one used everywhere else
      override val ReferenceDate = TransmogrifierDefaults.ReferenceDate
      override val PrependFeatureName: Boolean = false
    }

    val featureVector = Transmogrifier.transmogrify(Seq(rawF1, rawF2, rawF3))(TransmogrifierTestDefaults).combine()
    val transformed = new OpWorkflow().setResultFeatures(featureVector).transform(rawDF)
    if (log.isInfoEnabled) {
      log.info("transformed data:")
      transformed.show(10)
    }

    val summary = transformed.schema(featureVector.name).metadata
    log.info("summary:\n{}", summary)
    log.info(s"summary.getMetadataArray('${OpVectorMetadata.ColumnsKey}'):\n{}",
      summary.getMetadataArray(OpVectorMetadata.ColumnsKey).toList
    )
    // Transformer to construct a single map feature from the individual features
    val mapFeature = makeTernaryOPMapTransformer[F, FM, MT](rawF1, rawF2, rawF3)
    val mapFeatureVector = Transmogrifier.transmogrify(Seq(mapFeature))(TransmogrifierTestDefaults).combine()
    val transformedMap = new OpWorkflow().setResultFeatures(mapFeatureVector).transform(rawDF)
    val mapSummary = transformedMap.schema(mapFeatureVector.name).metadata
    if (log.isInfoEnabled) {
      log.info("transformedMap:")
      transformedMap.show(10)
    }

    // Check that the actual features are the same
    val vectorizedBaseFeatures = transformed.collect(featureVector)
    val vectorizedMapFeatures = transformedMap.collect(mapFeatureVector)
    log.info("vectorizedBaseFeatures: {}", vectorizedBaseFeatures.toList)
    log.info("vectorizedMapFeatures: {}", vectorizedMapFeatures.toList)

    val baseColMetaArray =
      summary.getMetadataArray(OpVectorMetadata.ColumnsKey).flatMap(OpVectorColumnMetadata.fromMetadata)
    val mapColMetaArray =
      mapSummary.getMetadataArray(OpVectorMetadata.ColumnsKey).flatMap(OpVectorColumnMetadata.fromMetadata)
    log.info("baseColMetaArray: {}", baseColMetaArray.map(_.toString).mkString("\n"))
    log.info("mapColMetaArray: {}", mapColMetaArray.map(_.toString).mkString("\n"))

    // val baseIndicesToCompare: Array[Int] = baseColMetaArray.filterNot(_.isNullIndicator).map(_.index).sorted
    val baseIndicesToCompare: Array[Int] = baseColMetaArray
      .map(f => (f.parentFeatureName.head, f.indicatorValue, f.indicatorGroup) match {
        case (pfName, Some(iv), Some(ig)) => (pfName + ig + iv, f.index)
        case (pfName, Some(iv), None) => (pfName + iv, f.index)
        case (pfName, None, Some(ig)) => (pfName + ig, f.index)
        case (pfName, None, None) => (pfName, f.index)
      }).sortBy(_._1).map(_._2)
    // Also need to sort map vectorized indices by feature name since they can come out in arbitrary orders
    val mapIndicesToCompare: Array[Int] = mapColMetaArray
      .map(f => (f.parentFeatureName.head, f.indicatorValue, f.indicatorGroup) match {
        case (pfName, Some(iv), Some(ig)) => (pfName + ig + iv, f.index)
        case (pfName, Some(iv), None) => (pfName + iv, f.index)
        case (pfName, None, Some(ig)) => (pfName + ig, f.index)
        case (pfName, None, None) => (pfName, f.index)
      }).sortBy(_._1).map(_._2)
    log.info("base indices to compare: {}", baseIndicesToCompare.toList)
    log.info("map indices to compare: {}", mapIndicesToCompare.toList)

    vectorizedBaseFeatures.zip(vectorizedMapFeatures).zipWithIndex.foreach {
      case ((baseFeat, mapFeat), i) =>
        log.info("baseFeat: {}", baseFeat)
        log.info("baseIndicesToCompare.map(baseFeat.value.apply): {}", baseIndicesToCompare.map(baseFeat.value.apply))
        log.info("mapFeat: {}", mapFeat)
        log.info("mapIndicesToCompare.map(mapFeat.value.apply): {}", mapIndicesToCompare.map(mapFeat.value.apply))

        val isTheSame =
          baseIndicesToCompare.map(baseFeat.value.apply) sameElements mapIndicesToCompare.map(mapFeat.value.apply)

        if (!isTheSame) {
          log.error(
            s"Mismatch when comparing ${i}th row!\n" +
              s"transformed.take(${i + 1}).drop($i):\n" +
              transformed.take(i + 1).drop(i).head.toSeq.mkString("   ", "\n   ", "") +
              s"\ntransformedMap.take(${i + 1}).drop($i):\n" +
              transformedMap.take(i + 1).drop(i).head.toSeq.mkString("   ", "\n   ", "")
          )
        }
        isTheSame shouldBe true
    }

    // TODO assert metadata
  }

  /**
   * Construct OPMap transformer for raw features
   */
  def makeTernaryOPMapTransformer[F <: FeatureType : TypeTag, FM <: OPMap[MT] : TypeTag, MT: TypeTag]
  (
    rawF1: FeatureLike[F],
    rawF2: FeatureLike[F],
    rawF3: FeatureLike[F]
  ): Feature[FM] = {
    val ftFactory = FeatureTypeFactory[FM]()

    // Transformer to construct a single map feature from the individual features
    val mapTransformer = new TernaryLambdaTransformer[F, F, F, FM](operationName = "mapify",
      transformFn = (f1, f2, f3) => {
        // For all the maps, the value in the original feature type is an Option[MT], but can't figure out how
        // to specify that here since that's not true in general for FeatureTypes (eg. RealNN)
        def asMap(v: F, featureName: String): Map[String, Any] = {
          v.value match {
            case Some(s) => Map(featureName -> s)
            case t: Traversable[_] if t.nonEmpty => Map(featureName -> t)
            case _ => Map.empty
          }
        }
        val mapData = asMap(f1, "f1") ++ asMap(f2, "f2") ++ asMap(f3, "f3")
        ftFactory.newInstance(mapData)
      }
    )
    mapTransformer.setInput(rawF1, rawF2, rawF3).getOutput().asInstanceOf[Feature[FM]]
  }

}


