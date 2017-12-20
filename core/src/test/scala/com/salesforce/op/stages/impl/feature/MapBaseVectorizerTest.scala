/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import java.util.{Date => JDate}

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.ternary.TernaryLambdaTransformer
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.testkit._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.{OpVectorMetadata, _}
import org.apache.log4j.Level
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class MapBaseVectorizerTest extends FlatSpec with TestSparkContext {

  val log = LoggerFactory.getLogger(this.getClass)
  loggingLevel(Level.ERROR)

  lazy val (dataSet, top, bot) = TestFeatureBuilder("top", "bot",
    Seq(
      (Map("a" -> "d", "b" -> "d"), Map("x" -> "W")),
      (Map("a" -> "e"), Map("z" -> "w", "y" -> "v")),
      (Map("c" -> "D"), Map("x" -> "w", "y" -> "V")),
      (Map("c" -> "d", "a" -> "d"), Map("z" -> "v"))
    ).map(v => v._1.toTextMap -> v._2.toTextMap)
  )
  val vectorizer = new TextMapPivotVectorizer[TextMap]().setCleanKeys(true).setMinSupport(0).setTopK(10)
    .setInput(top, bot)
  val vector = vectorizer.getOutput()

  Spec[TextMapPivotVectorizer[_]] should "return the expected vector with the default param settings" in {
    val fitted = vectorizer.fit(dataSet)
    val transformed = fitted.transform(dataSet)
    val vectorMetadata = fitted.getMetadata()
    log.info(OpVectorMetadata(vectorizer.outputName, vectorMetadata).toString)
    val expected = Array(
      Vectors.sparse(14, Array(2, 5, 7), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(3, 9, 12), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 7, 9), Array(1.0, 1.0, 1.0)),
      Vectors.sparse(14, Array(0, 2, 11), Array(1.0, 1.0, 1.0))
    ).map(_.toOPVector)

    transformed.collect(vector) shouldBe expected
    fitted.getMetadata() shouldBe transformed.schema.fields(2).metadata
  }

  "Base64 features" should "be vectorized the same way whether their in maps, or their own base features" in {
    val base64Data: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.3).limit(1000)
    val base64Data2: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.3).limit(1000)
    val base64Data3: Seq[Base64] = RandomText.base64(10, 50).withProbabilityOfEmpty(0.3).limit(1000)

    val generatedData: Seq[(Base64, Base64, Base64)] =
      base64Data.zip(base64Data2).zip(base64Data3).map {
        case ((f1, f2), f3) => (f1, f2, f3)
      }

    val (rawDF, rawBase64F1, rawBase64F2, rawBase64F3) = TestFeatureBuilder("base64_feature1", "base64_feature2",
      "base64_feature3", generatedData)

    val featureVector = Seq(rawBase64F1, rawBase64F2, rawBase64F3).transmogrify()
    val transformed = new OpWorkflow().setResultFeatures(featureVector).transform(rawDF)
    val summary = transformed.schema(featureVector.name).metadata

    // Transformer to construct a single map feature from the individual features
    val mapTransformer = new TernaryLambdaTransformer[Base64, Base64, Base64, Base64Map](operationName = "labelFunc",
      transformFn = (f1, f2, f3) => {
        val mapData: Map[String, String] = {
          val m1 = f1.v match {
            case Some(s) => Map[String, String]("f1" -> s)
            case None => Map.empty[String, String]
          }
          val m2 = f2.v match {
            case Some(s) => Map[String, String]("f2" -> s)
            case None => Map.empty[String, String]
          }
          val m3 = f3.v match {
            case Some(s) => Map[String, String]("f3" -> s)
            case None => Map.empty[String, String]
          }
          m1 ++ m2 ++ m3
        }
        Base64Map(mapData)
      }
    )
    val base64MapFeature = mapTransformer.setInput(rawBase64F1, rawBase64F2, rawBase64F3).getOutput()
      .asInstanceOf[Feature[Base64Map]]
    val mapFeatureVector = Seq(base64MapFeature).transmogrify()
    val transformedMap = new OpWorkflow().setResultFeatures(mapFeatureVector).transform(rawDF)
    val mapSummary = transformedMap.schema(mapFeatureVector.name).metadata
  }

  "ID features" should "be vectorized the same way whether their in maps, or their own base features" in {
    val IDData: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.3).limit(1000)
    val IDData2: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.3).limit(1000)
    val IDData3: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.3).limit(1000)

    val generatedData: Seq[(ID, ID, ID)] =
      IDData.zip(IDData2).zip(IDData3).map {
        case ((f1, f2), f3) => (f1, f2, f3)
      }

    val (rawDF, rawIDF1, rawIDF2, rawIDF3) = TestFeatureBuilder("ID_feature1", "ID_feature2",
      "ID_feature3", generatedData)

    val featureVector = Seq(rawIDF1, rawIDF2, rawIDF3).transmogrify()
    val transformed = new OpWorkflow().setResultFeatures(featureVector).transform(rawDF)
    val summary = transformed.schema(featureVector.name).metadata
    log.info(s"summary: $summary")
    log.info(s"summary.getMetadataArray('vector_columns'): ${summary.getMetadataArray("vector_columns").toList}")

    // Transformer to construct a single map feature from the individual features
    val mapTransformer = new TernaryLambdaTransformer[ID, ID, ID, IDMap](operationName = "labelFunc",
      transformFn = (f1, f2, f3) => {
        val mapData: Map[String, String] = {
          val m1 = f1.v.map(v => "f1" -> v).toMap
          val m2 = f2.v.map(v => "f2" -> v).toMap
          val m3 = f3.v.map(v => "f3" -> v).toMap
          m1 ++ m2 ++ m3
        }
        IDMap(mapData)
      }
    )
    val IDMapFeature = mapTransformer.setInput(rawIDF1, rawIDF2, rawIDF3).getOutput()
      .asInstanceOf[Feature[IDMap]]
    val mapFeatureVector = Seq(IDMapFeature).transmogrify()
    val transformedMap = new OpWorkflow().setResultFeatures(mapFeatureVector).transform(rawDF)
    val mapSummary = transformedMap.schema(mapFeatureVector.name).metadata

    // Main difference for correctly vectorized features is that nulls are not explicitly tracked in maps, while
    // they are in the underlying feature types. So, need to extract out the feature vector except the nullIndicator
    // columns

  }

  "private function" should "do the same thing" in {
    val iDData: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.5).limit(1000)
    val iDData2: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.5).limit(1000)
    val iDData3: Seq[ID] = RandomText.ids.withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[ID, IDMap, String](iDData, iDData2, iDData3)
  }

  "Binary features" should "be vectorized the same whether they're in maps or not" in {
    val binaryData: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)
    val binaryData2: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)
    val binaryData3: Seq[Binary] = RandomBinary(0.5).withProbabilityOfEmpty(0.5).limit(1000)

    testFeatureToMap[Binary, BinaryMap, Boolean](binaryData, binaryData2, binaryData3)
  }

  // TODO: Fix failing test
  "Base64 features" should "be vectorized the same whether they're in maps or not" ignore {
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

  // TODO: Fix failing test
  "MultiPickList features" should "be vectorized the same whether they're in maps or not" ignore {
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
  "Phone features" should "be vectorized the same whether they're in maps or not" ignore {
    val phoneData: Seq[Phone] = RandomText.phones.withProbabilityOfEmpty(0.5).limit(1000)
    val phoneData2: Seq[Phone] = RandomText.phones.withProbabilityOfEmpty(0.5).limit(1000)
    val phoneData3: Seq[Phone] = RandomText.phones.withProbabilityOfEmpty(0.5).limit(1000)

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

  // TODO: Fix failing test
  "TextArea features" should "be vectorized the same whether they're in maps or not" ignore {
    val textAreaData: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(100)
    val textAreaData2: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(100)
    val textAreaData3: Seq[TextArea] = RandomText.textAreas(minLen = 5, maxLen = 10)
      .withProbabilityOfEmpty(0.5).limit(100)

    testFeatureToMap[TextArea, TextAreaMap, String](textAreaData, textAreaData2, textAreaData3)
  }

  // TODO: Fix failing test
  "Text features" should "be vectorized the same whether they're in maps or not" ignore {
    val textData: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(100)
    val textData2: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(100)
    val textData3: Seq[Text] = RandomText.strings(minLen = 5, maxLen = 10).withProbabilityOfEmpty(0.5).limit(100)

    testFeatureToMap[Text, TextMap, String](textData, textData2, textData3)
  }

  // TODO: Fix failing test
  "URL features" should "be vectorized the same whether they're in maps or not" in {
    val urlData: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val urlData2: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)
    val urlData3: Seq[URL] = RandomText.urlsOn(RandomStream of List("www.google.com", "www.yahoo.co.jp"))
      .withProbabilityOfEmpty(0.5).limit(1000)

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

  // TODO: Fix failing test
  "Geolocation features" should "be vectorized the same whether they're in maps or not" ignore {
    val GeolocationData: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)
    val GeolocationData2: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)
    val GeolocationData3: Seq[Geolocation] = RandomList.ofGeolocations.limit(1000)

    testFeatureToMap[Geolocation, GeolocationMap, Seq[Double]](GeolocationData, GeolocationData2, GeolocationData3)
  }

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
  private def testFeatureToMap[F <: FeatureType : TypeTag, FM <: OPMap[MT] : TypeTag, MT: TypeTag]
  (f1Data: Seq[F], f2Data: Seq[F], f3Data: Seq[F]): Unit = {

    val generatedData: Seq[(F, F, F)] = f1Data.zip(f2Data).zip(f3Data).map { case ((f1, f2), f3) => (f1, f2, f3) }

    val (rawDF, rawF1, rawF2, rawF3) = TestFeatureBuilder("f1", "f2", "f3", generatedData)

    val featureVector = Seq(rawF1, rawF2, rawF3).transmogrify()
    val transformed = new OpWorkflow().setResultFeatures(featureVector).transform(rawDF)
    if (log.isInfoEnabled) {
      log.info(s"transformed:")
      transformed.show(10)
    }

    val summary = transformed.schema(featureVector.name).metadata
    log.info(s"summary: $summary")
    log.info(s"summary.getMetadataArray('vector_columns'): ${summary.getMetadataArray("vector_columns").toList}")
    val ftFactory = FeatureTypeFactory[FM]()

    // Transformer to construct a single map feature from the individual features
    val mapTransformer = new TernaryLambdaTransformer[F, F, F, FM](operationName = "labelFunc",
      transformFn = (f1, f2, f3) => {
        // For all the maps, the value in the original feature type is an Option[MT], but can't figure out how
        // to specify that here since that's not true in general for FeatureTypes (eg. RealNN)
        val mapData = {
          val m1 = f1.v match {
            case Some(s) => Map("f1" -> s)
            case seq: Seq[_] if seq.nonEmpty => Map("f1" -> seq)
            case _ => Map.empty
          }
          val m2 = f2.v match {
            case Some(s) => Map("f2" -> s)
            case seq: Seq[_] if seq.nonEmpty => Map("f2" -> seq)
            case _ => Map.empty
          }
          val m3 = f3.v match {
            case Some(s) => Map("f3" -> s)
            case seq: Seq[_] if seq.nonEmpty => Map("f3" -> seq)
            case _ => Map.empty
          }
          m1 ++ m2 ++ m3
        }
        ftFactory.newInstance(mapData)
      }
    )

    val mapFeature = mapTransformer.setInput(rawF1, rawF2, rawF3).getOutput()
      .asInstanceOf[Feature[FM]]
    val mapFeatureVector = Seq(mapFeature).transmogrify()
    val transformedMap = new OpWorkflow().setResultFeatures(mapFeatureVector).transform(rawDF)
    val mapSummary = transformedMap.schema(mapFeatureVector.name).metadata
    if (log.isInfoEnabled) {
      log.info(s"transformedMap:")
      transformedMap.show(10)
    }

    // Check that the actual features are the same
    val vectorizedBaseFeatures = transformed.collect(featureVector)
    val vectorizedMapFeatures = transformedMap.collect(mapFeatureVector)
    log.info(s"vectorizedBaseFeatures: ${vectorizedBaseFeatures.toList}")
    log.info(s"vectorizedMapFeatures: ${vectorizedMapFeatures.toList}")

    val baseColMetaArray: Array[OpVectorColumnMetadata] = summary.getMetadataArray("vector_columns").flatMap(
      OpVectorColumnMetadata.fromMetadata
    )
    val mapColMetaArray: Array[OpVectorColumnMetadata] = mapSummary.getMetadataArray("vector_columns").flatMap(
      OpVectorColumnMetadata.fromMetadata
    )

    log.info(s"baseColMetaArray.foreach(println):")
    baseColMetaArray.foreach(m => log.info(m.toString))
    log.info(s"mapColMetaArray.foreach(println):")
    mapColMetaArray.foreach(m => log.info(m.toString))

    // TODO: Until null-tracking is standardized between maps and base features, only compare non-null indicator cols
    // val baseIndicesToCompare: Array[Int] = baseColMetaArray.filterNot(_.isNullIndicator).map(_.index).sorted
    val baseIndicesToCompare: Array[Int] = baseColMetaArray.filterNot(_.isNullIndicator)
      .map(f => (f.parentFeatureName.head, f.indicatorValue, f.indicatorGroup) match {
        case (pfName, Some(iv), Some(ig)) => (ig + iv, f.index)
        case (pfName, Some(iv), None) => (pfName + iv, f.index)
        case (pfName, None, Some(ig)) => (ig, f.index)
        case (pfName, None, None) => (pfName, f.index)
      }).sortBy(_._1).map(_._2)
    // Also need to sort map vectorized indices by feature name since they can come out in arbitrary orders
    val mapIndicesToCompare: Array[Int] = mapColMetaArray
      .map(f => (f.parentFeatureName.head, f.indicatorValue, f.indicatorGroup) match {
        case (pfName, Some(iv), Some(ig)) => (ig + iv, f.index)
        case (pfName, Some(iv), None) => (pfName + iv, f.index)
        case (pfName, None, Some(ig)) => (ig, f.index)
        case (pfName, None, None) => (pfName, f.index)
      }).sortBy(_._1).map(_._2)
    log.info(s"base indices to compare: ${baseIndicesToCompare.toList}")
    log.info(s"map indices to compare: ${mapIndicesToCompare.toList}")

    vectorizedBaseFeatures.zip(vectorizedMapFeatures).forall {
      case (baseFeat, mapFeat) =>
        log.info(s"baseFeat: $baseFeat")
        log.info(s"baseIndicesToCompare.map(baseFeat.value.apply): " +
          s"${baseIndicesToCompare.map(baseFeat.value.apply).toList}")
        log.info(s"mapFeat: $mapFeat")
        log.info(s"mapIndicesToCompare.map(mapFeat.value.apply): " +
          s"${mapIndicesToCompare.map(mapFeat.value.apply).toList}")
        baseIndicesToCompare.map(baseFeat.value.apply) sameElements mapIndicesToCompare.map(mapFeat.value.apply)
    } shouldBe true
  }
}
