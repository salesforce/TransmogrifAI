/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.types._
import com.salesforce.op.test.{PassengerSparkFixtureTest, TestFeatureBuilder}
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichStructType._
import com.salesforce.op.utils.text.TextUtils
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TextTransmogrifyTest extends FlatSpec with PassengerSparkFixtureTest {

  val cityData: Seq[City] = RandomText.cities.take(1000).toList
  val countryData: Seq[Country] = RandomText.countries.take(1000).toList
  val postalData: Seq[PostalCode] = RandomText.postalCodes.take(1000).toList
  val textData: Seq[Text] = RandomText.textAreas(0, 1000).take(1000).toList

  val data: Seq[(City, Country, PostalCode, Text)] =
    cityData.zip(countryData).zip(postalData).zip(textData)
      .map{ case (((ci, co), p), t) => (ci, co, p, t) }

  val (ds, city, country, postal, text) = TestFeatureBuilder("city", "country", "postal", "text", data)

  "TextVectorizers" should "vectorize various text feature types" in {
    val feature = Seq(city, country, postal, text).transmogrify()
    val vectorized = new OpWorkflow().setResultFeatures(feature).transform(ds)
    val vectCollect = vectorized.collect(feature)

    for {vector <- vectCollect} {
      vector.v.size < TransmogrifierDefaults.DefaultNumOfFeatures + (TransmogrifierDefaults.TopK + 2) * 3 shouldBe true
      vector.v.size >= TransmogrifierDefaults.DefaultNumOfFeatures + 6 shouldBe true
    }

    val meta = vectorized.schema.toOpVectorMetadata(feature.name)
    meta.columns.length shouldBe vectCollect.head.value.size

    val history = meta.getColumnHistory()
    val cityColumns = history.filter(h =>
      h.parentFeatureOrigins.contains(city.name) &&
        !h.indicatorValue.contains(TransmogrifierDefaults.NullString) &&
        !h.indicatorValue.contains(TransmogrifierDefaults.OtherString)
    )
    cityColumns.forall(c => c.parentFeatureName.head == c.indicatorGroup.get) shouldBe true

    val allCities = cityData.map(v => v.value.map(TextUtils.cleanString(_)))
    cityColumns.forall(c => allCities.contains(c.indicatorValue)) shouldBe true
  }

  "Transmogrify" should "work on phone features" in {
    val phoneData = RandomText.phones.limit(1000)
    val (ds, phone) = TestFeatureBuilder("phone", phoneData)
    val feature = Seq(phone).transmogrify()
    val feature2 = phone.vectorize("US")
    val vectorized = new OpWorkflow().setResultFeatures(feature, feature2).transform(ds)
    val vectCollect = vectorized.collect(feature, feature2)

    for {(vector1, vector2) <- vectCollect} {
      vector1.v.size shouldBe 2
      vector1.v.toArray should contain theSameElementsAs vector2.v.toArray
    }
    val meta = vectorized.schema.toOpVectorMetadata(feature.name)
    meta.columns.length shouldBe vectCollect.head._2.value.size
  }

}
