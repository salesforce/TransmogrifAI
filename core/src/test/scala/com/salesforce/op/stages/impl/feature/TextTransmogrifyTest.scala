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

  val cityData: Seq[City] = RandomText.cities.take(10).toList
  val countryData: Seq[Country] = RandomText.countries.take(10).toList
  val postalData: Seq[PostalCode] = RandomText.postalCodes.take(10).toList
  val textAreaData: Seq[TextArea] = RandomText.textAreas(0, 10).take(10).toList
  val textData: Seq[Text] = RandomText.strings(0, 10).take(10).toList

  val data: Seq[(City, Country, PostalCode, Text, TextArea)] =
    cityData.zip(countryData).zip(postalData).zip(textData).zip(textAreaData)
      .map{ case ((((ci, co), p), t), ta) => (ci, co, p, t, ta) }

  val (ds, city, country, postal, text, textarea) = TestFeatureBuilder("city", "country", "postal", "text",
    "textarea", data)

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

  "Text and TextArea features" should "be vectorized with or without null tracking, as specified" in {
    val feature = Seq(text).transmogrify()
    val vectorized = new OpWorkflow().setResultFeatures(feature).transform(ds)
    val vectCollect = vectorized.collect(feature)

    vectCollect.forall(_.value.size == TransmogrifierDefaults.DefaultNumOfFeatures + 1)
  }

}
