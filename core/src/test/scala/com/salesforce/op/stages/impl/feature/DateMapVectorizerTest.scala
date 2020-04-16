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

import com.salesforce.op.{OpWorkflow, _}
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.joda.time.{DateTimeConstants, DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DateMapVectorizerTest extends FlatSpec with TestSparkContext with AttributeAsserts {

  // Sunday July 12th 1998 at 22:45
  private val defaultDate = new JDateTime(1998, 7, 12, 22, 45, DateTimeUtils.DefaultTimeZone).getMillis

  lazy val modelLocation = tempDir + "/dt-map-test-model-" + JDateTime.now().getMillis

  abstract class SampleData(val moment: JDateTime) {
    val (ds, f1) = TestFeatureBuilder(
      Seq[DateTimeMap](
        DateTimeMap(Map("a" -> 1, "b" -> defaultDate, "c" -> 3 * DateTimeConstants.MILLIS_PER_DAY)),
        DateTimeMap(Map("a" -> 1, "c" -> 0)),
        DateTimeMap(Map("b" -> 0, "c" -> moment.plusDays(100).plusMinutes(1).getMillis))
      )
    )
  }

  def checkAt(moment: JDateTime): Unit = new SampleData(moment) {
    val vector = f1.vectorize(defaultValue = 0, referenceDate = moment, trackNulls = false,
      circularDateReps = Seq())
    val transformed = new OpWorkflow().setResultFeatures(vector).transform(ds)
    val result = transformed.collect(vector)
    withClue(s"Checking transformation at $moment") {
      result shouldBe expected(moment)
    }
    val meta = OpVectorMetadata(vector.name, transformed.schema(vector.name).metadata)
    meta.columns.length shouldBe 3
    meta.columns.map(_.grouping) should contain theSameElementsAs Array(Option("a"), Option("b"), Option("c"))
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expected(moment).head.value.size)(false), result)

    val vector2 = f1.vectorize(defaultValue = 0, referenceDate = moment, trackNulls = true,
      circularDateReps = Seq())
    val transformed2 = new OpWorkflow().setResultFeatures(vector2).transform(ds)
    val result2 = transformed2.collect(vector2)
    result2.head.v.size shouldBe 6

    val meta2 = OpVectorMetadata(vector2.name, transformed2.schema(vector2.name).metadata)
    meta2.columns.length shouldBe 6
    meta2.history.keys.size shouldBe 1
    val field2 = transformed2.schema(vector2.name)
    assertNominal(field2, Array.fill(expected(moment).head.value.size)(Seq(false, true)).flatten, result2)

    val vector3 = f1.vectorize(defaultValue = 0)
    val transformed3 = new OpWorkflow().setResultFeatures(vector3).transform(ds)
    val result3 = transformed3.collect(vector3)
    result3.head.v.size shouldBe 30

    val meta3 = OpVectorMetadata(vector3.name, transformed3.schema(vector3.name).metadata)
    meta3.columns.length shouldBe 30
    meta2.history.keys.size shouldBe 1
    val field3 = transformed3.schema(vector3.name)
    val expectedNominal = Array.fill(24)(false) ++ Array.fill(3)(Seq(false, true)).flatten.asInstanceOf[Array[Boolean]]
    assertNominal(field3, expectedNominal, result3)
  }

  private def expected(moment: JDateTime) = {
    val nowMinusMilli = moment.minus(1L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val now = moment.minus(0L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val zero = 0
    val threeDaysAgo = moment.minus(3 * DateTimeConstants.MILLIS_PER_DAY).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val defaultTimeAgo = moment.minus(defaultDate).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val hundredDaysAgo = DateTimeUtils
      .getStandardDays(moment.plusDays(100).getMillis, moment.getMillis)

    Array(
      Array(nowMinusMilli, defaultTimeAgo, threeDaysAgo),
      Array(nowMinusMilli, zero, now),
      Array(zero, now, hundredDaysAgo)
    ).map(_.map(_.toDouble)).map(v => Vectors.dense(v).toOPVector)
  }

  Spec[DateMapVectorizer[_]] should "vectorize dates correctly any time" in {
    checkAt(DateTimeUtils.now().minusHours(1))
  }

  it should "vectorize dates correctly on test date" in {
    checkAt(new JDateTime(2017, 9, 28, 15, 45, 39, DateTimeUtils.DefaultTimeZone))
    checkAt(new JDateTime(1901, 1, 1, 0, 0, 0, DateTimeUtils.DefaultTimeZone))
  }

  it should "serialize correctly" in new SampleData(DateTimeUtils.now().minusHours(1)) {
    val vectorizer = new DateMapVectorizer()
      .setInput(f1).setDefaultValue(0).setReferenceDate(moment).setTrackNulls(false)
    val workflow = new OpWorkflow()
    val model = workflow.setInputDataset(ds).setResultFeatures(vectorizer.getOutput()).train()
    model.save(modelLocation)
    val loaded = workflow.loadModel(modelLocation)
    loaded.getStages().map(_.uid) should contain (vectorizer.uid)
  }

}
