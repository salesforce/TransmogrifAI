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

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.joda.time.{DateTime, DateTimeConstants, DateTimeZone}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DateVectorizerTest extends FlatSpec with TestSparkContext with AttributeAsserts {
  require(DateTimeUtils.DefaultTimeZone == DateTimeZone.UTC)

  // Sunday July 12th 1998 at 22:45
  private val defaultDate = new DateTime(1998, 7, 12, 22, 45, DateTimeUtils.DefaultTimeZone).getMillis


  private def checkAt(moment: DateTime) = {
    val (ds, f1, f2, f3) = buildTestData(moment)

    val vector = f1.vectorize(
      dateListPivot = TransmogrifierDefaults.DateListDefault,
      referenceDate = moment,
      trackNulls = false,
      circularDateReps = Seq(),
      others = Array(f2, f3)
    )
    val transformed = new OpWorkflow().setResultFeatures(vector).transform(ds)
    val result = transformed.collect(vector)
    withClue(s"Checking transformation at $moment") {
      result shouldBe expectedAt(moment)
    }

    val meta = OpVectorMetadata(vector.name, transformed.schema(vector.name).metadata)
    meta.columns.length shouldBe 3
    meta.history.keys.size shouldBe 3
    val field = transformed.schema(vector.name)
    assertNominal(field, Array.fill(expectedAt(moment).head.value.size)(false), result)

    val vector2 = f1.vectorize(
      dateListPivot = TransmogrifierDefaults.DateListDefault,
      referenceDate = moment,
      trackNulls = true,
      circularDateReps = Seq(),
      others = Array(f2, f3)
    )
    val transformed2 = new OpWorkflow().setResultFeatures(vector2).transform(ds)
    val result2 = transformed2.collect(vector2)
    result2.head.v.size shouldBe 6

    val meta2 = OpVectorMetadata(vector2.name, transformed2.schema(vector2.name).metadata)
    meta2.columns.length shouldBe 6
    meta2.history.keys.size shouldBe 3
    val field2 = transformed2.schema(vector2.name)
    assertNominal(field2, Array.fill(expectedAt(moment).head.value.size)(Seq(false, true)).flatten, result2)

    val vector3 = f1.vectorize(
      dateListPivot = TransmogrifierDefaults.DateListDefault,
      others = Array(f2, f3)
    )
    val transformed3 = new OpWorkflow().setResultFeatures(vector3).transform(ds)
    val result3 = transformed3.collect(vector3)
    result3.head.v.size shouldBe 30

    val meta3 = OpVectorMetadata(vector3.name, transformed3.schema(vector3.name).metadata)
    meta3.columns.length shouldBe 30
    meta3.history.keys.size shouldBe 6
    val field3 = transformed3.schema(vector3.name)
    val expectedNominal = Array.fill(24)(false) ++ Array.fill(3)(Seq(false, true)).flatten.asInstanceOf[Array[Boolean]]
    assertNominal(field3, expectedNominal, result3)
  }

  private def buildTestData(moment: DateTime) = {
    TestFeatureBuilder(
      Seq[(Date, Date, Date)](
        (Date(1), Date.empty, Date(0)),
        (Date(1), Date(defaultDate), Date(3 * DateTimeConstants.MILLIS_PER_DAY)),
        (Date.empty, Date(0), Date(moment.plusDays(100).plusMinutes(1).getMillis)) // 100d and 1m in the future
      )
    )
  }

  private def expectedAt(moment: DateTime) = {
    val nowMinusMilli = moment.minus(1L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val now = moment.minus(0L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val zero = 0
    val threeDaysAgo = moment.minus(3 * DateTimeConstants.MILLIS_PER_DAY).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val defaultTimeAgo = moment.minus(defaultDate).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val hundredDaysAgo = DateTimeUtils.getStandardDays(
      moment.plusDays(100).getMillis,
      moment.getMillis
    )

    Array(
      Array(nowMinusMilli, zero, now),
      Array(nowMinusMilli, defaultTimeAgo, threeDaysAgo),
      Array(zero, now, hundredDaysAgo)
    ).map(_.map(_.toDouble)).map(v => Vectors.dense(v).toOPVector)
  }

  it should "vectorize dates correctly at a variety of moments" in {
    for {hour <- 0 until 24} {
      val moment = new DateTime(2017, 9, 27, hour, 45, 39, DateTimeUtils.DefaultTimeZone)
      checkAt(moment)
    }
  }

  it should "vectorize dates correctly on default reference date" in {
    checkAt(DateTimeUtils.now())
  }
}
