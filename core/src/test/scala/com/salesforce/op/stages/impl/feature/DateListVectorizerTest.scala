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

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.DateListPivot._
import com.salesforce.op.test.TestOpVectorColumnType.IndCol
import com.salesforce.op.test.{TestFeatureBuilder, TestOpVectorMetadataBuilder, TestSparkContext}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.joda.time.{DateTime, DateTimeConstants}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DateListVectorizerTest extends FlatSpec with TestSparkContext {

  // Sunday July 12th 1998 at 22:45
  val defaultDate = new DateTime(1998, 7, 12, 22, 45, DateTimeUtils.DefaultTimeZone).getMillis
  val now = TransmogrifierDefaults.ReferenceDate.minusMillis(1).getMillis // make date time be in the past

  private def daysToMilliseconds(n: Int): Long = n * DateTimeConstants.MILLIS_PER_DAY
  private def monthsToMilliseconds(n: Int): Long = n * 2628000000L
  private def hoursToMilliseconds(n: Int): Long = n * DateTimeConstants.MILLIS_PER_HOUR

  val (testData, clicks, opens, purchases) = TestFeatureBuilder("clicks", "opens", "purchases",
    Seq(
      (Seq(defaultDate, defaultDate + daysToMilliseconds(7),
        defaultDate + monthsToMilliseconds(12)),
        Seq(defaultDate + daysToMilliseconds(1), defaultDate, defaultDate + daysToMilliseconds(8)),
        Seq(defaultDate)),
      (Seq(defaultDate, defaultDate - daysToMilliseconds(3), defaultDate + daysToMilliseconds(4)),
        Seq(defaultDate - monthsToMilliseconds(1), defaultDate + monthsToMilliseconds(11)),
        Seq(defaultDate - hoursToMilliseconds(3), defaultDate + hoursToMilliseconds(21))),
      (Seq(defaultDate, defaultDate + daysToMilliseconds(7), defaultDate + daysToMilliseconds(1),
        defaultDate + daysToMilliseconds(8)),
        Seq(defaultDate, defaultDate + monthsToMilliseconds(24), defaultDate - monthsToMilliseconds(2),
          defaultDate + monthsToMilliseconds(10)),
        Seq(defaultDate, defaultDate + hoursToMilliseconds(24), defaultDate - hoursToMilliseconds(6),
          defaultDate + hoursToMilliseconds(18))),
      (Seq.empty[Long], Seq.empty[Long], Seq.empty[Long])
    ).map(v => (v._1.toDateList, v._2.toDateList, v._3.toDateList))
  )

  val (testDataCurrent, _, _, _) = TestFeatureBuilder(clicks.name, opens.name, purchases.name,
    Seq(
      (Seq.empty[Long], Seq.empty[Long], Seq.empty[Long]),
      (Seq(now - daysToMilliseconds(2), now - daysToMilliseconds(3), now),
        Seq(now, now + daysToMilliseconds(2) + 600000L, now + daysToMilliseconds(1)),
        Seq(now)),
      (Seq(now - 1L),
        Seq(now),
        Seq(now, now + daysToMilliseconds(4) + 600000L)),
      (Seq(now, now - 34L, now - daysToMilliseconds(2),
        now + daysToMilliseconds(2) + 600000L),
        Seq(now + daysToMilliseconds(2) + hoursToMilliseconds(1), now - daysToMilliseconds(1) + 1L),
        Seq(now + daysToMilliseconds(1) + 600000L))
    ).map(v => (v._1.toDateList, v._2.toDateList, v._3.toDateList))
  )
  val testVectorizer = new DateListVectorizer[DateList]()
  val outputName = "vecDateList"

  Spec[DateListVectorizer[_]] should "have output name set correctly" in {
    testVectorizer.operationName shouldBe outputName
  }

  it should "throw an error if you try to get the output without setting the inputs" in {
    intercept[java.util.NoSuchElementException](testVectorizer.getOutput())
  }

  it should "return a single output feature of the correct type" in {
    val output = testVectorizer.setInput(clicks, opens, purchases).getOutput()
    output shouldBe new Feature[OPVector](
      name = testVectorizer.getOutputFeatureName,
      originStage = testVectorizer,
      isResponse = false,
      parents = Array(clicks, opens, purchases)
    )
  }

  it should "vectorize with SinceFirst" in {
    val testModelTimeSinceFirst = testVectorizer.setInput(clicks, opens, purchases).setPivot(SinceFirst)
      .setTrackNulls(false)

    testModelTimeSinceFirst.transformFn(Seq(
      Seq(now - daysToMilliseconds(1), now).toDateList,
      Seq(now - daysToMilliseconds(20), now - daysToMilliseconds(1)).toDateList,
      Seq(now).toDateList
    )) shouldEqual Vectors.dense(1.0, 20.0, 0.0).toOPVector

    val transformed = testModelTimeSinceFirst.transform(testDataCurrent)
    val output = testModelTimeSinceFirst.getOutput()
    transformed.schema.fieldNames shouldEqual Array(clicks.name, opens.name, purchases.name, output.name)

    transformed.collect(output) shouldBe Array(
      Vectors.dense(0.0, 0.0, 0.0).toOPVector,
      Vectors.dense(3.0, 0.0, 0.0).toOPVector,
      Vectors.dense(0.0, 0.0, 0.0).toOPVector,
      Vectors.dense(2.0, 1.0, -1.0).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelTimeSinceFirst.getMetadata() shouldEqual fieldMetadata
  }

  it should "vectorize with SinceFirst and track nulls" in {
    val testModelTimeSinceFirst = testVectorizer.setInput(clicks, opens, purchases).setPivot(SinceFirst)
      .setTrackNulls(true)

    testModelTimeSinceFirst.transformFn(Seq(
      Seq(now - daysToMilliseconds(1), now).toDateList,
      Seq(now - daysToMilliseconds(20), now - daysToMilliseconds(1)).toDateList,
      Seq(now).toDateList,
      Seq().toDateList
    )) shouldEqual Vectors.dense(1.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0).toOPVector

    val transformed = testModelTimeSinceFirst.transform(testDataCurrent)
    val output = testModelTimeSinceFirst.getOutput()
    transformed.schema.fieldNames shouldEqual Array(clicks.name, opens.name, purchases.name, output.name)

    transformed.collect(output) shouldBe Array(
      Vectors.dense(0.0, 1.0, 0.0, 1.0, 0.0, 1.0).toOPVector,
      Vectors.dense(3.0, 0.0, 0.0, 0.0, 0.0, 0.0).toOPVector,
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).toOPVector,
      Vectors.dense(2.0, 0.0, 1.0, 0.0, -1.0, 0.0).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelTimeSinceFirst.getMetadata() shouldEqual fieldMetadata
  }

  it should "vectorize with SinceFirst and reference date in the past" in {
    val testModelTimeSinceFirst =
      testVectorizer.setInput(clicks, opens, purchases).setPivot(SinceFirst).setTrackNulls(false)
        .setReferenceDate(TransmogrifierDefaults.ReferenceDate.minusDays(30).minusMillis(2))

    testModelTimeSinceFirst.transformFn(Seq(
      Seq(now - daysToMilliseconds(1), now).toDateList,
      Seq(now - daysToMilliseconds(20), now - daysToMilliseconds(1)).toDateList,
      Seq(now).toDateList
    )) shouldEqual Vectors.dense(-29.0, -10.0, -30.0).toOPVector

    val transformed = testModelTimeSinceFirst.transform(testDataCurrent)
    val output = testModelTimeSinceFirst.getOutput()
    transformed.schema.fieldNames shouldEqual Array(clicks.name, opens.name, purchases.name, output.name)

    transformed.collect(output) shouldBe Array(
      Vectors.dense(0.0, 0.0, 0.0).toOPVector,
      Vectors.dense(-27.0, -30.0, -30.0).toOPVector,
      Vectors.dense(-30.0, -30.0, -30.0).toOPVector,
      Vectors.dense(-28.0, -29.0, -31.0).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelTimeSinceFirst.getMetadata() shouldEqual fieldMetadata
  }

  it should "vectorize with ModeDay" in {
    val testModelModeDay = testVectorizer.setInput(clicks, opens, purchases).setPivot(ModeDay).setTrackNulls(false)
    testModelModeDay.transformFn(Seq(
      Seq(defaultDate).toDateList,
      Seq(defaultDate + daysToMilliseconds(1), defaultDate).toDateList,
      Seq(defaultDate, defaultDate + daysToMilliseconds(2), defaultDate + daysToMilliseconds(9)).toDateList
    )) shouldEqual Vectors.sparse(21, Array(6, 7, 15), Array(1.0, 1.0, 1.0)).toOPVector

    val transformed = testModelModeDay.transform(testData)

    val output = testModelModeDay.getOutput()
    transformed.collect(output) shouldBe Array(
      Vectors.sparse(21, Array(6, 7, 20), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(21, Array(3, 11, 14), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(21, Array(0, 8, 14), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(21, Array(), Array()).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelModeDay.getMetadata() shouldEqual fieldMetadata

    val daysOfWeek = List("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday").map(s =>
      IndCol(Some(s))
    )

    OpVectorMetadata(output.name, fieldMetadata) shouldEqual
      TestOpVectorMetadataBuilder(testVectorizer, clicks -> daysOfWeek, opens -> daysOfWeek, purchases -> daysOfWeek)
  }

  it should "vectorize with ModeDay with track nulls" in {
    val testModelModeDay = testVectorizer.setInput(clicks, opens, purchases).setPivot(ModeDay).setTrackNulls(true)
    testModelModeDay.transformFn(Seq(
      Seq(defaultDate).toDateList,
      Seq(defaultDate + daysToMilliseconds(1), defaultDate).toDateList,
      Seq(defaultDate, defaultDate + daysToMilliseconds(2), defaultDate + daysToMilliseconds(9)).toDateList
    )) shouldEqual Vectors.sparse(24, Array(6, 8, 17), Array(1.0, 1.0, 1.0)).toOPVector

    val transformed = testModelModeDay.transform(testData)

    val output = testModelModeDay.getOutput()
    transformed.collect(output) shouldBe Array(
      Vectors.sparse(24, Array(6, 8, 22), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(24, Array(3, 12, 16), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(24, Array(0, 9, 16), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(24, Array(7, 15, 23), Array(1.0, 1.0, 1.0)).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelModeDay.getMetadata() shouldEqual fieldMetadata

    val daysOfWeek = List("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
      TransmogrifierDefaults.NullString).map(s => IndCol(Some(s)))

    OpVectorMetadata(output.name, fieldMetadata) shouldEqual
      TestOpVectorMetadataBuilder(testVectorizer, clicks -> daysOfWeek, opens -> daysOfWeek, purchases -> daysOfWeek)
  }

  it should "vectorize with ModeMonth" in {
    val testModelModeMonth = testVectorizer.setInput(clicks, opens, purchases).setPivot(ModeMonth).setTrackNulls(false)
    testModelModeMonth.transformFn(Seq(
      Seq(defaultDate).toDateList,
      Seq(defaultDate + monthsToMilliseconds(1), defaultDate).toDateList,
      Seq(defaultDate, defaultDate + monthsToMilliseconds(2), defaultDate + monthsToMilliseconds(9)).toDateList
    )) shouldEqual Vectors.sparse(36, Array(6, 18, 27), Array(1.0, 1.0, 1.0)).toOPVector

    val transformed = testModelModeMonth.transform(testData)

    val output = testModelModeMonth.getOutput()
    transformed.collect(output) shouldBe Array(
      Vectors.sparse(36, Array(6, 18, 30), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(36, Array(6, 17, 30), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(36, Array(6, 16, 30), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(36, Array(), Array()).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelModeMonth.getMetadata() shouldEqual fieldMetadata

    val months = List(
      "January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December"
    ).map(s => IndCol(Some(s)))

    OpVectorMetadata(output.name, fieldMetadata) shouldEqual
      TestOpVectorMetadataBuilder(testVectorizer, clicks -> months, opens -> months, purchases -> months)
  }

  it should "vectorize with ModeHour" in {
    val testModelModeHour = testVectorizer.setInput(clicks, opens, purchases).setPivot(ModeHour).setTrackNulls(false)
    testModelModeHour.transformFn(Seq(
      Seq(defaultDate).toDateList,
      Seq(defaultDate + hoursToMilliseconds(1), defaultDate).toDateList,
      Seq(defaultDate, defaultDate + hoursToMilliseconds(2), defaultDate + hoursToMilliseconds(9)).toDateList
    )) shouldEqual Vectors.sparse(72, Array(22, 46, 48), Array(1.0, 1.0, 1.0)).toOPVector

    val transformed = testModelModeHour.transform(testData)

    val output = testModelModeHour.getOutput()
    transformed.collect(output) shouldBe Array(
      Vectors.sparse(72, Array(22, 46, 70), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(72, Array(22, 36, 67), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(72, Array(22, 26, 64), Array(1.0, 1.0, 1.0)).toOPVector,
      Vectors.sparse(72, Array(), Array()).toOPVector
    )

    val fieldMetadata = transformed.schema(output.name).metadata
    testModelModeHour.getMetadata() shouldEqual fieldMetadata

    val hours = (0 until 24).map(i => IndCol(Some(s"$i:00"))).toList

    OpVectorMetadata(output.name, fieldMetadata) shouldEqual
      TestOpVectorMetadataBuilder(testVectorizer, clicks -> hours, opens -> hours, purchases -> hours)
  }
}
