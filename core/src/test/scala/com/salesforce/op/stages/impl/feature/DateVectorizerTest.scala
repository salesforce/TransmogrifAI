/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.linalg.Vectors
import org.joda.time.{DateTime, DateTimeConstants, DateTimeZone, Days}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class DateVectorizerTest extends FlatSpec with TestSparkContext {
  require(DateTimeUtils.DefaultTimeZone == DateTimeZone.UTC)

  // Sunday July 12th 1998 at 22:45
  private val defaultDate = new DateTime(1998, 7, 12, 22, 45, DateTimeUtils.DefaultTimeZone).getMillis


  private def checkAt(moment: DateTime) = {
    val (ds, f1, f2, f3) = buildTestData(moment)

    val vector = f1.vectorize(
      dateListPivot = TransmogrifierDefaults.DateListDefault,
      referenceDate = moment,
      trackNulls = false,
      others = Array(f2, f3)
    )
    val transformed = new OpWorkflow().setResultFeatures(vector).transform(ds)
    withClue(s"Checking transformation at $moment") {
      transformed.collect(vector) shouldBe expectedAt(moment)
    }

    val meta = OpVectorMetadata(vector.name, transformed.schema(vector.name).metadata)
    meta.columns.length shouldBe 3
    meta.history.keys.size shouldBe 3

    val vector2 = f1.vectorize(
      dateListPivot = TransmogrifierDefaults.DateListDefault,
      referenceDate = moment,
      trackNulls = true,
      others = Array(f2, f3)
    )
    val transformed2 = new OpWorkflow().setResultFeatures(vector2).transform(ds)
    transformed2.collect(vector2).head.v.size shouldBe 6

    val meta2 = OpVectorMetadata(vector2.name, transformed2.schema(vector2.name).metadata)
    meta2.columns.length shouldBe 6
    meta2.history.keys.size shouldBe 3
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
    val hundredDaysAgo = Days
      .daysBetween(new DateTime(moment.plusDays(100).getMillis, DateTimeUtils.DefaultTimeZone), moment)
      .getDays

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
