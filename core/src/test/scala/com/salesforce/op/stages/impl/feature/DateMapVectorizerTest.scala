/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */
package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.OpWorkflow
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.OpVectorMetadata
import org.apache.spark.ml.linalg.Vectors
import org.joda.time.{DateTimeConstants, Days, DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DateMapVectorizerTest extends FlatSpec with TestSparkContext {

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
    val vector = f1.vectorize(defaultValue = 0, referenceDate = moment, trackNulls = false)
    val transformed = new OpWorkflow().setResultFeatures(vector).transform(ds)
    withClue(s"Checking transformation at $moment") {
      transformed.collect(vector) shouldBe expected(moment)
    }
    val meta = OpVectorMetadata(vector.name, transformed.schema(vector.name).metadata)
    meta.columns.length shouldBe 3
    meta.columns.map(_.indicatorGroup) should contain theSameElementsAs Array(Option("a"), Option("b"), Option("c"))

    val vector2 = f1.vectorize(defaultValue = 0, referenceDate = moment, trackNulls = true)
    val transformed2 = new OpWorkflow().setResultFeatures(vector2).transform(ds)
    transformed2.collect(vector2).head.v.size shouldBe 6

    val meta2 = OpVectorMetadata(vector2.name, transformed2.schema(vector2.name).metadata)
    meta2.columns.length shouldBe 6
    meta2.history.keys.size shouldBe 1
  }

  private def expected(moment: JDateTime) = {
    val nowMinusMilli = moment.minus(1L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val now = moment.minus(0L).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val zero = 0
    val threeDaysAgo = moment.minus(3 * DateTimeConstants.MILLIS_PER_DAY).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val defaultTimeAgo = moment.minus(defaultDate).getMillis / DateTimeConstants.MILLIS_PER_DAY
    val hundredDaysAgo = Days
      .daysBetween(new JDateTime(moment.plusDays(100).getMillis, DateTimeUtils.DefaultTimeZone), moment)
      .getDays

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
  }

  it should "serialize correctly" in new SampleData(DateTimeUtils.now().minusHours(1)) {
    val vectorizer = new DateMapVectorizer()
      .setInput(f1).setDefaultValue(0).setReferenceDate(moment).setTrackNulls(false)
    val workflow = new OpWorkflow()
    val model = workflow.setInputDataset(ds).setResultFeatures(vectorizer.getOutput()).train()
    model.save(modelLocation)
    val loaded = workflow.loadModel(modelLocation)
    loaded.stages.map(_.uid) should contain (vectorizer.uid)
  }

}
