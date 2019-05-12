package com.salesforce.op.dsl

import com.salesforce.op._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.TimePeriod
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.joda.time.{DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RichDateFeatureTest extends FlatSpec with TestSparkContext {

  val testDate: Long = JDateTime.parse("2019-04-30T13:23:00.000-00:00").getMillis
  val (inputData, testDateFeature) = TestFeatureBuilder(Seq(Date(testDate)))
  val (inputData2, testDateTimeFeature) = TestFeatureBuilder(Seq(DateTime(testDate)))

  def checkFeature(feature: FeatureLike[Integral], expected: Int): Unit = {
    val transformed = feature.originStage.asInstanceOf[Transformer].transform(inputData)
    val actual = transformed.collect(feature).head.value
    actual shouldBe Some(expected)
  }

  behavior of "RichDateFeatureTest"

  it should "make RichDateFeature time period transformations" in {
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.DayOfMonth), 30)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.DayOfWeek), 2)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.DayOfYear), 120)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.HourOfDay), 13)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.MonthOfYear), 4)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.WeekOfMonth), 4)
    checkFeature(testDateFeature.toTimePeriod(TimePeriod.WeekOfYear), 18)

    checkFeature(testDateFeature.toDayOfMonth(), 30)
    checkFeature(testDateFeature.toDayOfWeek(), 2)
    checkFeature(testDateFeature.toDayOfYear(), 120)
    checkFeature(testDateFeature.toHourOfDay(), 13)
    checkFeature(testDateFeature.toMonthOfYear(), 4)
    checkFeature(testDateFeature.toWeekOfMonth(), 4)
    checkFeature(testDateFeature.toWeekOfYear(), 18)
  }

  it should "make RichDateTimeFeature time period transformations" in {
    /*
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.DayOfMonth), 30)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.DayOfWeek), 2)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.DayOfYear), 120)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.HourOfDay), 13)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.MonthOfYear), 4)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.WeekOfMonth), 4)
    checkFeature(testDateTimeFeature.toTimePeriod(TimePeriod.WeekOfYear), 18)

    checkFeature(testDateTimeFeature.toDayOfMonth(), 30)
    checkFeature(testDateTimeFeature.toDayOfWeek(), 2)
    checkFeature(testDateTimeFeature.toDayOfYear(), 120)
    checkFeature(testDateTimeFeature.toHourOfDay(), 13)
    checkFeature(testDateTimeFeature.toMonthOfYear(), 4)
    checkFeature(testDateTimeFeature.toWeekOfMonth(), 4)
    checkFeature(testDateTimeFeature.toWeekOfYear(), 18)
     */
  }

}
