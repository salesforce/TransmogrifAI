/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.feature.TimePeriod._
import com.salesforce.op.test.{TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.OpVectorMetadata
import com.salesforce.op.utils.spark.RichDataset._

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Transformer
import org.joda.time.{DateTime => JDateTime}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class DateToUnitCircleTransformerTest extends FlatSpec with TestSparkContext {

  val eps = 1E-4
  val sampleDateTimes = Seq[JDateTime](
    new JDateTime(2018, 2, 11, 0, 0, 0, 0),
    new JDateTime(2018, 11, 28, 6, 0, 0, 0),
    new JDateTime(2018, 2, 17, 12, 0, 0, 0),
    new JDateTime(2017, 4, 17, 18, 0, 0, 0),
    new JDateTime(1918, 2, 13, 3, 0, 0, 0)
  )
  val expectedHourOfDayOutput = Array(
    Array(1.0, 0.0),
    Array(0.0, 1.0),
    Array(-1.0, 0.0),
    Array(0.0, -1.0),
    Array(math.sqrt(2.0) / 2, math.sqrt(2.0) / 2)
  ).map(Vectors.dense(_).toOPVector)

  def transformData[T <: TimePeriod](data: Seq[JDateTime], timePeriod: T): Array[OPVector] = {
    val dataTimeStamps: Seq[Date] = data.map(x => Date(x.getMillis()))
    val (ds, f) = TestFeatureBuilder(dataTimeStamps)
    val vectorizer = new DateToUnitCircleTransformer().setTimePeriod(timePeriod).setInput(f)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    transformed.collect(vector)
  }

  def indexSeqToUnitCircle(indices: Seq[Int], numIndices: Int): Seq[OPVector] = {
    indices.map(x => Array(math.cos(2 * math.Pi * x / numIndices), math.sin(2 * math.Pi * x / numIndices)))
      .map(Vectors.dense(_).toOPVector)
  }

  Spec[DateToUnitCircleTransformer[_]] should
    "take an array of features as input and return a single vector feature" in {
    val dataTimeStamps: Seq[Date] = sampleDateTimes.map(x => Date(x.getMillis()))
    val (ds, f) = TestFeatureBuilder(dataTimeStamps)
    val vectorizer = new DateToUnitCircleTransformer().setInput(f)
    val vector = vectorizer.getOutput()
    vector.name shouldBe vectorizer.getOutputFeatureName
    vector.typeName shouldBe FeatureType.typeName[OPVector]
    vector.isResponse shouldBe false
  }

  it should "work with its shortcut" in {
    val dataTimeStamps: Seq[Date] = sampleDateTimes.map(x => Date(x.getMillis()))
    val (ds, dateFeature) = TestFeatureBuilder(dataTimeStamps)
    val output = dateFeature.toUnitCircle(TimePeriod.HourOfDay)
    val transformed = output.originStage.asInstanceOf[Transformer].transform(ds)
    val actual = transformed.collect(output)
    all (actual.zip(expectedHourOfDayOutput).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "work with its DateTime shortcut" in {
    val dataTimeStamps: Seq[DateTime] = sampleDateTimes.map(x => DateTime(x.getMillis()))
    val (ds, dateTimeFeature) = TestFeatureBuilder(dataTimeStamps)
    val output = dateTimeFeature.toUnitCircle(TimePeriod.HourOfDay)
    val transformed = output.originStage.asInstanceOf[Transformer].transform(ds)
    val actual = transformed.collect(output)
    all (actual.zip(expectedHourOfDayOutput).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "store the proper meta data" in {
    val dataTimeStamps: Seq[Date] = sampleDateTimes.map(x => Date(x.getMillis()))
    val (ds, feature) = TestFeatureBuilder(dataTimeStamps)
    val vectorizer = new DateToUnitCircleTransformer().setTimePeriod(HourOfDay).setInput(feature)
    val transformed = vectorizer.transform(ds)
    val meta = OpVectorMetadata(transformed.schema(vectorizer.getOutput().name))
    meta.columns.length should equal (2)
    meta.columns(0).indicatorValue.get should equal("x_HourOfDay")
    meta.columns(1).indicatorValue.get should equal("y_HourOfDay")
  }

  it should "transform the data correctly when there are null dates" in {
    val sampleDateTimesWithNulls = Seq[Date](
      Date.empty,
      Date(new JDateTime(2018, 2, 11, 0, 0, 0, 0).getMillis())
    )
    val (ds, f) = TestFeatureBuilder(sampleDateTimesWithNulls)
    val vectorizer = new DateToUnitCircleTransformer().setTimePeriod(HourOfDay).setInput(f)
    val transformed = vectorizer.transform(ds)
    val vector = vectorizer.getOutput()
    val actual = transformed.collect(vector)
    val expected = Array(
      Array(0.0, 0.0),
      Array(1.0, 0.0)
    ).map(Vectors.dense(_).toOPVector)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is HourOfDay" in {
    val actual = transformData(sampleDateTimes, HourOfDay)
    all (actual.zip(expectedHourOfDayOutput).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be <  eps
  }

  it should "transform the data correctly when the timePeriod is DayOfYear" in {
    val dateTimes = Seq[JDateTime](
      new JDateTime(2018, 1, 1, 0, 0, 0, 0),
      new JDateTime(2018, 1, 2, 6, 0, 0, 0),
      new JDateTime(2018, 1, 3, 12, 0, 0, 0),
      new JDateTime(2017, 1, 4, 12, 0, 0, 0),
      new JDateTime(1918, 2, 1, 3, 0, 0, 0)
    )
    val actual = transformData(dateTimes, DayOfYear)
    val sampleDaysOfYearMinusOne = Array(0, 1, 2, 3, 31)
    val expected = indexSeqToUnitCircle(sampleDaysOfYearMinusOne, 366)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is DayOfWeek" in {
    val dateTimes = Seq[JDateTime](
      new JDateTime(2018, 2, 12, 0, 0, 0, 0), // Monday
      new JDateTime(2018, 2, 13, 0, 0, 0, 0),
      new JDateTime(2018, 2, 14, 0, 0, 0, 0),
      new JDateTime(2018, 2, 15, 0, 0, 0, 0),
      new JDateTime(2018, 2, 16, 0, 0, 0, 0),
      new JDateTime(2018, 2, 17, 0, 0, 0, 0),
      new JDateTime(2018, 2, 18, 0, 0, 0, 0)
    )
    val actual = transformData(dateTimes, DayOfWeek)
    val expectedDaysOfWeekMinusOne = indexSeqToUnitCircle(Seq(0, 1, 2, 3, 4, 5, 6), 7)
    all (actual.zip(expectedDaysOfWeekMinusOne).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is WeekOfYear" in {
    val dateTimes = Seq[JDateTime](
      new JDateTime(2017, 1, 1, 0, 0, 0, 0), // Sunday: last week of the week year
      new JDateTime(2017, 1, 2, 0, 0, 0, 0), // Monday: first week of the week year
      new JDateTime(2017, 1, 9, 0, 0, 0, 0),
      new JDateTime(2017, 1, 16, 0, 0, 0, 0),
      new JDateTime(2017, 1, 23, 0, 0, 0, 0)
    )
    val actual = transformData(dateTimes, WeekOfYear)
    val sampleWeeksOfYearMinusOne = Seq(51, 0, 1, 2, 3)
    val expected = indexSeqToUnitCircle(sampleWeeksOfYearMinusOne, 53)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is DayOfMonth" in {
    val actual = transformData(sampleDateTimes, DayOfMonth)
    val sampleDaysOfMonthMinusOne = Seq(10, 27, 16, 16, 12)
    val expected = indexSeqToUnitCircle(sampleDaysOfMonthMinusOne, 31)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is MonthOfYear" in {
    val dateTimes = Seq[JDateTime](
      new JDateTime(2017, 1, 1, 0, 0, 0, 0),
      new JDateTime(2017, 2, 1, 0, 0, 0, 0),
      new JDateTime(2017, 3, 9, 0, 0, 0, 0),
      new JDateTime(2017, 4, 16, 0, 0, 0, 0),
      new JDateTime(2017, 12, 23, 0, 0, 0, 0)
    )
    val actual = transformData(dateTimes, MonthOfYear)
    val sampleMonthsOfYearMinusOne = Seq(0, 1, 2, 3, 11)
    val expected = indexSeqToUnitCircle(sampleMonthsOfYearMinusOne, 12)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }

  it should "transform the data correctly when the timePeriod is WeekOfMonth" in {
    val dateTimes = Seq[JDateTime](
      new JDateTime(2018, 2, 1, 0, 0, 0, 0),
      new JDateTime(2018, 2, 5, 0, 0, 0, 0),
      new JDateTime(2018, 2, 12, 0, 0, 0, 0),
      new JDateTime(2018, 2, 19, 0, 0, 0, 0),
      new JDateTime(2018, 2, 26, 0, 0, 0, 0)
    )
    val actual = transformData(dateTimes, WeekOfMonth)
    val expected = indexSeqToUnitCircle(Seq(0, 1, 2, 3, 4), 6)
    all (actual.zip(expected).map(g => Vectors.sqdist(g._1.value, g._2.value))) should be < eps
  }
}
