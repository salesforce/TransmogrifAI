/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import org.joda.time.Duration

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.io.csv.CSVOptions
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import scala.reflect.runtime.universe._


// need this to be external to (not nested in) CSVProductReaderTest for spark sql to work correctly
case class PassengerCaseClass
(
  passengerId: Int,
  age: Option[Int],
  gender: Option[String],
  height: Option[Int],
  weight: Option[scala.math.BigInt],
  description: Option[String],
  boarded: Option[Long],
  recordDate: Option[Long],
  survived: Option[Boolean],
  randomTime: Option[java.sql.Timestamp],
  randomFloating: Option[Double]
)

@RunWith(classOf[JUnitRunner])
class CSVProductReadersTest extends FlatSpec with TestSparkContext {

  def testDataPath: String = "../test-data"

  def csvWithoutHeaderPath: String = s"$testDataPath/BigPassenger.csv"

  def csvWithHeaderPath: String = s"$testDataPath/BigPassengerWithHeader.csv"

  val age = FeatureBuilder.Integral[PassengerCaseClass]
    .extract(_.age.toIntegral)
    .asPredictor

  val survived = FeatureBuilder.Binary[PassengerCaseClass]
    .extract(_.survived.toBinary)
    .aggregate(zero = Some(true), (l, r) => Some(l.getOrElse(false) && r.getOrElse(false)))
    .asResponse


  import spark.implicits._

  Spec[CSVProductReader[_]] should "read in data correctly with header" in {
    val dataReader = new CSVProductReader[PassengerCaseClass](
      readPath = Some(csvWithHeaderPath),
      key = _.passengerId.toString,
      options = CSVOptions(header = true)
    )
    val data = dataReader.readDataset().collect()
    data.foreach(_ shouldBe a[PassengerCaseClass])
    data.length shouldBe 8
  }

  it should "read in data correctly without header" in {
    val dataReader = DataReaders.Simple.csvCase[PassengerCaseClass](
      path = Some(csvWithoutHeaderPath),
      key = _.passengerId.toString
    )
    val data = dataReader.readDataset().collect()
    data.foreach(_ shouldBe a[PassengerCaseClass])
    data.length shouldBe 8
  }

  it should "generate a dataframe" in {
    val dataReader = new CSVProductReader[PassengerCaseClass](
      readPath = Some(csvWithHeaderPath),
      key = _.passengerId.toString,
      options = CSVOptions(header = true)
    )
    val tokens =
      FeatureBuilder.TextList[PassengerCaseClass]
        .extract(p => p.description.map(_.split(" ")).toSeq.flatten.toTextList).asPredictor
    val data = dataReader.generateDataFrame(rawFeatures = Array(tokens)).collect()
    data.collect { case r if r.get(0) == "3" => r.get(1) } shouldBe Array(Array("this", "is", "a", "description"))
    data.length shouldBe 8
  }

  Spec[AggregateCSVProductReader[_]] should "read and aggregate data correctly" in {
    val dataReader = DataReaders.Aggregate.csvCase[PassengerCaseClass](
      path = Some(csvWithoutHeaderPath),
      key = _.passengerId.toString,
      aggregateParams = AggregateParams(
        timeStampFn = Some[PassengerCaseClass => Long](_.recordDate.getOrElse(0L)),
        cutOffTime = CutOffTime.UnixEpoch(1471046600)
      )
    )

    val data = dataReader.readDataset().collect()
    data.foreach(_ shouldBe a[PassengerCaseClass])
    data.length shouldBe 8

    val aggregatedData = dataReader.generateDataFrame(rawFeatures = Array(age, survived)).collect()
    aggregatedData.length shouldBe 6
    aggregatedData.collect { case r if r.get(0) == "4" => r} shouldEqual Array(Row("4", 60, false))

    dataReader.fullTypeName shouldBe typeOf[PassengerCaseClass].toString
  }

  Spec[ConditionalCSVProductReader[_]] should "read and conditionally aggregate data correctly" in {
    val dataReader = DataReaders.Conditional.csvCase[PassengerCaseClass](
      path = Some(csvWithoutHeaderPath),
      key = _.passengerId.toString,
      conditionalParams = ConditionalParams(
        timeStampFn = _.recordDate.getOrElse(0L),
        targetCondition = _.height.contains(186), // Function to figure out if target event has occurred
        responseWindow = Some(Duration.millis(800)), // How many days after target event to aggregate for response
        predictorWindow = None, // How many days before target event to include in predictor aggregation
        timeStampToKeep = TimeStampToKeep.Min,
        dropIfTargetConditionNotMet = true
      )
    )

    val data = dataReader.readDataset().collect()
    data.foreach(_ shouldBe a[PassengerCaseClass])
    data.length shouldBe 8

    val aggregatedData = dataReader.generateDataFrame(rawFeatures = Array(age, survived)).collect()
    aggregatedData.length shouldBe 2
    aggregatedData shouldEqual Array(Row("3", null, true), Row("4", 10, false))

    dataReader.fullTypeName shouldBe typeOf[PassengerCaseClass].toString
  }
}
