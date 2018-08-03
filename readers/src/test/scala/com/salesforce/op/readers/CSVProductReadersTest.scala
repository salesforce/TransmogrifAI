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
