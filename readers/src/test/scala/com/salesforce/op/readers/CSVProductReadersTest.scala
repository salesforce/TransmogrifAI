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

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.test.{TestCommon, TestSparkContext}
import com.salesforce.op.utils.io.csv.CSVOptions
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


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
class CSVProductReadersTest extends FlatSpec with TestSparkContext with TestCommon {
  def csvWithoutHeaderPath: String = s"$testDataDir/BigPassenger.csv"

  def csvWithHeaderPath: String = s"$testDataDir/BigPassengerWithHeader.csv"

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
}
