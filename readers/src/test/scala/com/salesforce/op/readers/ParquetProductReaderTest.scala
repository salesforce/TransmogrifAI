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
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

// Need this case class to be external to (not nested in) ParquetProductReaderTest for spark sql to work correctly.
// Fields in the case class are case-sensitive and should exactly match the parquet column names.
case class PassengerType
(
  PassengerId: Int,
  Survived: Int,
  Pclass: Option[Int],
  Name: Option[String],
  Sex: String,
  Age: Option[Double],
  SibSp: Option[Int],
  Parch: Option[Int],
  Ticket: String,
  Fare: Double,
  Cabin: Option[String],
  Embarked: Option[String]
)

@RunWith(classOf[JUnitRunner])
class ParquetProductReaderTest extends FlatSpec with TestSparkContext with TestCommon {
  def passengerFilePath: String = s"$testDataDir/PassengerDataAll.parquet"

  val parquetRecordCount = 891

  import spark.implicits._
  val dataReader = new ParquetProductReader[PassengerType](
    readPath = Some(passengerFilePath),
    key = _.PassengerId.toString
  )

  Spec[ParquetProductReader[_]] should "read in data correctly" in {
    val data = dataReader.readDataset().collect()
    data.foreach(_ shouldBe a[PassengerType])
    data.length shouldBe parquetRecordCount
  }

  it should "read in byte arrays as valid strings" in {
    val caseReader = DataReaders.Simple.parquetCase[PassengerType](
      path = Some(passengerFilePath),
      key = _.PassengerId.toString
    )

    val records = caseReader.readDataset().collect()
    records.collect { case r if r.PassengerId == 1 => r.Ticket } shouldBe Array("A/5 21171")
  }

  it should "map the columns of data to types defined in schema" in {
    val caseReader = DataReaders.Simple.parquetCase[PassengerType](
      path = Some(passengerFilePath),
      key = _.PassengerId.toString
    )

    val records = caseReader.readDataset().collect()
    records(0).Survived shouldBe a[java.lang.Integer]
    records(0).Fare shouldBe a[java.lang.Double]
    records(0).Ticket shouldBe a[java.lang.String]
    records.collect { case r if r.PassengerId == 1 => r.Age } shouldBe Array(Some(22.0))
  }

  it should "generate a dataframe" in {
    val tokens = FeatureBuilder.TextList[PassengerType]
      .extract(p => p.Name.map(_.split(" ")).toSeq.flatten.toTextList)
      .asPredictor
    val data = dataReader.generateDataFrame(rawFeatures = Array(tokens)).collect()

    data.collect { case r if r.get(0) == "3" => r.get(1) } shouldBe Array(Array("Heikkinen,", "Miss.", "Laina"))
    data.length shouldBe parquetRecordCount
  }
}
