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

package com.salesforce.op.test

import java.io.File

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.readers._
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.Suite

import scala.language.postfixOps

trait PassengerSparkFixtureTest extends TestSparkContext with PassengerFeaturesTest {
  self: Suite =>

  def testDataPath: String = {
    Some(new File("test-data")) filter (_.isDirectory) getOrElse new File("../test-data") getPath
  }
  def passengerAvroPath: String = s"$testDataPath/PassengerData.avro"
  def passengerCsvPath: String = s"$testDataPath/PassengerData.csv"
  def passengerCsvWithHeaderPath: String = s"$testDataPath/PassengerDataWithHeader.csv"
  def passengerProfileCsvPath: String = s"$testDataPath/PassengerProfileData.csv"

  lazy val simpleReader: AvroReader[Passenger] =
    DataReaders.Simple.avro[Passenger](
      path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
      key = _.getPassengerId.toString // Entity to score
    )

  val simpleCsvReader = DataReaders.Simple.csv[PassengerCSV](
    path = Some(passengerCsvPath), // Path should be optional so can also pass in as a parameter
    schema = PassengerCSV.getClassSchema.toString, // Input schema
    key = _.getPassengerId.toString // Entity to score
  )

  val simpleStreamingReader = StreamingReaders.Simple.avro[Passenger](
    key = _.getPassengerId.toString  // Entity to score
  )

  lazy val dataReader: AggregateAvroReader[Passenger] =
    DataReaders.Aggregate.avro[Passenger](
      path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
      key = _.getPassengerId.toString, // Entity to score
      aggregateParams = AggregateParams(Option(_.getRecordDate.toLong), CutOffTime.UnixEpoch(1471046600))
    )

  lazy val profileReader: CSVReader[PassengerProfile] = DataReaders.Simple.csv[PassengerProfile](
    path = Some(passengerProfileCsvPath),
    schema = PassengerProfile.getClassSchema.toString,
    key = _.getPassengerId.toString // Entity to score
  )

  lazy val passengersDataSet: DataFrame = dataReader.generateDataFrame(
    Array(survived, age, gender, height, weight, description, boarded, stringMap, numericMap, booleanMap)
  )

  lazy val passengersArray: Array[Row] = passengersDataSet.collect()
}
