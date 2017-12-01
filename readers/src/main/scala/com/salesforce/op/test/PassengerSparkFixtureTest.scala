/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import language.postfixOps
import java.io.File

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.readers._
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.Suite

trait PassengerSparkFixtureTest extends TestSparkContext with PassengerFeaturesTest {
  self: Suite =>

  def testDataPath: String = {
    Some(new File("test-data")) filter (_.isDirectory) getOrElse new File("../test-data") getPath
  }
  def passengerAvroPath: String = s"$testDataPath/PassengerData.avro"
  def passengerCsvPath: String = s"$testDataPath/PassengerData.csv"
  def passengerCsvWithHeaderPath: String = s"$testDataPath/PassengerDataWithHeader.csv"
  def passengerProfileCsvPath: String = s"$testDataPath/PassengerProfileData.csv"

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
