/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.test._
import org.joda.time.{DateTimeConstants, Duration}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class JoinedReadersTest extends FlatSpec with PassengerSparkFixtureTest {

  val sparkReader = DataReaders.Aggregate.csv[SparkExample](
    path = Some("../test-data/SparkExample.csv"),
    schema = SparkExample.getClassSchema.toString,
    key = _.getLabel.toString,
    aggregateParams = AggregateParams(None, CutOffTime.NoCutoff())
  )

  val passengerReader = DataReaders.Conditional.avro[Passenger](
    path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
    key = _.getPassengerId.toString, // Entity to score
    conditionalParams = ConditionalParams(
      timeStampFn = _.getRecordDate.toLong, // Record field which defines the date for the rest of the columns
      targetCondition = _.getBoarded >= 1471046600, // Function to figure out if target event has occurred
      responseWindow = None, // How many days after target event to include in response aggregation
      predictorWindow = None, // How many days before target event to include in predictor aggregation
      timeStampToKeep = TimeStampToKeep.Min
    )
  )

  Spec[JoinedReader[_, _]] should "take any kind of reader as the leftmost input" in {
    profileReader.innerJoin(sparkReader) shouldBe a[JoinedDataReader[_, _]]
    dataReader.outerJoin(sparkReader) shouldBe a[JoinedDataReader[_, _]]
    passengerReader.leftOuterJoin(sparkReader) shouldBe a[JoinedDataReader[_, _]]

  }

  it should "allow simple readers for right inputs" in {
    sparkReader.innerJoin(profileReader).joinType shouldBe JoinTypes.Inner
    sparkReader.outerJoin(profileReader).joinType shouldBe JoinTypes.Outer
    sparkReader.leftOuterJoin(profileReader).joinType shouldBe JoinTypes.LeftOuter
  }

  it should "have all subreaders correctly ordered" in {
    val joinedReader = profileReader.innerJoin(sparkReader).outerJoin(dataReader)
    joinedReader.subReaders should contain theSameElementsAs Seq(profileReader, sparkReader, dataReader)
  }

  it should "correctly set leftKey in left outer and inner joins" in {
    dataReader.leftOuterJoin(sparkReader, joinKeys = JoinKeys(leftKey = "id")).joinKeys.leftKey shouldBe "id"
    dataReader.innerJoin(sparkReader, joinKeys = JoinKeys(leftKey = "id")).joinKeys.leftKey shouldBe "id"
  }

  it should "throw an error if you try to perform a self join" in {
    a[AssertionError] should be thrownBy {
      dataReader.innerJoin(dataReader)
    }
  }

  it should "throw an error if you try to use the same reader twice" in {
    a[AssertionError] should be thrownBy {
      dataReader.innerJoin(sparkReader).innerJoin(dataReader)
    }
  }

  it should "throw an error if you try to read the same data type twice with different readers" in {
    a[AssertionError] should be thrownBy {
      passengerReader.innerJoin(sparkReader).outerJoin(dataReader)
    }
  }

  it should "throw an error if you try to use an invalid key combination" in {
    a[RuntimeException] should be thrownBy {
      dataReader.innerJoin(sparkReader, joinKeys = JoinKeys(resultKey = DataFrameFieldNames.KeyFieldName))
        .generateDataFrame(Array.empty)
    }
  }

  it should "produce a JoinedAggregateDataReader when withSecondaryAggregation is called" in {
    val joinedReader = profileReader.innerJoin(sparkReader)
    val timeFilter = TimeBasedFilter(
      condition = new TimeColumn(boardedTime),
      primary = new TimeColumn(boardedTime),
      timeWindow = Duration.standardDays(DateTimeConstants.DAYS_PER_WEEK)
    )
    joinedReader.withSecondaryAggregation(timeFilter) shouldBe a[JoinedAggregateDataReader[_, _]]
  }

}
