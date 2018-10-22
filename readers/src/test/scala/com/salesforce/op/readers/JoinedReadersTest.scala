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
    a[IllegalArgumentException] should be thrownBy {
      dataReader.innerJoin(dataReader)
    }
  }

  it should "throw an error if you try to use the same reader twice" in {
    a[IllegalArgumentException] should be thrownBy {
      dataReader.innerJoin(sparkReader).innerJoin(dataReader)
    }
  }

  it should "throw an error if you try to read the same data type twice with different readers" in {
    a[IllegalArgumentException] should be thrownBy {
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
