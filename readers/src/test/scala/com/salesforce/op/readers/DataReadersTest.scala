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

import com.salesforce.op.OpParams
import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.test._
import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner



@RunWith(classOf[JUnitRunner])
class DataReadersTest extends FlatSpec with PassengerSparkFixtureTest {

  // scalastyle:off
  Spec(DataReaders.getClass) should "define readers" in {
    /* Set type of extraction with data reader definition */
    val simpleAvroReader: DataReader[Passenger] = DataReaders.Simple.avro[Passenger](
      path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
      key = _.getPassengerId.toString // Entity to score
    )

    val simpleCustomReader: DataReader[Passenger] = new CustomReader[Passenger](_.getPassengerId.toString) {
      def readFn(params: OpParams)
        (implicit spark: SparkSession): Either[RDD[Passenger], Dataset[Passenger]] = {
        // TODO: implement some custom reader
        null
      }
    }

    /* Set type of extraction with data reader definition.
     * When using aggregate need to define the cutoff information
     * for features versus label */
    val dataSourceAggregate: DataReader[Passenger] = DataReaders.Aggregate.avro[Passenger](
      path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
      key = _.getPassengerId.toString, // Entity to score
      aggregateParams = AggregateParams(
        timeStampFn = Option(_.getRecordDate.toLong), // Record field to use as timestamp for label versus feature aggregation
        cutOffTime = CutOffTime.UnixEpoch(1471046600) // Type of cutoff defined "DaysAgo", "UnixEpoch" etc.
      )
    )

    /* Set type of extraction with data reader definition.
    * When using conditional need to include the target info,
    * response window, predictor window, */
    val dataSourceConditional: DataReader[Passenger] = DataReaders.Conditional.avro[Passenger](
      path = Some(passengerAvroPath), // Path should be optional so can also pass in as a parameter
      key = _.getPassengerId.toString, // Entity to score
      conditionalParams = ConditionalParams(
        timeStampFn = _.getRecordDate.toLong, // Record field which defines the date for the rest of the columns
        targetCondition = _.getBoarded >= 1471046600, // Function to figure out if target event has occurred
        responseWindow = Some(Duration.standardDays(7)), // How many days after target event to include in response aggregation
        predictorWindow = Some(Duration.standardDays(7)), // How many days before target event to include in predictor aggregation
        timeStampToKeep = TimeStampToKeep.Min
      )
    )

  }
  // scalastyle:on

  Spec[ConditionalParams[_]] should "validate if user functions are serializable" in {
    class NotSerializable(val v: Double)
    val ns = new NotSerializable(1.0)

    assertTaskNotSerializable(ConditionalParams[Passenger](
      timeStampFn = _ => ns.v.toLong, targetCondition = _.getBoarded >= 1471046600
    ))
    assertTaskNotSerializable(ConditionalParams[Passenger](
      timeStampFn = _.getRecordDate.toLong, targetCondition = _ => ns.v > 1
    ))
    assertTaskNotSerializable(ConditionalParams[Passenger](
      timeStampFn = _.getRecordDate.toLong, targetCondition = _.getBoarded >= 1471046600,
      cutOffTimeFn = Some((_, _) => CutOffTime.UnixEpoch(ns.v.toLong))
    ))

    def assertTaskNotSerializable(v: => Any): Unit = {
      val error = intercept[IllegalArgumentException](v)
      error.getMessage shouldBe "Function is not serializable"
      error.getCause shouldBe a[SparkException]
    }

  }

  // TODO: test the readers
}

