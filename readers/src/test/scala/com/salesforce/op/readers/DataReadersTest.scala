/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.OpParams
import com.salesforce.op.aggregators.CutOffTime
import com.salesforce.op.test._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


// scalastyle:off
@RunWith(classOf[JUnitRunner])
class DataReadersTest extends FlatSpec with PassengerSparkFixtureTest {

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
  // TODO: test the readers
}

