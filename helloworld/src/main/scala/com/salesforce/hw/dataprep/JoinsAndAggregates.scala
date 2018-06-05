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

package com.salesforce.hw.dataprep

import com.salesforce.op._
import com.salesforce.op.aggregators.{CutOffTime, SumRealNN, SumReal}
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{AggregateParams, DataReaders}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.joda.time.Duration
import org.joda.time.format.DateTimeFormat


/**
 * In this example, we will use OP's aggregate and join readers to specify fairly complex data preparation with
 * just a few lines of code. The data used in this example are two tables of "Email Sends" and "Email Clicks".
 * We would like to assemble a training data set where the predictors are features like the number of clicks in
 * the past day and the CTR in the past week. And the response variable is the number of clicks the next day.
 *
 * The ClicksReader in this example is an aggregate reader, which means that any feature computed on the clicks
 * table will be aggregated by the specified key. Predictors will be aggregated up until the cutOffTime, 09/04/2017,
 * response variables will be aggregated after the cutOffTime.
 *
 * Further, by using the joint reader, null values will automatically be handled for features like CTR that are
 * obtained by joining the two tables.
 *
 * This is how you run this example from your command line:
 * ./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.dataprep.JoinsAndAggregates -Dargs="\
 * `pwd`/src/main/resources/EmailDataset/Clicks.csv `pwd`/src/main/resources/EmailDataset/Sends.csv"
 */


case class Click(clickId: Int, userId: Int, emailId: Int, timeStamp: String)
case class Send(sendId: Int, userId: Int, emailId: Int, timeStamp: String)

object JoinsAndAggregates {

  def main(args: Array[String]): Unit = {

    if (args.length != 2) throw new IllegalArgumentException("Full paths to Click and Send datasets were not provided")

    val conf = new SparkConf().setAppName("JoinsAndAggregates")
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    val numClicksYday = FeatureBuilder.Real[Click]
      .extract(click => 1.toReal)
      .aggregate(SumReal)
      .window(Duration.standardDays(1))
      .asPredictor

    val numSendsLastWeek = FeatureBuilder.Real[Send]
      .extract(send => 1.toReal)
      .aggregate(SumReal)
      .window(Duration.standardDays(7))
      .asPredictor

    val numClicksTomorrow = FeatureBuilder.Real[Click]
      .extract(click => 1.toReal)
      .aggregate(SumReal)
      .window(Duration.standardDays(1))
      .asResponse

    // .alias ensures that the resulting dataframe column name is 'ctr'
    // and not the default transformed feature name
    val ctr = (numClicksYday / (numSendsLastWeek + 1)).alias

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd::HH:mm:ss")

    val clicksReader = DataReaders.Aggregate.csvCase[Click](
      path = Option(args(0)),
      key = _.userId.toString,
      aggregateParams = AggregateParams(
        timeStampFn = Some[Click => Long](c => formatter.parseDateTime(c.timeStamp).getMillis),
        cutOffTime = CutOffTime.DDMMYYYY("04092017")
      )
    )

    val sendsReader = DataReaders.Aggregate.csvCase[Send](
      path = Option(args(1)),
      key = _.userId.toString,
      aggregateParams = AggregateParams(
        timeStampFn = Some[Send => Long](s => formatter.parseDateTime(s.timeStamp).getMillis),
        cutOffTime = CutOffTime.DDMMYYYY("04092017")
      )
    )

    val workflowModel = new OpWorkflow()
      .setReader(sendsReader.leftOuterJoin(clicksReader))
      .setResultFeatures(numClicksYday, numClicksTomorrow, numSendsLastWeek, ctr)
      .train()

    val dataFrame = workflowModel.score()

    dataFrame.show()

    /* Expected Output
    +---+---+-----------------+-------------+----------------+
    |ctr|key|numClicksTomorrow|numClicksYday|numSendsLastWeek|
    +---+---+-----------------+-------------+----------------+
    |0.0|789|             null|         null|             1.0|
    |0.0|456|              1.0|          0.0|             0.0|
    |1.0|123|              1.0|          2.0|             1.0|
    +---+---+-----------------+-------------+----------------+
     */
  }

}
