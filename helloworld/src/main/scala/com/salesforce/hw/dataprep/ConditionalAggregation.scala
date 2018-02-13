/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.dataprep

import com.salesforce.op.OpWorkflow
import com.salesforce.op.aggregators.SumRealNN
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{ConditionalParams, DataReaders}
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.joda.time.Duration
import org.joda.time.format.DateTimeFormat

/**
 * In this example, we demonstrate use of OP's conditional readers to, once again, simplify complex data preparation.
 * Conditional readers should be used to prep data when computing conditional probabilities.
 *
 * In the example below, we have a collection of web visit data, and would like to predict the likelihood of a user
 * to make a purchase on the website within a day of visiting a particular landing page.
 *
 * We specify the landing page in the targetCondition of the conditional reader. All predictors and responses
 * will now be aggregated with respect to the time when this condition is met for each user. If the condition
 * is never met, we simply drop the corresponding user from the prepared dataset.
 *
 * This is how you run this example from your command line:
 * ./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.dataprep.ConditionalAggregation -Dargs="\
 * `pwd`src/main/resources/WebVisitsDataset/WebVisits.csv"
 */

case class WebVisit(userId: String, url: String, productId: Option[Int], price: Option[Double], timestamp: String)

object ConditionalAggregation {

  def main(args: Array[String]): Unit = {

    if (args.length != 1) throw new IllegalArgumentException("Full path to WebVisit dataset was not provided")

    val conf = new SparkConf().setAppName("ConditionalAggregation")
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    val numVisitsWeekPrior = FeatureBuilder.RealNN[WebVisit]
      .extract(visit => 1.toRealNN)
      .aggregate(SumRealNN)
      .window(Duration.standardDays(7))
      .asPredictor

    val numPurchasesNextDay = FeatureBuilder.RealNN[WebVisit]
      .extract(visit => visit.productId.map{_ => 1D}.toRealNN)
      .aggregate(SumRealNN)
      .window(Duration.standardDays(1))
      .asResponse

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd::HH:mm:ss")

    val visitsReader = DataReaders.Conditional.csvCase[WebVisit](
      path = Option(args(0)),
      key = _.userId,
      conditionalParams = ConditionalParams(
        timeStampFn = visit => formatter.parseDateTime(visit.timestamp).getMillis,
        targetCondition = _.url == "http://www.amazon.com/SaveBig",
        responseWindow = Some(Duration.standardDays(1)),
        dropIfTargetConditionNotMet = true
      )
    )

    val workflowModel = new OpWorkflow()
      .setReader(visitsReader)
      .setResultFeatures(numVisitsWeekPrior, numPurchasesNextDay)
      .train()

    val dataFrame = workflowModel.score()

    dataFrame.show()

    /* Expected Output
    +------------------+-------------------+------------------+
    |               key|numPurchasesNextDay|numVisitsWeekPrior|
    +------------------+-------------------+------------------+
    |xyz@salesforce.com|                1.0|               3.0|
    |lmn@salesforce.com|                1.0|               0.0|
    |abc@salesforce.com|                0.0|               1.0|
    +------------------+-------------------+------------------+
     */
  }

}
