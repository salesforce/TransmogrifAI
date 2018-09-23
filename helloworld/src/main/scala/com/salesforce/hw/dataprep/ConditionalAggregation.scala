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
      .extract(visit => visit.productId.map(_ => 1.0).toRealNN(0.0))
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
