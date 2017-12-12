package com.salesforce.hw.dataprep

import com.salesforce.op._
import com.salesforce.op.aggregators.{CutOffTime, SumRealNN}
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
 * ./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.dataprep.JoinsAndAggregates
 */

case class Click(clickId: Int, userId: Int, emailId: Int, timeStamp: String)
case class Send(sendId: Int, userId: Int, emailId: Int, timeStamp: String)

object JoinsAndAggregates {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("JoinsAndAggregates")
    implicit val spark = SparkSession.builder.config(conf).getOrCreate()
    import spark.implicits._

    val numClicksYday = FeatureBuilder.RealNN[Click]
      .extract(click => 1.toRealNN)
      .aggregate(SumRealNN)
      .window(Duration.standardDays(1))
      .asPredictor

    val numSendsLastWeek = FeatureBuilder.RealNN[Send]
      .extract(send => 1.toRealNN)
      .aggregate(SumRealNN)
      .window(Duration.standardDays(7))
      .asPredictor

    val numClicksTomorrow = FeatureBuilder.RealNN[Click]
      .extract(click => 1.toRealNN)
      .aggregate(SumRealNN)
      .window(Duration.standardDays(1))
      .asResponse

    // .alias ensures that the resulting dataframe column name is 'ctr'
    // and not the default transformed feature name
    val ctr = (numClicksYday / (numSendsLastWeek + 1)).alias

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd::HH:mm:ss")

    val clicksReader = DataReaders.Aggregate.csvCase[Click](
      path = Some("src/main/resources/EmailDataset/Clicks.csv"),
      key = _.userId.toString,
      aggregateParams = AggregateParams(
        timeStampFn = Some[Click => Long](c => formatter.parseDateTime(c.timeStamp).getMillis),
        cutOffTime = CutOffTime.DDMMYYYY("04092017")
      )
    )

    val sendsReader = DataReaders.Aggregate.csvCase[Send](
      path = Some("src/main/resources/EmailDataset/Sends.csv"),
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
