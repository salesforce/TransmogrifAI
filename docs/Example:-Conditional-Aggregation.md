In this example, we demonstrate use of TransmogrifAI's conditional readers to, once again, simplify complex data preparation. Code for this example can be found [here](https://github.com/salesforce/TransmogrifAI/tree/master/helloworld/src/main/scala/com/salesforce/hw/dataprep/ConditionalAggregation.scala), and the data can be found [here](https://github.com/salesforce/op/tree/master/helloworld/src/main/resources/WebVisitsDataset/WebVisits.csv).

In the previous [example](Example%3A-Time-Series-Aggregates-and-Joins), we showed how TransmogrifAI FeatureBuilders and Aggregate Readers could be used to aggregate predictors and response variables with respect to a reference point in time. However, sometimes, aggregations need to be computed with respect to the time of occurrence of a particular event, and this time may vary from key to key. In particular, let's consider a situation where we are analyzing website visit data, and would like to build a model that predicts the number of purchases a user makes  on the website within a day of visiting a particular landing page. In this scenario, we need to construct a training dataset that for each user, identifies the time when he visited the landing page, and then creates a response which is the number of times the user made a purchase within a day of that time. The predictors for the user would be aggregated from the web visit behavior of the user up unto that point in time.

Let's start once again by looking at the reader. The web visit data is described by the following case class:

```scala
case class WebVisit(
    userId: String,
    url: String,
    productId: Option[Int],
    price: Option[Double],
    timestamp: String
)
``` 

We read this data using a Conditional Aggregate Reader:

```scala
val visitsReader = DataReaders.Conditional.csvCase[WebVisit](
    path = Some("src/main/resources/WebVisitsDataset/WebVisits.csv"),
    key = _.userId,
    conditionalParams = ConditionalParams(
        timeStampFn = visit => formatter.parseDateTime(visit.timestamp).getMillis,
        targetCondition = _.url == "http://www.amazon.com/SaveBig",
        dropIfTargetConditionNotMet = true
    )
)
```

Once again, there are a few different parameters of note in this reader. 
* The ```key``` specifies the key in the table that should be used to aggregate the predictors or response variables
* The ```targetCondition``` specifies the function to be applied to a record to see if the target condition is met. In this case, the event of interest is whether the user visited the Amazon Save Big landing page. 
* The ```timeStampFn``` provides the function to be applied to a record to extract its timestamp and compare to the timestamp of the target event. 
*  ```dropIfTargetConditionNotMet``` when set to ```true``` drops all keys where the target condition was not met.

The predictor and response variables are specified as before:


```scala
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
```

And finally, the predictors, response variables, and readers are all fed to a workflow and the training dataset is materialized:

```scala
val workflowModel = new OpWorkflow()
    .setReader(visitsReader)
    .setResultFeatures(numVisitsWeekPrior, numPurchasesNextDay)
    .train()

val dataFrame = workflowModel.score()
```

The TransmogrifAI workflow automatically identifies when the target condition was met for each key in the table, and aggregates the predictor and response variables for each appropriately:

```scala
dataFrame.show()

+------------------+-------------------+------------------+
|               key|numPurchasesNextDay|numVisitsWeekPrior|
+------------------+-------------------+------------------+
|xyz@salesforce.com|                1.0|               3.0|
|lmn@salesforce.com|                1.0|               0.0|
|abc@salesforce.com|                0.0|               1.0|
+------------------+-------------------+------------------+
``` 
