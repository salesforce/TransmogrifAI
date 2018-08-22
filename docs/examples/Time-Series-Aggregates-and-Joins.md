# Time Series Aggregates and Joins

In this example, we will walk you through some of the powerful tools TransmogrifAI has for data preparation, in particular for time series aggregates and joins. The code for this example can be found [here](https://github.com/salesforce/TransmogrifAI/tree/master/helloworld/src/main/scala/com/salesforce/hw/dataprep/JoinsAndAggregates.scala), and the data over [here](https://github.com/salesforce/op/tree/master/helloworld/src/main/resources/EmailDataset). 

In this example, we would like to build a training data set from two different tables -- a table of Email Sends, and a table of Email Clicks. The following case classes describe the schemas of the two tables:

```scala
case class Click(clickId: Int, userId: Int, emailId: Int, timeStamp: String)
case class Send(sendId: Int, userId: Int, emailId: Int, timeStamp: String)
```

The goal is to build a model that will predict the number of times a user will click on emails on day ```x+1```, given his click behavior in the lead-up to day ```x```. The ideal training dataset would be constructed by taking a certain point in time as a reference point. And then for every user in the tables, computing a response that is the number of times the user clicked on an email within a day of that reference point. The features for every user would be computed by aggregating his click behavior up until that reference point. 

Unlike the previous examples, these tables represent events -- a single user may have been sent multiple emails, or clicked on multiple emails, and the events need to be aggregated in order to produce meaningful predictors and response variables for a training data set. 

TransmogrifAI provides an easy way for us to define these aggregate features. Using a combination of FeatureBuilders and Aggregate Readers. Let's start with the readers. We define two readers for the two different tables as follows:

```scala
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
```

There are a few different parameters of interest in these readers: 
* The first is a ```key``` parameter, that specifies the key in the table that should be used to aggregate either the predictors or response variables. 
* The second is a ```timeStampFn``` parameter that allows the user to specify a function for extracting timestamps from records in the table. This is the timestamp that will be used to compare against the reference time. 
* And the third is a ```cutOffTime```, which is the reference time to be used.
All predictors will be aggregated from records up until the ```cutOffTime```, and all response variables will be aggregated from records following the ```cutOffTime```.

Now let's look at how the predictors and response variables are defined. In this example, we define two aggregate predictors using TransmogrifAI's FeatureBuilders:

```scala    
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
```
Here ```numClicksYday``` is a non-nullable real predictor, extracted from the Clicks table, by mapping each click to a ```1```, then aggregating for each key of the Click table by summing up the ```1's``` that occur in a 1 day window before the ```cutOffTime``` specified in the ```clicksReader```. 

Similarly, ```numSendsLastWeek``` is obtained by aggregating for each key of the Send table, all the sends that occur in a 7 day windown prior to the ```cutOffTime``` specified in the ```sendsReader```.

The response variable on the other hand, is obtained by aggregating all the clicks that occur in a 1 day window following the ```cutOffTime``` specified in the ```clicksReader```:

```scala
val numClicksTomorrow = FeatureBuilder.RealNN[Click]
    .extract(click => 1.toRealNN)
    .aggregate(SumRealNN)
    .window(Duration.standardDays(1))
    .asResponse
```

Now we can also create a predictor from the combination of the clicks and sends predictors as follows:

```scala
// .alias ensures that the resulting dataframe column name is 'ctr'
// and not the default transformed feature name
val ctr = (numClicksYday / (numSendsLastWeek + 1)).alias
```

In order to materialize all of these predictors and response variables, we can add them to a workflow with the appropriate readers:

```scala
// fit the workflow to the data
val workflowModel = new OpWorkflow()
    .setReader(sendsReader.leftOuterJoin(clicksReader))
    .setResultFeatures(numClicksYday, numClicksTomorrow, numSendsLastWeek, ctr)
    .train()

// materialize the features
val dataFrame = workflowModel.score()
```

Note that the reader for the workflow is a joined reader, obtained by joining the ```sendsReader``` with the ```clicksReader```. The joined reader deals with nulls in the two tables appropriately:

```scala
dataFrame.show()

+---+---+-----------------+-------------+----------------+
|ctr|key|numClicksTomorrow|numClicksYday|numSendsLastWeek|
+---+---+-----------------+-------------+----------------+
|0.0|789|             null|         null|             1.0|
|0.0|456|              1.0|          0.0|             0.0|
|1.0|123|              1.0|          2.0|             1.0|
+---+---+-----------------+-------------+----------------+
```
