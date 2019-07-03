# Titanic Binary Classification

Here we describe a very simple TransmogrifAI workflow for predicting survivors in the often-cited Titanic dataset. The code for building and applying the Titanic model can be found [here](https://github.com/salesforce/TransmogrifAI/blob/master/helloworld/src/main/scala/com/salesforce/hw/OpTitanicSimple.scala), and the data can be found [here](https://github.com/salesforce/op/blob/master/helloworld/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv).

You can run this code as follows:
```bash
cd helloworld
./gradlew compileTestScala installDist
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.OpTitanicSimple -Dargs="\
`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv"
```

Let's break down what's happening. The code starts off by describing the schema of the data via a case class:

```scala
// Passenger data schema
case class Passenger(
  id: Int,
  survived: Int,
  pClass: Option[Int],
  name: Option[String],
  sex: Option[String],
  age: Option[Double],
  sibSp: Option[Int],
  parCh: Option[Int],
  ticket: Option[String],
  fare: Option[Double],
  cabin: Option[String],
  embarked: Option[String]
)
```
In the main function, we create a spark session as per usual:

```scala
// Set up a SparkSession as normal
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
import spark.implicits._ // Needed for Encoders for the Passenger case class
```

We then define the set of raw features that we would like to extract from the data. The raw features are defined using [FeatureBuilders](../developer-guide#featurebuilders), and are strongly typed. TransmogrifAI supports the following basic feature types: Text, Numeric, Vector, List , Set, Map. In addition it supports many specific feature types which extend these base types: Email extends Text; Integral, Real and Binary extend Numeric; Currency and Percentage extend Real. For a complete view of the types supported see the [Type Hierarchy and Automatic Feature Engineering](../developer-guide#type-hierarchy-and-automatic-feature-engineering) section in the Documentation.

Basic FeatureBuilders will be created for you if you use the TransmogrifAI CLI to bootstrap your project as described [here](../examples/Bootstrap-Your-First-Project.html). However, it is often useful to edit this code to customize feature generation and take full advantage of the Feature types available (selecting the appropriate type will improve automatic feature engineering steps).

When defining raw features, specify the extract logic to be applied to the raw data, and  also  annotate the features as either predictor or response variables via the FeatureBuilders:
```scala
// import necessary packages
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op._

// Define features using the TransmogrifAI types based on the data
val survived = FeatureBuilder.RealNN[Passenger].extract(_.survived.toRealNN).asResponse

val pClass = FeatureBuilder.PickList[Passenger].extract(_.pClass.map(_.toString).toPickList).asPredictor

val name = FeatureBuilder.Text[Passenger].extract(_.name.toText).asPredictor

val sex = FeatureBuilder.PickList[Passenger].extract(_.sex.map(_.toString).toPickList).asPredictor

val age = FeatureBuilder.Real[Passenger].extract(_.age.toReal).asPredictor

val sibSp = FeatureBuilder.Integral[Passenger].extract(_.sibSp.toIntegral).asPredictor

val parCh = FeatureBuilder.Integral[Passenger].extract(_.parCh.toIntegral).asPredictor

val ticket = FeatureBuilder.PickList[Passenger].extract(_.ticket.map(_.toString).toPickList).asPredictor

val fare = FeatureBuilder.Real[Passenger].extract(_.fare.toReal).asPredictor

val cabin = FeatureBuilder.PickList[Passenger].extract(_.cabin.map(_.toString).toPickList).asPredictor

val embarked = FeatureBuilder.PickList[Passenger].extract(_.embarked.map(_.toString).toPickList).asPredictor
```

Now that the raw features have been defined, we go ahead and define how we would like to manipulate them via Stages ([Transformers](../developer-guide#transformers) and [Estimators](../developer-guide#estimators)). A TransmogrifAI Stage takes one or more Features, and returns a new Feature. TransmogrifAI provides numerous handy short cuts for specifying common feature manipulations. For basic arithmetic operations, you can just use "+", "-", "*" and "/". In addition, shortcuts like "normalize", "pivot" and "map" are also available.

```scala
val familySize = sibSp + parCh + 1
val estimatedCostOfTickets = familySize * fare

// normalize the numeric feature age to create a new transformed feature
val normedAge = age.zNormalize()

// pivot the categorical feature, sex, into a 0-1 vector of (male, female) 
val pivotedSex = sex.pivot() 

// divide age into adult and child
val ageGroup = age.map[PickList](_.value.map(v => if (v > 18) "adult" else "child").toPickList)
```

The above notation is short-hand for the following, more formulaic way of invoking TransmogrifAI Stages:

```scala
val normedAge: FeatureLike[Numeric] = new NormalizeEstimator().setInput(age).getOutput
val pivotedSex: FeatureLike[Vector] = new PivotEstimator().setInput(sex).getOutput
```
See [“Creating Shortcuts for Transformers and Estimators”](../developer-guide#creating-shortcuts-for-transformers-and-estimators) for more documentation on how shortcuts for stages can be created.
We now define a Feature of type Vector, that is a vector representation of all the features we would like to use as predictors in our workflow.

```scala
val passengerFeatures: FeatureLike[OPVector] = Seq(
   pClass, name, sex, age, sibSp, parCh, ticket,
   cabin, embarked, familySize, estimatedCostOfTickets, normedAge,
   pivotedSex, ageGroup
).transmogrify()
```

The ```.transmogrify()``` shortcut is a special AutoML Estimator that applies a default set of transformations to all the specified inputs and combines them into a single vector. This is in essence the [automatic feature engineering Stage](../automl-capabilities#vectorizers-and-transmogrification) of TransmogrifAI. This stage can be discarded in favor of hand-tuned feature engineering and manual vector creation followed by combination using the VectorsCombiner Transformer (short-hand ```Seq(....).combine()```) if the user desires to have complete control over feature engineering.

The next stage applies another powerful AutoML Estimator — the [SanityChecker](../automl-capabilities#sanitychecker). The SanityChecker applies a variety of statistical tests to the data based on Feature types and discards predictors that are indicative of label leakage or that show little to no predictive power. This is in essence the automatic feature selection Stage of TransmogrifAI:

```scala
// Optionally check the features with a sanity checker
val checkedFeatures = survived.sanityCheck(passengerFeatures, removeBadFeatures = true)
```
Finally, the OpLogisticRegression Estimator is applied to derive a new triplet of Features which are essentially probabilities and predictions returned by the logistic regression algorithm:

```scala
// Define the model we want to use (here a simple logistic regression) and get the resulting output
import com.salesforce.op.stages.impl.classification.OpLogisticRegression

// Define the model we want to use (here a simple logistic regression) and get the resulting output
val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
  modelTypesToUse = Seq(OpLogisticRegression)
).setInput(survived, checkedFeatures).getOutput()
```
We could alternatively have used the [ModelSelector](../automl-capabilities#modelselectors) — another powerful AutoML Estimator that automatically tries out a variety of different classification algorithms and then selects the best one.

Notice that everything we've done so far has been purely at the level of definitions. We have defined how we would like to extract our raw features from data of type 'Passenger', and we have defined how we would like to manipulate them. In order to actually manifest the data described by these features, we need to add them to a workflow and attach a data source to the workflow:

```scala
import com.salesforce.op.readers.DataReaders

val dataReader = DataReaders.Simple.csvCase[Passenger](path = Option(csvFilePath), key = _.id.toString)   

val workflow =
   new OpWorkflow()
      .setResultFeatures(survived, prediction)
      .setReader(dataReader)
```

When we now call 'train' on this workflow, it automatically computes and executes the entire DAG of Stages needed to compute the features ```survived, prediction, rawPrediction```, and ```prob```, fitting all the estimators on the training data in the process. Calling ```scoreAndEvaluate``` on the model then transforms the underlying training data to produce a DataFrame with the all the features manifested. The ```scoreAndEvaluate``` method can optionally be passed an evaluator that produces metrics. 

```scala
import com.salesforce.op.evaluators.Evaluators

// Fit the workflow to the data
val model = workflow.train()

val evaluator = Evaluators.BinaryClassification()
   .setLabelCol(survived)
   .setPredictionCol(prediction)

// Apply the fitted workflow to the train data and manifest
// the resulting dataframe together with metrics
val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)
```

We can find information about the model via ```ModelInsights```. In this example, we extracting each feature's contribution to the model by first listing all of the features that the model derives from the input features and finding their maximum contribution. We can also find the correlation of the feature with the label, mean of the feature values, variance of the values, etc... More information about ```ModelInsights``` can be found [here](../developer-guide#extracting-modelinsights-from-a-fitted-workflow).
```scala
// Extract information (i.e. feature importance) via model insights
val modelInsights = model.modelInsights(prediction)
val modelFeatures = modelInsights.features.flatMap( feature => feature.derivedFeatures)
val featureContributions = modelFeatures.map( feature => (feature.derivedFeatureName,
  feature.contribution.map( contribution => math.abs(contribution))
    .foldLeft(0.0) { (max, contribution) => math.max(max, contribution)}))
val sortedContributions = featureContributions.sortBy( contribution => -contribution._2)
```



