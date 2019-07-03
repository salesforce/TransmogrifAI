# Iris MultiClass Classification

The following code illustrates how TransmogrifAI can be used to do classify multiple classes over the Iris dataset. This example is very similar to the Titanic Binary Classification example, so you should look over that example first if you have not already. 
The code for this example can be found [here](https://github.com/salesforce/TransmogrifAI/tree/master/helloworld/src/main/scala/com/salesforce/hw/OpIrisSimple.scala), and the data over [here](https://github.com/salesforce/op/tree/master/helloworld/src/main/resources/IrisDataset/iris.csv).

**Data Schema**

```scala
case class Iris
(
  id: Int,
  sepalLength: Double,
  sepalWidth: Double,
  petalLength: Double,
  petalWidth: Double,
  irisClass: String
)
```

**Define Features**

```scala
val sepalLength = FeatureBuilder.Real[Iris].extract(_.getSepalLength.toReal).asPredictor
val sepalWidth = FeatureBuilder.Real[Iris].extract(_.getSepalWidth.toReal).asPredictor
val petalLength = FeatureBuilder.Real[Iris].extract(_.getPetalLength.toReal).asPredictor
val petalWidth = FeatureBuilder.Real[Iris].extract(_.getPetalWidth.toReal).asPredictor
val irisClass = FeatureBuilder.Text[Iris].extract(_.getClass$.toText).asResponse
```

**Feature Engineering**

```scala
val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()
val label = irisClass.indexed()
val checkedFeatures = label.sanityCheck(features, removeBadFeatures = true)
```

**Modeling & Evaluation**

In MultiClass Classification, we use the ```MultiClassificationModelSelector``` to select the model we want to run on, which is Logistic Regression in this case. You can find more information on model selection [here](../developer-guide#modelselector).

```scala
val prediction = MultiClassificationModelSelector
  .withTrainValidationSplit(
    modelTypesToUse = Seq(OpLogisticRegression))
  .setInput(label, checkedFeatures).getOutput()

val evaluator = Evaluators.MultiClassification()
  .setLabelCol(label)
  .setPredictionCol(prediction)

val workflow = new OpWorkflow().setResultFeatures(prediction, label).setReader(dataReader)

val model = workflow.train()
```

**Results**

We can still find the contributions of each feature for the model, but in MultiClass Classification, ```ModelInsights``` has a contribution of each feature to the prediction of each class. This code takes the max of all of these contributions as the overall contribution.

```scala
val modelInsights = model.modelInsights(prediction)
val modelFeatures = modelInsights.features.flatMap( feature => feature.derivedFeatures)
val featureContributions = modelFeatures.map( feature => (feature.derivedFeatureName,
  feature.contribution.map( contribution => math.abs(contribution))
    .foldLeft(0.0) { (max, contribution) => math.max(max, contribution)}))
val sortedContributions = featureContributions.sortBy( contribution => -contribution._2)
    
val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)
```

You can run the code using the following command:
```bash
cd helloworld
./gradlew compileTestScala installDist
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.OpIrisSimple -Dargs="\
`pwd`/src/main/resources/IrisDataset/iris.csv"
```
