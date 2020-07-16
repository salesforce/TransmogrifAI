# Boston Regression

The following code illustrates how TransmogrifAI can be used to do linear regression. We use Boston dataset to predict housing prices. This example is very similar to the Titanic Binary Classification example, so you should look over that example first if you have not already.
The code for this example can be found [here](https://github.com/salesforce/TransmogrifAI/tree/master/helloworld/src/main/scala/com/salesforce/hw/boston), and the data over [here](https://github.com/salesforce/op/tree/master/helloworld/src/main/resources/BostonDataset).

**Define features**
```scala
val rowId = FeatureBuilder.Integral[BostonHouse].extract(_.rowId.toIntegral).asPredictor
val crim = FeatureBuilder.RealNN[BostonHouse].extract(_.crim.toRealNN).asPredictor
val zn = FeatureBuilder.RealNN[BostonHouse].extract(_.zn.toRealNN).asPredictor
val indus = FeatureBuilder.RealNN[BostonHouse].extract(_.indus.toRealNN).asPredictor
val chas = FeatureBuilder.PickList[BostonHouse].extract(x => Option(x.chas).toPickList).asPredictor
val nox = FeatureBuilder.RealNN[BostonHouse].extract(_.nox.toRealNN).asPredictor
val rm = FeatureBuilder.RealNN[BostonHouse].extract(_.rm.toRealNN).asPredictor
val age = FeatureBuilder.RealNN[BostonHouse].extract(_.age.toRealNN).asPredictor
val dis = FeatureBuilder.RealNN[BostonHouse].extract(_.dis.toRealNN).asPredictor
val rad = FeatureBuilder.Integral[BostonHouse].extract(_.rad.toIntegral).asPredictor
val tax = FeatureBuilder.RealNN[BostonHouse].extract(_.tax.toRealNN).asPredictor
val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(_.ptratio.toRealNN).asPredictor
val b = FeatureBuilder.RealNN[BostonHouse].extract(_.aa.toRealNN).asPredictor
val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor
val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.toRealNN).asResponse

```
**Feature Engineering**

```scala
val features = Seq(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat).transmogrify()
val label = medv
val checkedFeatures = label.sanityCheck(features, removeBadFeatures = true)
```

**Modeling & Evaluation**

For regression problems, we use ```RegressionModelSelector``` to choose which type of models to use, which in this case is Linear Regression. You can find more model types [here](../developer-guide#modelselector).

```scala
val prediction = RegressionModelSelector
  .withTrainValidationSplit(
    modelTypesToUse = Seq(OpLinearRegression))
  .setInput(label, checkedFeatures).getOutput()

val workflow = new OpWorkflow().setResultFeatures(prediction)

val evaluator = Evaluators.Regression().setLabelCol(label).setPredictionCol(prediction)

val model = workflow.train()
```

**Results**

We can extract each feature's contribution to the model via ```ModelInsights```, like in the Titanic Binary Classification example.

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
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.OpBostonSimple -Dargs="\
`pwd`/src/main/resources/BostonDataset/housingData.csv"
```
