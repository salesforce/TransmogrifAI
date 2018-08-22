The following code illustrates how TransmogrifAI can be used to do linear regression. We use Boston dataset to predict housing prices.
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
val b = FeatureBuilder.RealNN[BostonHouse].extract(_.b.toRealNN).asPredictor
val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor
val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.toRealNN).asResponse

```
**Feature Engineering**

```scala
val houseFeatures = Seq(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat).transmogrify()
```
**Modeling & Evaluation**
```scala
val prediction = RegressionModelSelector
  .withCrossValidation(dataSplitter = Option(DataSplitter(seed = randomSeed)), seed = randomSeed)
  .setRandomForestSeed(randomSeed)
  .setGradientBoostedTreeSeed(randomSeed)
  .setInput(medv, houseFeatures)
  .getOutput()

val workflow = new OpWorkflow().setResultFeatures(prediction)

val evaluator = Evaluators.Regression().setLabelCol(medv).setPredictionCol(prediction)

def runner(opParams: OpParams): OpWorkflowRunner =
  new OpWorkflowRunner(
    workflow = workflow,
    trainingReader = trainingReader,
    scoringReader = scoringReader,
    evaluationReader = Option(trainingReader),
    evaluator = Option(evaluator),
    scoringEvaluator = None,
    featureToComputeUpTo = Option(houseFeatures)
  )
```
You can run the code using the following commands for train, score and evaluate:
```bash
cd helloworld
./gradlew compileTestScala installDist
```
**Train**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=train \
--model-location=/tmp/boston-model \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data"
```
**Score**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=score \
--model-location=/tmp/boston-model \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data \
--write-location=/tmp/boston-scores"
```
**Evaluate**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=evaluate \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data \
--write-location=/tmp/boston-eval \
--model-location=/tmp/boston-model \
--metrics-location=/tmp/boston-metrics"
```
