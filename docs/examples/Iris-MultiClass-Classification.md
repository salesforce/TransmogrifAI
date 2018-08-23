# Iris MultiClass Classification

The following code illustrates how TransmogrifAI can be used to do classify multiple classes over the Iris dataset.
The code for this example can be found [here](https://github.com/salesforce/TransmogrifAI/tree/master/helloworld/src/main/scala/com/salesforce/hw/iris), and the data over [here](https://github.com/salesforce/op/tree/master/helloworld/src/main/resources/IrisDataset).

**Define features**
```scala
val id = FeatureBuilder.Integral[Iris].extract(_.getID.toIntegral).asPredictor
val sepalLength = FeatureBuilder.Real[Iris].extract(_.getSepalLength.toReal).asPredictor
val sepalWidth = FeatureBuilder.Real[Iris].extract(_.getSepalWidth.toReal).asPredictor
val petalLength = FeatureBuilder.Real[Iris].extract(_.getPetalLength.toReal).asPredictor
val petalWidth = FeatureBuilder.Real[Iris].extract(_.getPetalWidth.toReal).asPredictor
val irisClass = FeatureBuilder.Text[Iris].extract(_.getClass$.toText).asResponse

```
**Feature Engineering**

```scala
val labels = irisClass.indexed()
val features = Seq(sepalLength, sepalWidth, petalLength, petalWidth).transmogrify()
```
**Modeling & Evaluation**
```scala
val pred = MultiClassificationModelSelector
  .withCrossValidation(splitter = Some(DataCutter(reserveTestFraction = 0.2, seed = randomSeed)), seed = randomSeed)
  .setInput(labels, features).getOutput()

private val evaluator = Evaluators.MultiClassification.f1()
  .setLabelCol(labels)
  .setPredictionCol(pred)

private val wf = new OpWorkflow().setResultFeatures(pred, labels)

def runner(opParams: OpParams): OpWorkflowRunner =
  new OpWorkflowRunner(
    workflow = wf,
    trainingReader = irisReader,
    scoringReader = irisReader,
    evaluationReader = Option(irisReader),
    evaluator = Option(evaluator),
    featureToComputeUpTo = Option(features)
  )
```
You can run the code using the following commands for train, score and evaluate:
```bash
cd helloworld
./gradlew compileTestScala installDist
```
**Train**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=train \
--model-location=/tmp/iris-model \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/iris.data"
```
**Score**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=score \
--model-location=/tmp/iris-model \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/bezdekIris.data \
--write-location=/tmp/iris-scores"
```
**Evaluate**
```bash
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=evaluate \
--model-location=/tmp/iris-model \
--metrics-location=/tmp/iris-metrics \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/bezdekIris.data \
--write-location=/tmp/iris-eval"
```
