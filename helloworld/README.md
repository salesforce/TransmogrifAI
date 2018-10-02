# Hello World project for TransmogrifAI apps

There are four example workflows in this project:
1) **(start here)** A minimal classifier based on the Titanic dataset `com.salesforce.hw.titanic.OpTitanicMini`
2) A simple classifier based on the Titanic dataset with examples of overriding default reader behavior  - `com.salesforce.hw.titanic.OpTitanic`
3) A simple classifier for multiclass labels on the Iris dataset - `com.salesforce.hw.iris.OpIris`
4) A simple regression based on boston housing data - `com.salesforce.hw.boston.OpBoston`

In addition, there are two examples of more complex kinds of data preparation that can be done using OP Readers and FeatureBuilders:
1) An example that computes time series aggregations and joins `com.salesforce.hw.dataprep.JoinsAndAggregates`
2) An example that computes conditional aggregations `com.salesforce.hw.dataprep.ConditionalAggregation`

Each project can be either be run with the gradle task, `sparkSubmit` (**recommended**) or with the standard `spark-submit` command. We show examples of running the Titanic case with both gradle and spark-submit for completeness, but the rest of the instructions are for gradle only since that is the recommended submission method (it defines many other useful spark parameters). You should not mix submission methods (eg. don't train with the gradle task and score with spark-submit), as you may get class serialization errors.

Note: make sure you have all the [prerequisites](http://docs.transmogrif.ai/en/stable/installation/index.html).

### Titanic Mini

First, build project with `./gradlew installDist`. Then run:

```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanicMini -Dargs="\
`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv"
```

### Titanic model - run with gradle (**recommended**)

First, build project with `./gradlew installDist`.

#### Train
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic -Dargs="\
--run-type=train \
--model-location=/tmp/titanic-model \
--read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv"
```
#### Score
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic -Dargs="\
--run-type=score \
--model-location=/tmp/titanic-model \
--read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv \
--write-location /tmp/titanic-scores"
```
#### Evaluate
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic -Dargs="\
--run-type=evaluate \
--model-location=/tmp/titanic-model \
--read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv \
--write-location /tmp/titanic-eval \
--metrics-location /tmp/titanic-metrics"
```

### Titanic model - run with `spark-submit`

First, build project with `./gradlew shadowJar`.

#### Train
```shell
$SPARK_HOME/bin/spark-submit --class com.salesforce.hw.titanic.OpTitanic \
  build/libs/transmogrifai-helloworld-0.0.1-all.jar \
  --run-type train \
  --model-location /tmp/titanic-model \
  --read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv
```
#### Score
```shell
$SPARK_HOME/bin/spark-submit --class com.salesforce.hw.titanic.OpTitanic \
  build/libs/transmogrifai-helloworld-0.0.1-all.jar \
  --run-type score \
  --model-location /tmp/titanic-model \
  --read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv \
  --write-location /tmp/titanic-scores
```
#### Evaluate
```shell
$SPARK_HOME/bin/spark-submit --class com.salesforce.hw.titanic.OpTitanic \
  build/libs/transmogrifai-helloworld-0.0.1-all.jar \
  --run-type evaluate \
  --model-location /tmp/titanic-model \
  --read-location Passenger=`pwd`/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv \
  --write-location /tmp/titanic-eval \
  --metrics-location /tmp/titanic-metrics
```

### Boston model

First, build project with `./gradlew installDist`.

#### Train
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=train \
--model-location=/tmp/boston-model \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data"
```
#### Score
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=score \
--model-location=/tmp/boston-model \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data \
--write-location=/tmp/boston-scores"
```
#### Evaluate
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.boston.OpBoston -Dargs="\
--run-type=evaluate \
--read-location BostonHouse=`pwd`/src/main/resources/BostonDataset/housing.data \
--write-location=/tmp/boston-eval \
--model-location=/tmp/boston-model \
--metrics-location=/tmp/boston-metrics"
```

### Iris model

First, build project with `./gradlew installDist`.

#### Train
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=train \
--model-location=/tmp/iris-model \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/iris.data"
```
#### Score
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=score \
--model-location=/tmp/iris-model \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/bezdekIris.data \
--write-location=/tmp/iris-scores"
```
#### Evaluate
```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.iris.OpIris -Dargs="\
--run-type=evaluate \
--model-location=/tmp/iris-model \
--metrics-location=/tmp/iris-metrics \
--read-location Iris=`pwd`/src/main/resources/IrisDataset/bezdekIris.data \
--write-location=/tmp/iris-eval"
```

### Data Preparation

First, build project with `./gradlew installDist`. Then run:

```shell
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.dataprep.JoinsAndAggregates -Dargs="\
`pwd`/src/main/resources/EmailDataset/Clicks.csv `pwd`/src/main/resources/EmailDataset/Sends.csv"

./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.dataprep.ConditionalAggregation -Dargs="\
`pwd`/src/main/resources/WebVisitsDataset/WebVisits.csv"
```

## Verify the Results

Look for the output file(s) in the location you specified. For instance, you can use `avro-tools` to inspect the scores files (on mac simply run `brew install avro-tools` to install it).

Other than that, the best way to verify the results is to look through the logs that should have been generated during the run. It has all kinds of information about the features the processing and the model reliability.

## Generate your own workflow

Experiment with adding feature changes or exploring more models in any of the provided workflows.

See how high you can get your AUROC!
