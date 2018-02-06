# Simple /* << APP_NAME */

This is an Optimus Prime project created with the 'simple' template.

## Prerequisites

- Java 1.8
- Scala ${scalaVersion}.${scalaVersionRevision}
- Spark ${sparkVersion}
- IntelliJ Idea 2017+ recommended
- Optimus Prime ${opVersion}


## Structure

The primary build file is in `build.gradle`.
This file defines dependencies on Scala, Spark, and Optimus Prime, and also defines how the project will be built
and deployed.

The primary sources for your project live in `src/main/scala`.
The spark application that you should run whenever you want to train/score/evaluate/etc. is the Simple /* << APP_NAME */
file in `src/main/scala/com/salesforce/app`.
Definitions for your features should reside in `src/main/scala/com/salesforce/app/Features.scala`, while the code that defines
where to get feature data from, what models to use, and any evaluation metrics lives in the application file.

## Workflow

You can run build commands by running `./gradlew` in this directory. Make sure that you have Spark installed, and that your
`SPARK_HOME` environment variable set to where you installed Spark.

### Building
To build the project, run `./gradlew build`. This will compile your sources and tell you of any compile errors.

### Training

Note: this platform runs on Spark, so you must download Spark ${sparkVersion} (prebuilt against hadoop 2.7), unpack and export `SPARK_HOME` before trying to run.

To train your project, run

```
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic /* << README_MAIN_CLASS */ \
-Dargs="--run-type=train --model-location /tmp/titanic-model /* << README_MODEL_LOCATION */ \
--read-location Passenger=\$ophw/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv /* << README_READ_LOCATION */"
```

### Scoring
To score your project, run

```
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic /* << README_MAIN_CLASS */ \
-Dargs="--run-type=score --model-location /tmp/titanic-model /* << README_MODEL_LOCATION */ \
--read-location Passenger=\$ophw/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv /* << README_READ_LOCATION */
--write-location /tmp/titanic-scores /* << README_SCORE_LOCATION */"
```

Replace the `read-location` parameter with whatever file you want to read for scoring.

## Evaluation
To evaluate your project, run

```
./gradlew -q sparkSubmit -Dmain=com.salesforce.hw.titanic.OpTitanic /* << README_MAIN_CLASS */ \
-Dargs="--run-type=evaluate --model-location /tmp/titanic-mode /* << README_MODEL_LOCATION */
 --read-location Passenger=\$ophw/src/main/resources/TitanicDataset/TitanicPassengersTrainData.csv /* << README_READ_LOCATION */
 --write-location /tmp/titanic-eval /* << README_EVAL_LOCATION */
 --metrics-location /tmp/titanic-metrics /* << README_METRICS_LOCATION */"
```

## Read More

- [Optimus Prime](https://github.com/salesforce/op)
- [Wiki](https://github.com/salesforce/op/wiki)
- [Hello World examples](https://github.com/salesforce/optimus-prime-helloworld)