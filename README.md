# TransmogrifAI

[![Download](https://api.bintray.com/packages/salesforce/maven/TransmogrifAI/images/download.svg)](https://bintray.com/salesforce/maven/TransmogrifAI/_latestVersion) [![TravisCI Build Status](https://travis-ci.com/salesforce/TransmogrifAI.svg?token=Ex9czVEUD7AzPTmVh6iX&branch=master)](https://travis-ci.com/salesforce/TransmogrifAI) [![CircleCI Build Status](https://circleci.com/gh/salesforce/TransmogrifAI.svg?&style=shield&circle-token=e84c1037ae36652d38b49207728181ee85337e0b)](https://circleci.com/gh/salesforce/TransmogrifAI) [![Codecov](https://codecov.io/gh/salesforce/TransmogrifAI/graph/badge.svg?token=snKCVButEm)](https://codecov.io/gh/salesforce/TransmogrifAI) [![Spark version](https://img.shields.io/badge/spark-2.2-brightgreen.svg)](https://spark.apache.org/downloads.html) [![Scala version](https://img.shields.io/badge/scala-2.11-brightgreen.svg)](https://www.scala-lang.org/download/2.11.12.html) [![License](http://img.shields.io/:license-BSD--3-blue.svg)](./LICENSE)

TransmogrifAI (pronounced trăns-mŏgˈrə-fī) is an AutoML library written in Scala that runs on top of Spark. It was developed with a focus on accelerating machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity, and reuse.
_Through automation, it achieves accuracies close to hand-tuned models with almost 100x reduction in time._

Use TransmogrifAI if you need a machine learning library to:

* Build production ready machine learning applications in hours, not months
* Build machine learning models without getting a Ph.D. in machine learning
* Build modular, reusable, strongly typed machine learning workflows

Skip to [Quick Start and Documentation](#quick-start-and-documentation).

## Predicting Titanic Survivors with TransmogrifAI

The Titanic dataset is an often-cited dataset in the machine learning community. The goal is to build a machine learnt model that will predict survivors from the Titanic passenger manifest. Here is how you would build the model using TransmogrifAI:

```scala
import com.salesforce.op._
import com.salesforce.op.readers._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.impl.classification._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
import spark.implicits._

// Read Titanic data as a DataFrame
val passengersData = DataReaders.Simple.csvCase[Passenger](path = pathToData).readDataset().toDF()

// Extract response and predictor Features
val (survived, predictors) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = "survived")

// Automated feature engineering
val featureVector = predictors.transmogrify()

// Automated feature validation and selection
val checkedFeatures = survived.sanityCheck(featureVector, removeBadFeatures = true)

// Automated model selection
val (pred, raw, prob) = BinaryClassificationModelSelector().setInput(survived, checkedFeatures).getOutput()

// Setting up a TransmogrifAI workflow and training the model
val model = new OpWorkflow().setInputDataset(passengersData).setResultFeatures(pred).train()

println("Model summary:\n" + model.summaryPretty())
```
Model summary:

```
Evaluated Logistic Regression, Random Forest models with 3 folds and AuPR metric.
Evaluated 3 Logistic Regression models with AuPR between [0.6751930383321765, 0.7768725281794376]
Evaluated 16 Random Forest models with AuPR between [0.7781671467343991, 0.8104798040316159]

Selected model Random Forest classifier with parameters:
|-----------------------|--------------|
| Model Param           |     Value    |
|-----------------------|--------------|
| modelType             | RandomForest |
| featureSubsetStrategy |         auto |
| impurity              |         gini |
| maxBins               |           32 |
| maxDepth              |           12 |
| minInfoGain           |        0.001 |
| minInstancesPerNode   |           10 |
| numTrees              |           50 |
| subsamplingRate       |          1.0 |
|-----------------------|--------------|

Model evaluation metrics:
|-------------|--------------------|---------------------|
| Metric Name | Hold Out Set Value |  Training Set Value |
|-------------|--------------------|---------------------|
| Precision   |               0.85 |   0.773851590106007 |
| Recall      | 0.6538461538461539 |  0.6930379746835443 |
| F1          | 0.7391304347826088 |  0.7312186978297163 |
| AuROC       | 0.8821603927986905 |  0.8766642291593114 |
| AuPR        | 0.8225075757571668 |   0.850331080886535 |
| Error       | 0.1643835616438356 | 0.19682151589242053 |
| TP          |               17.0 |               219.0 |
| TN          |               44.0 |               438.0 |
| FP          |                3.0 |                64.0 |
| FN          |                9.0 |                97.0 |
|-------------|--------------------|---------------------|

Top model insights computed using correlation:
|-----------------------|----------------------|
| Top Positive Insights |      Correlation     |
|-----------------------|----------------------|
| sex = "female"        |   0.5177801026737666 |
| cabin = "OTHER"       |   0.3331391338844782 |
| pClass = 1            |   0.3059642953159715 |
|-----------------------|----------------------|
| Top Negative Insights |      Correlation     |
|-----------------------|----------------------|
| sex = "male"          |  -0.5100301587292186 |
| pClass = 3            |  -0.5075774968534326 |
| cabin = null          | -0.31463114463832633 |
|-----------------------|----------------------|

Top model insights computed using CramersV:
|-----------------------|----------------------|
|      Top Insights     |       CramersV       |
|-----------------------|----------------------|
| sex                   |    0.525557139885501 |
| embarked              |  0.31582347194683386 |
| age                   |  0.21582347194683386 |
|-----------------------|----------------------|
```

While this may seem a bit too magical, for those who want more control, TransmogrifAI also provides the flexibility to completely specify all the features being extracted and all the algorithms being applied in your ML pipeline. See [Wiki](https://github.com/salesforce/TransmogrifAI/wiki) for full documentation, getting started, examples and other information.


## Adding TransmogrifAI into your project
You can simply add TransmogrifAI as a regular dependency to an existing project.

For Gradle in `build.gradle` add:
```groovy
repositories {
    mavenCentral()
    maven { url 'https://dl.bintray.com/salesforce/maven' }
}
dependencies {
    // TransmogrifAI core dependency
    compile 'com.salesforce.transmogrifai:transmogrifai-core_2.11:0.3.4'

    // TransmogrifAI pretrained models, e.g. OpenNLP POS/NER models etc. (optional)
    // compile 'com.salesforce.transmogrifai:transmogrifai-models_2.11:0.3.4'
}
```

For SBT in `build.sbt` add:
```scala
scalaVersion := "2.11.12"

resolvers += Resolver.bintrayRepo("salesforce", "maven")

// TransmogrifAI core dependency
libraryDependencies ++= "com.salesforce.transmogrifai" %% "transmogrifai-core" % "0.3.4"

// TransmogrifAI pretrained models, e.g. OpenNLP POS/NER models etc. (optional)
// libraryDependencies ++= "com.salesforce.transmogrifai" %% "transmogrifai-models" % "0.3.4"
```

Then import TransmogrifAI into your code:
```scala
// TransmogrifAI functionality: feature types, feature builders, feature dsl, readers, aggregators etc.
import com.salesforce.op._
import com.salesforce.op.aggregators._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.readers._

// Spark enrichments (optional)
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRDD._
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.RichStructType._
```


## Quick Start and Documentation

See the [Wiki](https://github.com/salesforce/TransmogrifAI/wiki) for full documentation, getting started, examples and other information.

See [Scaladoc](https://docs.transmogrif.ai) for the programming API (can also be viewed [locally](docs/README.md)).

## License

[BSD 3-Clause](LICENSE) © Salesforce.com, Inc.
