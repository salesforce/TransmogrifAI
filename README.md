# TransmogrifAI [![Build Status](https://travis-ci.com/salesforce/TransmogrifAI.svg?token=Ex9czVEUD7AzPTmVh6iX&branch=master)](https://travis-ci.com/salesforce/TransmogrifAI) [![Scala version](https://img.shields.io/badge/scala-2.11-brightgreen.svg)](https://www.scala-lang.org/download/2.11.12.html) [![Spark version](https://img.shields.io/badge/spark-2.2.1-brightgreen.svg)](https://spark.apache.org/news/spark-2-2-1-released.html)


TransmogrifAI (pronounced trăns-mŏgˈrə-fī) is an AutoML library written in Scala that runs on top of Spark. It was developed with a focus on accelerating machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity, and reuse.
_Through automation, it achieves accuracies close to hand-tuned models with almost 100x reduction in time._

Use TransmogrifAI if you need a machine learning library to:

* Build production ready machine learning applications in hours, not months
* Build machine learning models without getting a Ph.D. in machine learning
* Build modular, reusable, strongly typed machine learning workflows

TransmogrifAI is compatible with Spark 2.3.1 and Scala 2.11.

[Skip to Quick Start and Documentation](https://github.com/salesforce/TransmogrifAI#quick-start-and-documentation)

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

// Extract response and predictor variables
val (survived, features) = FeatureBuilder.fromDataFrame[RealNN](passengersData, response = "survived")

// Automated feature engineering of predictors
val featureVector = features.toSeq.transmogrify()

// Automated feature selection
val checkedFeatures = survived.sanityCheck(featureVector, checkSample = 1.0, sampleSeed = 42, removeBadFeatures = true)

// Automated model selection
val (pred, raw, prob) = BinaryClassificationModelSelector().setInput(survived, checkedFeatures).getOutput()
val model = new OpWorkflow().setInputDataset(passengersData).setResultFeatures(pred).train()

println("Model summary: " + model.summary())
```
Model summary:

```
Evaluated Logistic Regression, Random Forest models with 3 folds and AuPR metric.
Evaluated 3 Logistic Regression models with AuPR between [0.6751930383321765, 0.7768725281794376]
Evaluated 16 Random Forest models with AuPR between [0.7781671467343991, 0.8104798040316159]

Selected model Random Forest classifier with parameters:
|-----------------------|:------------:|
| Model Param           |     Value    |
|-----------------------|:------------:|
| modelType             | RandomForest |
| featureSubsetStrategy |         auto |
| impurity              |         gini |
| maxBins               |           32 |
| maxDepth              |           12 |
| minInfoGain           |        0.001 |
| minInstancesPerNode   |           10 |
| numTrees              |           50 |
| subsamplingRate       |          1.0 |
|-----------------------|:------------:|

Model evaluation metrics:
|-------------|:------------------:|:-------------------:|
| Metric Name | Hold Out Set Value |  Training Set Value |
|-------------|:------------------:|:-------------------:|
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
|-------------|:------------------:|:-------------------:|

Top model insights computed using correlation:
|-----------------------|:--------------------:|
| Top Positive Insights |      Correlation     |
|-----------------------|:--------------------:|
| sex = "female"        |   0.5177801026737666 |
| cabin = "OTHER"       |   0.3331391338844782 |
| pClass = 1            |   0.3059642953159715 |
|-----------------------|:--------------------:|
| Top Negative Insights |      Correlation     |
|-----------------------|:--------------------:|
| sex = "male"          |  -0.5100301587292186 |
| pClass = 3            |  -0.5075774968534326 |
| cabin = null          | -0.31463114463832633 |
|-----------------------|:--------------------:|

Top model insights computed using CramersV:
|-----------------------|:--------------------:|
|      Top Insights     |       CramersV       |
|-----------------------|:--------------------:|
| sex                   |    0.525557139885501 |
| embarked              |  0.31582347194683386 |
| age                   |  0.21582347194683386 |
|-----------------------|:--------------------:|
```

While this may seem a bit too magical, for those who want more control, TransmogrifAI also provides the flexibility to completely specify all the features being extracted and all the algorithms being applied in your ML pipeline. See [Wiki](https://github.com/salesforce/TransmogrifAI/wiki) for full documentation, getting started, examples and other information.


## Adding TransmogrifAI into your project
You can simply add TransmogrifAI as a regular dependency to your existing project. Example for gradle below:

```groovy
repositories {
    mavenCentral()
    maven { url "https://jitpack.io" }
    maven {
        url "s3://op-repo/releases"
        credentials(AwsCredentials) {
            // user: op-repo-reader
            accessKey "AKIAJ6AZFFSFRJI3IKHQ"
            secretKey "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"
        }
    }
}
ext {
    scalaVersion = '2.11'
    scalaVersionRevision = '8'
    sparkVersion = '2.2.1'
    opVersion = '3.3.3'
}
dependencies {
    // Scala
    scalaLibrary "org.scala-lang:scala-library:$scalaVersion.$scalaVersionRevision"
    scalaCompiler "org.scala-lang:scala-compiler:$scalaVersion.$scalaVersionRevision"
    compile "org.scala-lang:scala-library:$scalaVersion.$scalaVersionRevision"

    // Spark
    compileOnly "org.apache.spark:spark-core_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-core_$scalaVersion:$sparkVersion"
    compileOnly "org.apache.spark:spark-mllib_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-mllib_$scalaVersion:$sparkVersion"
    compileOnly "org.apache.spark:spark-sql_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-sql_$scalaVersion:$sparkVersion"

    // TransmogrifAI
    compile "com.salesforce.op:octopus-prime-core_$scalaVersion:$opVersion"

    // Pretrained models used in TransmogrifAI, e.g. OpenNLP POS/NER models etc. (optional)
    // compile "com.salesforce.op:octopus-prime-models_$scalaVersion:$opVersion"

    // All your other depdendecies go below
    // ...
}
```

## Quick Start and Documentation

See [Wiki](https://github.com/salesforce/TransmogrifAI/wiki) for full documentation, getting started, examples and other information.

See [Scaladoc](https://op-docs.herokuapp.com/scaladoc/#package) for the programming API (can also be viewed [locally](docs/README.md)).

## License

[BSD 3-Clause](LICENSE) © Salesforce.com, Inc.
