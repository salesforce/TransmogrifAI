# Octopus Prime (aka Optimus Prime) [![Build Status](https://travis-ci.com/salesforce/op.svg?token=Ex9czVEUD7AzPTmVh6iX&branch=master)](https://travis-ci.com/salesforce/op)

Abstract away the redundant, repeatable feature engineering, feature selection and model selection tasks slowing down ML development.

## Overview
Octopus Prime (aka Optimus Prime) is an AutoML library written in Scala that runs on top of Spark. It was developed with a focus on accelerating machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity, and reuse.
_Through automation, it achieves accuracies close to hand-tuned models with almost 100x reduction in time._


Use Optimus Prime if you need a machine learning library to:

* Build machine learning applications in hours, not months
* Build machine learning models without having a machine learning background
* Build modular, reusable, strongly typed machine learning workflows

Optimus Prime is compatible with Spark 2.2.1 and Scala 2.11.

[Skip to Quick Start and Documentation](https://github.com/salesforce/op#quick-start-and-documentation)

## Motivation
_Building real life machine learning applications need fair amount of tribal knowledge and intuition. Coupled with the explosion of ML use cases in the world that need to be addressed, there are not enough data scientists to build all the applications and democratize it. Automation is the solution to making machine learning truly accessible._

Creating a ML model for a particular application requires many steps, usually manually performed by a data scientist or engineer. These steps include ETL, feature engineering, model selection (including hyper-parameter tuning), safeguarding against data leakage, operationalizing models, scoring and updating models.

![Alt text](resources/pipeline.png?raw=true)

If your organization or product has multiple use cases, it is necessary to create many individually tuned models (for each such use case). Scaling this is difficult, time consuming and expensive.

![Alt text](resources/pipelineN.png?raw=true)

Optimus Prime provides a solution for these use cases by automating feature engineering, feature selection (including data leakage detection), model selection and hyper-parameter tuning with a simple interface that allows users to focus on what they are trying to model rather than ML details.

## AutoML
The AutoML functionality provided by Optimus Prime allows development of specialized machine learning applications across many different customers in a code-once "meta" workflow.

### Feature engineering
Optimus Prime vectorizers (shortcut  ```.autoTransform()``` , aka ```.transmogrify()``` ) take in a sequence of features, automatically apply default transformations to them based on feature types (e.g. split Emails and pivot out the top K domains) and combine them into a single vector. This is in essence the automatic feature engineering Stage of Optimus Prime.
![Alt text](resources/feateng.png?raw=true)

```scala
val features = Seq(email, phone, age, subject, zipcode).autoTransform()
```
Of course specific feature engineering is also possible and can be used in combination with automatic type specific transformations.

This can be manipulated directly by users (using type safe operations with editor tab completion) to achieve the desired feature engineering steps:

```scala
val ageGroup = age.bucketize(splits = Seq(0, 10, 18, 40, 60, 120)).toMultiPickList().pivot()
```
### Feature Selection
The feature selection step happens within the sanity checker. It does the following:

1. Analyze every feature and out put descriptive statistics such as mean, min, max, variance, number of nulls to ensure features have acceptable ranges
2. Compute association of every feature to the label (correlations, cramersV, pointwise mutual information and other statistical tests) and drop those with low predictive power
3. Detect data leakage (having information that mirrors what you are trying to predict - which would not be available in scoring).


```scala
val checkedFeatures = new SanityChecker().setInput(label, features).getOutput()
```

### Model selection and tuning
Optimus Prime will select the best model and hyper-parameters for you based on the class of modeling you are doing (eg. Classification, Regression etc.).
Smart model selection and comparison gives the next layer of improvements over traditional ML workflows.

```scala
val (pred, raw, prob) = BinaryClassificationModelSelector().setInput(label, checkedFeatures).getOutput()
```

Of course, you can use a single model with manually chosen hyper-parameters if you prefer.

```scala
val (pred, raw, prob) = new OpRandomForest()
   .setMaxDepth(5)
   .setMinInfoGain(0.1)
   .setNumTrees(100)
   .setInput(label, checkedFeatures)
   .getOutput()
```
### Putting it all together

```scala
case class Schema
(
  id: Int,
  email: Option[String],
  phone: Option[String],
  age: Int,
  subject:Option[String],
  zipcode:Option[String],
  label: int
)

//Build Features
val email = FeatureBuilder.Email[Schema].extract(asEmail(_.getEmail)).asPredictor
val phone = FeatureBuilder.Phone[Schema].extract(asPhone(_.getPhone)).asPredictor
val subject = FeatureBuilder.Text[Schema].extract(asText(_.getSubject)).asPredictor
val zipcode = FeatureBuilder.PostalCode[Schema].extract(asPostalCode(_.getZipcode)).asPredictor
val age = FeatureBuilder.Real[Schema].extract(_.getAge.toReal).asPredictor
val label = FeatureBuilder.RealNN[Schema].extract(_.label.toRealNN).asResponse

//Automated Feature Engineering
val features = Seq(email, phone, age, subject, zipcode).autoTransform()

//Automated Feature Selection
val checkedFeatures = new SanityChecker().setInput(label, features).getOutput()

//Automated Model Selection
val (prediction, rawPrediction, probability) = BinaryClassificationModelSelector().setInput(label, checkedFeatures).getOutput()

//Training Data
val trainDataReader = DataReaders.Simple.csvCase[Schema](path = Some("PathToDataFile"), key = _.id.toString)

//Create and fit workflow
val workflow = new OpWorkflow().setResultFeatures(label, rawPrediction, probability, prediction).setReader(trainDataReader)
val fittedWorkflow = workflow.train()
```

## Adding OP into an existing project
You can simply add OP as a regular dependency to your existing project. Example for gradle below:

```groovy
repositories {
    maven {
       url "https://raw.githubusercontent.com/salesforce/op/mvn-repo/releases"
       credentials {
           // Generate github api token here - https://goo.gl/ANZ9oz
           // and then set it as an environment variable `export GITHUB_API_TOKEN=<MY_TOKEN>`
           username = System.getenv("GITHUB_API_TOKEN")
           password "" // leave the password empty
       }
       authentication { digest(BasicAuthentication) }
    }
}
ext {
    scalaVersion = '2.11'
    scalaVersionRevision = '8'
    sparkVersion = <SPARK_VERSION> // Set the required Spark version here
    opVersion = <OP_VERSION> // Set the required OP version here
}
dependencies {
    // Scala
    scalaLibrary "org.scala-lang:scala-library:$scalaVersion.$scalaVersionRevision"
    scalaCompiler "org.scala-lang:scala-compiler:$scalaVersion.$scalaVersionRevision"
    compile "org.scala-lang:scala-library:$scalaVersion.$scalaVersionRevision"

    // Spark (needed only at compile / test time)
    compileOnly "org.apache.spark:spark-core_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-core_$scalaVersion:$sparkVersion"
    compileOnly "org.apache.spark:spark-mllib_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-mllib_$scalaVersion:$sparkVersion"
    compileOnly "org.apache.spark:spark-sql_$scalaVersion:$sparkVersion"
    testCompile "org.apache.spark:spark-sql_$scalaVersion:$sparkVersion"

    // Optimus Prime
    compile "com.salesforce:optimus-prime-core_$scalaVersion:$opVersion"

    // Other depdendecies
}
```

## Abstractions

Optimus Prime is designed to simplify the creation of machine learning workflows. To this end we have created an abstraction for creating and running machine learning workflows.
The abstraction is made up of Features, Stages, Workflows and Readers which interact as shown in the diagram below.

![Alt text](resources/abstractions.png?raw=true)

**Features:** The primary abstraction that users of Optimus Prime interact with are Features. Features are essentially type-safe pointers to data columns with additional metadata built in. Features are the elements which users manipulate and interact with in order to define all steps in the machine learning workflow. In our abstraction Features are acted on by Stages in order to produce new Features. Part of the metadata contained in Features is strict type information about the column. This is used both to determine which Stages can be called on a given Feature and which Stages should be called when automatic feature engineering Stages are used. Because the output of every Stage is a Feature or set of Features, any sequence of type safe operations can be strung together to create a machine learning workflow.

**Stages:** Stages define actions that you wish to perform on Features in your workflow. Those familiar with Spark ML will recognize the idea of Stages being either Transformers or Estimators. Transformers provide functions for transforming one or more Features in your data to one or more new Features. Estimators provide algorithms which when applied to one or more Features produce Transformers. The Optimus Prime Transformers and Estimators extend Spark ML Transformers and Estimators and can be used as standard Spark stages if desired. In both Spark ML and Optimus Prime when Stages are used within a workflow the user does not need to distinguish between types of Stages (Estimator or Transformer), this distinction is only important for developers developing new Estimators or Transformers.

**Workflows and Readers:** Once the final desired Feature, or Features, have been defined they are materialized by feeding the final Features into a Workflow. The Workflow will trace back how the final Features were created and make an optimized DAG of Stage executions in order to produce the final Features. The Workflow must also be provided a DataReader. The DataReader can do complex data pre-processing steps or simply load a dataset. The key component of the DataReader is that the type of the data produced by the reader must match the type of the data expected by the initial feature generation stages.

## Quick Start and Documentation

See [Wiki](https://github.com/salesforce/op/wiki) for full documentation, getting started, examples and other information.

See [Scaladoc](https://op-docs.herokuapp.com/scaladoc/#package) for the programming API (can also be viewed [locally](docs/README.md)).


## License

Copyright (c) 2017, Salesforce.com, Inc.
All rights reserved.
