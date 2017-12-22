# Octopus Prime (aka Optimus Prime)

An AutoML library for building modular, reusable, strongly typed machine learning workflows on Spark with minimal hand tuning.

## Overview
Optimus Prime is a Machine Learning (ML) library to simplfy development of modeling
workflows for multiple customers without hand tuning the models.
The library is written in Scala to be a type-safe layer on top of Spark.

## How does it work?
 A common machine learning example is modeling [survival for Titanic passengers](https://www.kaggle.com/c/titanic).
Going one step further, lets assume that you wanted to create survival models for *all* ship disasters
(and that the data for ships always came in the same format). You can define your workflow in Optimus Prime:

```scala
val features = Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).vectorize()
val (pred, raw, prob) = BinaryClassificationModelSelector().setInput(survived, features).getOutput()
val workflow = new OpWorkflow().setResultFeatures(pred)
```

Then simply pass in the training data for each ship disaster you wish to make a model for:

```scala
val titanicModel = workflow.setParameters(titanicDataParams).train()
val edmundFitzgeraldModel = workflow.setParameters(edmundFitzgeraldDataParams).train()
val wahineModel = workflow.setParameters(wahineDataParams).train()
```

Optimus Prime will make a highly customized and performant model for each ship disaster with no manual intervention from the developer.

## Motivation
 Creating a ML model for a particular application requires many steps, usually manually
performed by a data scientist or engineer. These steps include feature engineering, model selection,
hyperparameter tuning, exclusion of information that might falsely lead one to believe the model is
good (label leakage). In instances where it is necessary to create many individually tuned models
(for each client of a particular application) it is not feasible to have someone do this for
every single model. Similarly, companies or people with a great deal of domain knowledge but
lacking a deep understanding of ML may wish to create models for use in their domain. Optimus
Prime provides a solution for these use cases by automating feature engineering, model selection, 
hyperparameter tuning, and label leakage detection with a simple interface that allows users
to focus on what they are trying to model rather than ML details.

## AutoML
The AutoML functionality provided by Optimus Prime allows development of specialized machine learning
applications across many different customers in a code-once "meta" workflow. It also aids developers with
domain knowledge but lacking a machine learning background in creating high quality machine learning models.

### Feature engineering
The first component of Optimus Prime AutoML is smart feature engineering based on our rich
[type hierarchy](https://github.com/salesforce/op/wiki/Documentation#type-hierarchy-and-automatic-feature-engineering).
Optimus Prime defines many specific input feature types: Email, Phone, PostalCode, Categorical, Percentage, Currency, etc.

Default (type specific) feature transformations can be used to create a feature vector:

```scala
val features = Seq(pClass, name, sex, age, sibSp, parch, ticket, cabin, embarked).vectorize()
```

Or each of these feature types can be manipulated directly by users (using type safe operations with editor tab completion) to achieve the desired feature engineering steps:

```scala
val ageGroup = age.bucketize(splits = Seq(0, 10, 18, 40, 60, 120)).toMultiPickList().pivot()
```

### Sanity checking
The Sanity Checker can do feature selection, data cleaning, and detect label leakage
(having information that mirrors what you are trying to predict - which would not be available in scoring) automatically.

```scala
val checkedFeatures = survived.sanityCheck(featureVector)
```

### Model selection and tuning
Optimus Prime will select the best model and hyperparameters for you
based on the category of modeling you are doing (eg. Classification, Regression, Clustering, Recommendation,
etc). Smart model selection and comparison gives the next layer of improvements over traditional ML workflows.

```scala
val (pred, raw, prob) = BinaryClassificationModelSelector().setInput(survived, checkedFeatures).getOutput()
```

Of course, you can use a single model with manually chosen hyperparameters if you prefer.

```scala
val (pred, raw, prob) = new OpRandomForest()
   .setMaxDepth(5)
   .setMinInfoGain(0.1)
   .setNumTrees(100)
   .setInput(survived, checkedFeatures)
   .getOutput()
```

## Developer productivity
Optimus Prime is designed to help developers produce high quality customized models with a minimum of boilerplate code.

### Type safe operations
All operations on features and workflows in Optimus Prime are type safe to reduce the possibility of nasty runtime errors.

### Simple abstraction
![Alt text](resources/AbstractionDiagram-cropped.png?raw=true)

### Simple developer interface
Optimus Prime provides a number of base classes which can be used to easily add your own custom business logic
or algorithms to the platform. 

```scala
val scaleBy2 = new UnaryLambdaTransformer[Real, Real](
   operationName = "scaleBy2",
   transformFn = _.value.map(_ * 2.0).toReal
)
```

The above code is an example of an inline injection of custom logic. Of course your transform function could be much more
complicated in practice (the above can also be achieved by simply doing `feature * 2.0`). The point is that all code external
to the operation is handled for you, this stage can now be used, copied, saved, and reloaded just like any other Optimus
Prime stage.

### Easy code re-use across projects
Feature definitions and new stages are modular and can easily be shared across projects. This allows developers
to utilize others work rather than repeating the same feature engineering steps in every project that shares a
data source.

## QuickStart

We provide a convenient way to bootstrap you first project with Optimus Prime using the OP CLI.
Lets generate a binary classification model with Titanic passengers data.

Clone Optimus Prime repo:
```bash
git clone https://github.com/salesforce/op.git
```

Build the OP CLI by running:
```bash
cd ./optimus-prime
./gradlew cli:shadowJar
alias op="java -cp `pwd`/cli/build/libs/\* com.salesforce.op.cli.CLI"
```

Finally generate your Titanic model project (follow the instructions on screen):
```
op gen --input `pwd`/templates/simple/src/main/resources/PassengerData.csv \
  --id passengerId --response survived \
  --schema `pwd`/templates/simple/src/main/avro/Passenger.avsc Titanic
```

Your Titanic model project is ready to go. Happy modeling!

For more information on OP CLI read [this](cli/README.md).

## Adding OP into an existing project

You can simply add OP as a regular dependency to your existing project. Example for gradle below:

```gradle
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

## Documentation

See [Wiki](https://github.com/salesforce/op/wiki) for full documentation, getting started, examples and other information.

See [Scaladoc](https://op-docs.herokuapp.com/scaladoc/#package) for the programming API (can also be viewed [locally](docs/README.md)).


## License

Copyright (c) 2017, Salesforce.com, Inc.
All rights reserved.

