# Abstractions

TransmogrifAI is designed to simplify the creation of machine learning workflows. To this end we have created an abstraction for creating and running machine learning workflows. The abstraction is made up of Features, Stages, Workflows and Readers which interact as shown in the diagram below.

![TransmogrifAI Abstractions](https://github.com/salesforce/TransmogrifAI/raw/master/resources/AbstractionDiagram-cropped.png)

## Features

The primary abstraction introduced in TransmogrifAI is that of a Feature. A Feature is essentially a type-safe pointer to a column in a DataFrame and contains all information about that column -- it's name, the type of data to be found in it, as well as lineage information about how it was derived. Features are defined using FeatureBuilders:

```scala
val name: Feature[Text] = FeatureBuilder.Text[Passenger].extract(_.name.toText).asPredictor
val age: Feature[RealNN] = FeatureBuilder.RealNN[Passenger].extract(_.age.toRealNN).asPredictor
```

The above lines of code define two ```Features``` of type ```Text``` and ```RealNN``` called ```name``` and ```age``` that are extracted from data of type ```Passenger``` by applying the stated extract methods. 

One can also define Features that are the result of complex time-series aggregates. Take a look at this [example](/Examples/Time-Series-Aggregates-and-Joins.html) and this [page](/Developer-Guide#aggregate-data-readers) for more advanced reading on FeatureBuilders.

Features can then be manipulated using Stages to produce new Features. In TransmogrifAI, as in SparkML, there are two types of Stages -- Transformers and Estimators.

## Stages

### Transformers

Transformers specify functions for transforming one or more Features to one or more *new* Features. Here is an example of applying a tokenizing Transformer to the ```name``` Feature defined above:

```scala
val nameTokens = new TextTokenizer[Text]().setAutoDetectLanguage(true).setInput(name).getOutput()
```

The output ```nameTokens``` is a new Feature of type ```TextList```. Because Features are strongly typed, it is also possible to create shortcuts for these Transformers and create a Feature operation syntax. The above line could alternatively have been written as:

```scala
val nameTokens = name.tokenize()
``` 
TransmogrifAI provides an easy way for wrapping all Spark Transformers, and additionally provides many Transformers of its own. For more reading about creating new Transformers and shortcuts, follow the links [here](/Developer-Guide#transformers) and [here](/Developer-Guide#creating-shortcuts-for-transformers-and-estimators).

### Estimators

Estimators specify algorithms that can be applied to one or more Features to produce Transformers that in turn produce new Features. Think of Estimators as learning algorithms, that need to be fit to the data, in order to then be able to transform it. Users of TransmogrifAI do not need to worry about the fitting of algorithms, this happens automatically behind the scenes when a TransmogrifAI workflow is trained. Below we see an example of a use of a bucketizing estimator that determines the buckets that maximize information gain when fit to the data, and then transforms the Feature ```age``` to a new bucketized Feature of type  ```OPVector```:

```scala
val bucketizedAge = new DecisionTreeNumericBucketizer[Double, Real]().setInput(label, age).getOutput()
```

Similar to Transformers above, one can easily create shortcuts for Estimators, and so the line of code above could have been alternatively written as: 

```scala
val bucketizedAge = age.autoBucketize(label = label)
```
TransmogrifAI provides an easy way for wrapping all Spark Estimators, and additionally provides many Estimators of its own. For more reading about creating new Estimators follow the link [here](/Developer-Guide#estimators).

## Workflows and Readers

Once all the Features and Feature transformations have been defined, actual data can be materialized by adding the desired Features to a TransmogrifAI Workflow and feeding it a DataReader. When the Workflow is trained, it infers the entire DAG of Features, Transformers, and Estimators that are needed to materialize the result Features. It then prepares this DAG by passing the data specified by the DataReader through the DAG and fitting all the intermediate Estimators in the DAG to Transformers.

In the example below, we would like to materialize ```bucketizedAge``` and ```nameTokens```. So we set these two Features as the result Features for a new Workflow:

```
val workflow = new OPWorkflow().setResultFeatures(bucketizedAge, nameTokens).setReader(PassengerReader)
```

The PassengerReader is a DataReader that essentially specifies a ```read``` method that can be used for loading the Passenger data. When we train this workflow, it reads the Passenger data and fits the bucketization estimator by determining the optimal buckets for ```age```:

```
val workflowModel = workflow.train()
```

The workflowModel now has a prepped DAG of Transformers. By calling the ```score``` method on the workflowModel, we can transform any data of type Passenger to a DataFrame with two columns for ```bucketizedAge``` and ```nameTokens``` 

```
val dataFrame = workflowModel.setReader(OtherPassengerReader).score()
```

WorkflowModels can be saved and loaded. For more advanced reading on topics like stacking workflows, aggregate DataReaders for time-series data, or joins for DataReaders, follow our links to [Workflows](/Developer-Guide#workflows) and [Readers](/Developer-Guide#datareaders).





