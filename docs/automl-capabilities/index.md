# AutoML Capabilities

## Vectorizers and Transmogrification

This is the Stage that automates the feature engineering step in the machine learning pipeline.

The TransmogrifAI [transmogrifier](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/stages/impl/feature/Transmogrifier.scala) (shortcut ```.transmogrify()```) takes in a sequence of features, automatically applies default transformations to them based on feature types (e.g. imputation, null value tracking, one hot encoding, tokenization, split Emails and pivot out the top K domains) and combines them into a single vector.  

```scala
val features = Seq(email, phone, age, subject, zipcode).transmogrify()
```


If you want to do the feature engineering at a single feature level, you can do so in combination with automatic type specific transformations. Each feature type has an associated ```.vectorize(....)``` method that will transform the feature into a feature vector given some input parameters. Each ```.vectorize(....)``` method behaves differently according to the type of feature being transformed. 

```scala
val emailFeature = email.vectorize()
val features = Seq(emailFeature, phone, age, subject, zipcode).transmogrify()
```

For advanced users, you can also completely [customize automatic feature engineering](../developer-guide#transmogrification).

## Feature Validation

#### SanityChecker

This is the Stage that automates the feature selection step in the machine learning pipeline.

The SanityChecker is an Estimator that can analyze a particular dataset for obvious issues prior to fitting a model on it.  It applies a variety of statistical tests to the data based on Feature types and discards predictors that are indicative of [label leakage](http://machinelearningmastery.com/data-leakage-machine-learning/) or that show little to no predictive power. In addition to flagging and fixing data issues, the SanityChecker also outputs statistics about the data for diagnostics and insight generation further down the ML pipeline.

The SanityChecker can be instantiated as follows:

```scala
// Add sanity checker estimator
val checkedFeatures = new SanityChecker().setRemoveBadFeatures(true).setInput(label, features).getOutput()
```
For advanced users, check out how to [customize default parameters](../developer-guide#sanitychecker) and peek into the SanityChecker metadata using model insights.

#### RawFeatureFilter

One of the fundamental assumptions of machine learning is that the data you are using to train your model reflects the data that you wish to score. In the real world, this assumption is often not true. TransmogrifAI has an optional stage after data reading that allows you to check that your features do not violate this assumption and remove any features that do. This stage is called the [RawFeatureFilter](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/filters/RawFeatureFilter.scala), and to use it you call the method `withRawFeatureFilter(Option(trainReader), Option(scoreReader),...)` on your [Workflows](../developer-guide#workflows). This method takes the training and scoring data readers as inputs.

```scala
// Add raw feature filter estimator
val workflow =
   new OpWorkflow()
      .setResultFeatures(survived, rawPrediction, prob, prediction)
      .withRawFeatureFilter(Option(trainReader), Option(scoreReader), None)
```

It will load the training and scoring data and exclude individual features based on fill rate, relative fill rates between training and scoring, or differences in the distribution of data between training and scoring. This stage can eliminate many issues, such as leakage of information that is only filled out after the label and changes in data collection practices, before they affect your model.

For advanced users, check out how to set [optional parameters](../developer-guide#rawfeaturefilter) for when to exclude features.


## ModelSelectors

This is the Stage that automates the model selection step in the machine learning pipeline.

TransmogrifAI will select the best model and hyper-parameters for you based on the class of modeling you are doing (eg. Classification, Regression etc.).
Smart model selection and comparison gives the next layer of improvements over traditional ML workflows.

```scala
val pred = BinaryClassificationModelSelector().setInput(label, features).getOutput()
```

The ModelSelector is an Estimator that uses data to find the best model. BinaryClassificationModelSelector is for  binary classification tasks, multi classification tasks can be done using MultiClassificationModelSelector. Best Regression model are done through RegressionModelSelector. Currently the possible classification models that can be applied in the selector are `GBTCLassifier`, `LinearSVC`, `LogisticRegression`, `DecisionTrees`, `RandomForest` and `NaiveBayes`, though `GBTClassifier` and `LinearSVC` only support binary classification. The possible regression models are `GeneralizedLinearRegression`,  `LinearRegression`, `DecisionTrees`, `RandomForest` and `GBTreeRegressor`. The best model is selected via a CrossValidation or TrainingSplit, by picking the best model and wrapping it. By default each of these models comes with a predefined set of hyperparameters that will be tested in determining the best model.  

For advanced users, check out how to specify specific models and hyperparameters, add your own models, set validation parameters, and balance datasets [here](../developer-guide#modelselector).

