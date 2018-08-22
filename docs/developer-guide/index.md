# Developer Guide

## Features

Features are objects within TransmogrifAI which contain all information about what data is in a column and how it was created from previous data. The `name` value contained within the Feature is the same as the name of the column that will be created in the DataFrame when the data is materialized.

```scala
/**
 * Feature instance
 *
 * @param name        name of feature, represents the name of the column in the dataframe.
 * @param isResponse  whether or not this feature is a response feature, ie dependent variable
 * @param originStage reference to OpStage responsible for generating the feature.
 * @param parents     references to the features that are transformed by the originStage that produces this feature
 * @param uid         unique identifier of the feature instance
 * @param wtt         feature's value type tag
 * @tparam O feature value type
 */
private[op] case class Feature[O <: FeatureType]
(
  name: String,
  isResponse: Boolean,
  originStage: OpPipelineStage[O],
  parents: Seq[OPFeature] = Seq.empty,
  uid: String = UID[Feature[O]]
)(implicit val wtt: WeakTypeTag[O]) extends FeatureLike[O] {
  /* ... */
  def history: FeatureHistory // contains history of how feature was created
}
```

### Type Hierarchy and Automatic Feature Engineering 

The FeatureType associated with each  Feature (`Feature[O <: FeatureType]`) must be part of our FeatureType hierarchy. When features are defined with a specific type (e.g. Email rather than Text), that type determines which Stages can be applied to the feature. In addition, TransmogrifAI can use this type information to automatically apply appropriate feature engineering manipulations (e.g. split Emails and pivot out the top K domains). This means that rather than specifying all manipulations necessary for each feature type the user can simply use the `.transmogrify()` method on all input features. Of course specific feature engineering is also possible and can be used in combination with automatic type specific transformations.

The FeatureType hierarchy of TransmogrifAI is shown below:
![TransmogrifAI Type hierarchy](https://github.com/salesforce/TransmogrifAI/raw/master/resources/Type-Hierarchy.png)


### Feature Creation

Each Feature is created either using a FeatureBuilder (which is a special type of OpStage that takes in the type of the raw data and produces a single feature) or by manipulating other Features (by applying Stages) in order to make new Features. The full history of all stages used to make a feature are contained within the Feature values. In essence, Features contain the typed data transformation plan for the workflow the user creates. 

## FeatureBuilders

FeatureBuilders are used to specify how raw features must be extracted and aggregated from the source data. They are passed into the [DataReaders](/Developer-Guide#datareaders) during the  feature generation stage in order to produce the data expected by later Stages (Estimators and Transformers) in the Workflow. FeatureBuilders specify all of the information needed to make the feature with various parts of the build command, for example:

```scala
val sex = FeatureBuilder.PickList[Passenger]
  .extract(d => d.sex.map(_.toString).toSet[String].toPickList).asPredictor
```

specifies the type of feature you are creating (`.PickList`), the type of the data you are extracting the feature from (`[Passenger]`), how to get the feature from that data (`.extract(d => d.sex.map(_.toString).toSet[String].toPickList)` ) and whether the feature is a response or predictor (`.asPredictor`). In addition the name of the feature is inferred from the variable name (`val sex`),  the name can also be explicitly passed in if it needs to be different from the variable name (`FeatureBuilder.PickList[Passenger](“name”)`). This is all the information that is required to extract a simple feature from a dataset. 

If the feature creation will also involve aggregation across rows (for example if multiple days of snapshot data is being loaded) additional information is needed:

```scala
val fare = FeatureBuilder.Currency[Passenger].extract(_.fare.toCurrency)
  .aggregate(_ + _).window(Duration.standardDays(7)).asPredictor
```

Both the aggregation function (`.aggregate(_ + _)`) and the aggregation time window (`.window(Duration.standardDays(7))`) have default values which will be used if the user does not specify them. These methods specify how to combine rows and which timestamped rows to aggregate respectively.

## Stages

Stages are the actually manipulations performed on Features. Notice that some of the manipulations are defined via Estimators and some via Transformers. An Estimator defines an algorithm that can be applied to one or more Features to produce a Transformer. The key difference between the two types of Stages is that Estimators can access all the information in the column or columns pointed to by the input Features. Transformers, on the other hand, simply act as transformations applied to the Feature value of a single row of the input data. The `NormalizeEstimator`, for instance, computes the mean and standard deviation of the input Feature (e.g. age) in order to produce a Transformer. Estimators are converted to Transformers when a Workflow is fitted.

ML algorithms, such as Logistic Regression, are examples of Estimators that when fitted to data produce a fitted Transformer (or Model -  which is just a Transformer that was produced by an Estimator). All Estimators and Transformers are used to produce new Features which can then be manipulated in turn by any Estimator or Transformer that accepts that Feature type as an input.

## Transformers

Once you have specified how the raw features are extracted, you can also specify how you would like them to be transformed via Transformers.

Transformers are a Spark ML concept that describes classes which perform a map operation from one or several columns in a dataset to a new column. The important thing to keep in mind about Transformers is that because they are map operations they act only within a row - no information about the contents of other rows can be integrated into transformers directly. If information about the contents of the full column are needed (for instance if you need to know the distribution of a numeric column) you must use an [Estimator](/Developer-Guide#estimators) to produce a transformer which contains that information.

### TransmogrifAI Transformers

TransmogrifAI Transformers extend Spark Transformers but are designed to interact with Features rather than DataFrame column names. Rather than directly passing in the data to transform function, the input Feature (or Features) are set using the setInput method and the return Feature (or Features) are obtained using the getOutput method. Both the input and output Features can then be acted on as many times as desired to obtain the final result Features. When a Workflow is called, the data described by the output Feature is materialized by passing the data described by the input Features into the transformer.

### Writing your own transformer

TransmogrifAI Transformers can easily be created by finding the appropriate base class and extending it. The TransmogrifAI Transformer base classes have default implementations for all the book keeping methods associated with Spark and OP. Thus when using OpTranfomer base classes the only things that need to be defined are the name of the operation associated with your Transformer, a default uid function for your transformer,  and the function that maps the input to the output.

**Note that when creating a new stage (Transformer or Estimator) the uid should always be a constructor argument.** If the uid is not a constructor  the stages will not be serialized correctly when models are saved.

TransmogrifAI Transformer base classes are defined by the number of input features and the number of output features. Transformers with a single output feature can take one (UnaryTransformers), two (BinaryTransformers), three (TernaryTransformers), four (QuaternaryTransformers), or a sequence (of a single type - SequenceTransformers) inputs. If multiple outputs are desired multistage Transformers can be used to generate up to three output features.

**Lambda expressions**

If the transformation you wish to perform can be written as a function with no external inputs the lambda base classes can be used. 

Simply find the appropriate base class:

```scala
class UnaryLambdaTransformer[I, O]
(
  override val operationName: String,
  val transformFn: I => O,
  override val uid: String = UID[UnaryTransformerBase[I, O])
  ) (implicit override ...)
  extends UnaryTransformerBase[I, O](operationName = operationName, uid = uid)
```

And extend it to create the new transformer:

```scala
class LowerCaseTransformer
(
  override val uid: String = UID[LowerCaseTransformer]
) extends UnaryTransformer[Text, Text](
  operationName = "lowerCase",
  transformFn = _.map(_.toLowerCase),
  uid = uid
)
```


**Creating your own params**

Sometimes transformation functions could be carried out in multiple ways. Rather than creating many very similar transformers it is desirable to allow parameters to be set for transformations so that a single transformer can carry out any of these functions. The lambda transformer base classes cannot be used in this instance. Rather the class from which they inherit the *aryTransformer class should be used:

```scala
abstract class UnaryTransformer[I, O]
(
  override val operationName: String,
  val uid: String
)(implicit ...) extends OpTransformer1[I, O]
```

This abstract class does not take the transformFn as a constructor parameter but requires that it be implemented in the body of the new class. By having the function in the body of the class it can access parameters defined within the class. All parameters should be written using the Spark Params syntax. This is important because it marks these vals as parameters so that they can be manipulated by the workflow in addition to direct set methods.

```scala
class TextTokenizer
(
  override val uid = UID[TextTokenizer]
) extends UnaryTransformer[Text, PickList](
  operationName = "tokenize"
  uid = uid
)(

  final val numWords = new IntParam(
    parent = this,
    name = "numWords",
    doc = s"number of words to include in each token, eg: 1, bigram, trigram",
    isValid = ParamValidators.inRange(
      lowerBound = 0, upperBound = 4,
      lowerInclusive = false, upperInclusive = false
    )
  )

  setDefault(numWords, 1)

  def setNumWords(v: Int): this.type = set(numWords, v)

  override def transformFn: (Text) => PickList = in => {
    $(numWords) match {
       case 1 => ...
       case 2 => ...
       case 3 => ...
  }
)
```

**Testing your transformer**

As part of TransmogrifAI Test Kit we provide a handy base class to test transformers: `OpTransformerSpec`. It includes checks that transformer's code & params are serializable, transformer transforms data & schema as expected, as well as metadata checks. Below is example of a transformer test:

```scala
@RunWith(classOf[JUnitRunner])
class UnaryTransformerTest extends OpTransformerSpec[Real, UnaryLambdaTransformer[Real, Real]] {

  /**
   * Input Dataset to transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq(Some(1), Some(2), Some(3), None).map(_.toReal))

  /**
   * [[OpTransformer]] instance to be tested
   */
  val transformer = new UnaryLambdaTransformer[Real, Real](
    operationName = "unary",
    transformFn = r => r.v.map(_ * 2.0).toReal
  ).setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult = Seq(Real(2), Real(4), Real(6), Real.empty)

  // Any additional tests you might have
  it should "do another thing" in {
    // your asserts here
  }
}
```

### Wrapping a SparkML transformer

Many of SparkML's transformers inherit from their [UnaryTransformer](https://spark.apache.org/docs/2.0.0/api/java/org/apache/spark/ml/UnaryTransformer.html), which is an abstract class for transformers that take one input column, apply a transformation, and output the result as a new column.  An example of such a transformer is [Normalizer](https://spark.apache.org/docs/2.0.0/api/java/org/apache/spark/ml/feature/Normalizer.html), which normalizes a vector to have unit norm using the given p-norm.  We can use the Normalizer to illustrate how to wrap a SparkML transformer that inherits from their UnaryTransformer:

```scala
val sparkNormalizer = new Normalizer().setP(1.0)

val wrappedNormalizer: OpUnaryTransformerWrapper[OpVector, OpVector, Normalizer] =
    new OpUnaryTransformerWrapper[OpVector, OpVector, Normalizer](sparkNormalizer)

val normalizedFeature: Feature[OpVector] = wrappedNormalizer
   .setInput(unNormalizedfeature).getOutput()
```

The flow illustrated above  instantiates and configures the transformer we aim to wrap, Normalizer, and then to passes it in as a constructor parameter to an TransmogrifAI transformer wrapper, in this case, OpUnaryTransformerWrapper.

The spark wrappers built into TransmogrifAI allow users to take advantage of any estimator or transformer in Spark ML, whether or not it has been explicitly wrapped for use within TransmogrifAI.


### Wrapping a non serializable external library

Sometimes there is a need to use an external library from within a transformer / estimator, but when trying to do so one gets a `Task not serializable: java.io.NotSerializableException` exception. That is quite common, especially when one or more of class instances you are trying to use are not `Serializable`. Luckily there is a simple trick to overcome this issue using a singleton and a function.

Let's assume we have a text processing library, such as Lucene, we would like to use to tokenize the text in our transformer, but it's `StandardAnalyzer` is not `Serializable`.

```scala
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.standard.StandardAnalyzer

// We create a object to hold the instance of the analyzer
// and an apply function to retrieve it
object MyTextAnalyzer {
  private val instance = new StandardAnalyzer
  def apply(): Analyzer = instance
}

// And then instead of using the analyzer instance directly
// we refer to a function in our transformer
class TextTokenizer[T <: Text]
(
  val analyzer: () => Analyzer = MyTextAnalyzer.apply
  uid: String = UID[TextTokenizer[_]]
)(implicit tti: TypeTag[T])
  extends UnaryTransformer[T, TextList](operationName = "txtToken", uid = uid) {

  override def transformFn: T => TextList = text => {
     // Now we safely retrieve the analyzer instance
     val tokenStream = analyzer().tokenStream(null, text)
     // process tokens etc.
  }

}
```

Note: as with any singleton object one should take care of thread safety. In the above example the `StandardAnalyzer` is thread safe, though there are scenarios where an additional coordination would be required.

Alternative solution is to create an instance of the desirable non serializable class on the fly (inside the `transformFn`). But only do so if its instance creation & destruction is lightweight, because `transformFn` is being called on per row basis.


## Estimators

Within the context of OpWorkflows Estimators and Transformers can be used interchangeably. However, the distinction between the two classes is important to understand for TransmogrifAI developers.

Estimators are a Spark ML concept that describes classes which use information contained in a column or columns to create a Transformer which will perform a map operation on the data. The important distinction between Estimators and Transformers is that Estimators have access to all the information in the columns while transformers only act within a row. For example, if you wish to normalize a numeric column you must first find the distribution of data for that column (using the information in all the rows) using an estimator. That estimator will then produce a transformer which contains the distribution to normalize over and performs a simple map operation on each row.

### TransmogrifAI Estimators

TransmogrifAI Estimators extend Spark Estimators but are designed to interact with Features rather than DataFrame column names. Rather than directly passing in the data to fit function, the input Feature (or Features) are set using the setInput method and the return Feature (or Features) are obtained using the getOutput method. Both the input and output Features can then be acted on as many times as desired to obtain the final result Features. When a Workflow is called, the data described by the output Feature is materialized by passing the data described by the input Features into the fit function and obtaining a Transformer which is then applied to the input data.

### Writing your own estimator

Like TransmogrifAI Transformers, TransmogrifAI Estimators can be defined by extending appropriate base classes. Again only the name of the operation and `fitFn,` the estimation function, need to be defined.

The possible base classes are, `UnaryEstimator` (one input),  `BinaryEstimator` (two inputs), `TernaryEstimator` (three inputs), `QuaternaryEstimator` (four inputs) and `SequenceEstimator` (multiple inputs of the same feature type). MultiStage Estimators can be used to return up to three outputs.

**Note**: that when creating a new stage (Transformer or Estimator) the uid should always be a constructor argument.** If the uid is not a constructor  the stages will not be serialized correctly when models are saved.

**Creating your own params**

For many Estimators it is useful to be able to parameterize the way the fit function is performed. For example in writing a  LogisticRegression Estimator one may wish to either fit the intercept or not. Such changes in the fit operation can be achieved by adding parameters to the Estimator. The Estimator will need to inherit from the *aryEstimator class and the companion model object will inherit from the *aryModel class. For instance, for a single input estimator the classes below should be extended: 

```scala
abstract class UnaryEstimator[I, O]
(
  val operationName: String,
  val uid: String
)(implicit ...) extends Estimator[UnaryModel[I, O]] with OpPipelineStage1[I, O]


abstract class UnaryModel[I, O]
(
  val operationName: String,
  val uid: String
)(implicit ...) extends Model[UnaryModel[I, O]] with OpTransformer1[I, O]
```

The `fitFn` cannot be a constructor parameter as `transformFn` was in LambdaTransformers, rather it is created in the body of the class. By having `fitFn`  implemented in the body of the class it can access parameters defined within the class. All parameters should be written using the Spark Params syntax. Using the Spark Params syntax allows the parameters to be set by the workflow as well as directly on the class and ensures that parameters are serialized correctly on save.

The `fitFn` must return an instance of the companion model class. Passing the values derived in the `fitFn` as constructor arguments into the companion model object allows these values to be serialized when the workflow is saved. *Note that the companion Model must take the Estimator uid as a constructor argument.*

```scala
class MinMaxScaler
(
  override val uid = UID[MinMaxScaler]
) extends UnaryEstimator[RealNN, RealNN](
  operationName = "min max scaler"
){
// Parameter
final val defaultScale = new DoubleParam(
    parent = this,
    name = "defaultScale",
    doc =  "default scale in case of min = max ",
    isValid = ParamValidators.gt(0.0)
 )
setDefault(defaultScale, 2.0)
def setDefaultScale(value: RealNN): this.type = set(defaultScale, value)

def fitFn(dataset: Dataset[I#Value]): UnaryModel[I, O] = {
        val grouped = data.groupBy()
        val maxData = grouped.max().first().getAs[Double](0)
        val minData = grouped.min().first().getAs[Double](0)
        val scale = if(minData == maxData) $(defaultScale) else maxData - minData
        new MinMaxModel(uid, minData, scale)
      }
}

class MinMaxModel(override val uid, val min: Double, val scale: Double)
 extends UnaryModel[RealNN, RealNN](uid, "min max scaler") {
  def transformFn: RealNN => RealNN =
   (input: RealNN) => { ((input.value - min) / scale).toRealNN }
}
```


**Testing your estimator**

As part of TransmogrifAI Test Kit we provide a handy base class to test estimators: `OpEstimatorSpec`. It includes checks that estimator's code & params are serializable, fitted model is of expected type and verifies that the model transforms data & schema as expected, as well as metadata checks. Below is example of an estimator test:

```scala
@RunWith(classOf[JUnitRunner])
class UnaryEstimatorTest extends OpEstimatorSpec[Real, UnaryModel[Real, Real], UnaryEstimator[Real, Real]] {

  /**
   * Input Dataset to fit & transform
   */
  val (inputData, f1) = TestFeatureBuilder(Seq(1.0, 5.0, 3.0, 2.0, 6.0).toReal)

  /**
   * Estimator instance to be tested
   */
  val estimator = new MinMaxNormEstimator().setInput(f1)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  val expectedResult = Seq(0.0, 0.8, 0.4, 0.2, 1.0).map(_.toReal)

  // Any additional tests you might have
  it should "do another thing" in {
    // your asserts here
  }
}
```

### Wrapping a SparkML estimator

To wrap a SparkML estimator, we follow a similar pattern as when wrapping a SparkML transformer.  SparkML estimators all inherit from [Estimator](https://spark.apache.org/docs/2.0.1/api/java/org/apache/spark/ml/Estimator.html), with a couple of specializations, like [Predictor](https://spark.apache.org/docs/2.0.1/api/java/org/apache/spark/ml/Predictor.html) and [ProbabilisticClassifier](https://spark.apache.org/docs/2.0.0/api/java/org/apache/spark/ml/classification/ProbabilisticClassifier.html).  We have wrapper classes for each of those, respectively: OpEstimatorWrapper, OpPredictorWrapper, and OpProbabilisticClassifierWrapper.  For example, to wrap SparkML's [LinearRegression](https://spark.apache.org/docs/2.0.2/api/java/org/apache/spark/ml/regression/LinearRegression.html) estimator, which inherits from Predictor, we proceed as follows:

```scala
val linReg = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFitIntercept(true)

val wrappedLinReg: OpPredictorWrapper[RealNN, RealNN, LinearRegression, LinearRegressionModel] =
   new OpPredictorWrapper[Numeric, Numeric, LinearRegression, LinearRegressionModel](linReg)
```

Basically, we instantiate and configure the estimator to be wrapped, LinearRegression, and then pass it in as a constructor parameter to an TransmogrifAI estimator wrapper, in this case, OpPredictorWrapper.

### Creating Shortcuts for Transformers and Estimators

One of the main benefits of having type information associated with Features is the ability to extend Feature operations syntax by adding shortcuts toTransformers/Estimators. Ultimately all Feature operations should be done using such shortcuts allowing cleaner and safer syntax.
 
For the sake of the example, let's assume we want to compute TF/IDF for the “name” text Feature. Which requires tokenizing the text, hashing it then computing the inverted document frequency. One can do as follows:

```scala
// Tokenize the "name"
val tokenized: Feature[PickList] = new TextTokenizer().setInput(name).getOutput()

// Then apply hashing transformer on tokenized "name"
val htf = new HashingTF().setNumOfFeatures(numFeatures).setBinary(binary)
val hashingTransformer = new OpTransformerWrapper[PickList, Vector, HashingTF](htf)
val hashed = hashingTransformer.setInput(tokenized).getOutput()

// Compute inverse document frequency
val idf = new IDF().setMinDocFreq(minDocFreq)
val idfEstimator = new OpEstimatorWrapper[Vector, Vector, IDF, IDFModel]()
val nameIdf: Feature[Vector] = idfEstimator.setInput(hashed).getOutput()
```

One can also add shortcuts to significantly shorten the code required and make the application of stages much clearer. Let's add a few shortcuts to allow us to perform the same transformations on features using [Implicit Classes](http://docs.scala-lang.org/overviews/core/implicit-classes.html).

```scala
// Implicit classes enrich each feature type,
// see com.salesforce.op.dsl package for more examples
implicit class RichTextFeature(val f: FeatureLike[Text]) {
    def tokenize(): FeatureLike[PickList] = new TextTokenizer()
       .setInput(f).getOutput()
}
implicit class RichPickListFeature(val f: FeatureLike[PickList]) {
   def tf(numFeatures: Int = 1 << 8, binary: Boolean = false): FeatureLike[Vector] = {
      val htf = new HashingTF().setNumFeatures(numFeatures).setBinary(binary)
      new OpTransformerWrapper[PickList, Vector, HashingTF](htf)
         .setInput(f).getOutput()
   }
   def tfidf(numFeatures: Int = 1 << 8, binary: Boolean = false,minDocFreq: Int = 0): FeatureLike[Vector] = {
      f.tf(numFeatures = numFeatures, binary = binary).idf(minDocFreq = minDocFreq)
   }
}
implicit class RichVectorFeature(val f: FeatureLike[Vector]) {
    def idf(minDocFreq: Int = 0): FeatureLike[Vector] = {
      val idf = new IDF().setMinDocFreq(minDocFreq)
      new OpEstimatorWrapper[Vector, Vector, IDF, IDFModel](idf)
         .setInput(f).getOutput()
    }
}
```

Once we defined the above implicit classes we can use them as follows:

```scala
// Once you have the implicits in scope you can write simply this
val nameIdf: Feature[Vector] = name.tokenize().tf().idf()
// Or even shorter
val nameIdf: Feature[Vector] = name.tokenize().tfidf(ocFreq)
```

Standard stages within TransmogrifAI have shortcuts that can be imported using:

```scala
import com.salesforce.op._
```

Shortcuts can also be created for custom stages and placed in a separate namespace.

### Shortcuts Naming Convention

When adding shortcuts one should follow the below conventions:

1. For Binary, Ternary etc. Transformers/Estimators one should add a concrete class **AND** a shortcut - `f1.operand(f2, f3, ..., options)`
2. For Unary and Sequences Transformers/Estimators one should add **AT LEAST** a shortcut - `f1.operand(options)` and `Seq(f1, f2, ...).operand(options)` accordingly.
3. For verbs shortcuts should be: `f.tokenize`, `f.pivot`, `f.normalize`, `f.calibrate`, etc.
4. For nouns one should prepend “to”, i.e `f.toPercentiles` etc.
5. For adjectives one should prepend “is”, i.e  `f.isValid`, `f.isDividable` etc.

## Customizing AutoML Stages

Each of the special [AutoML Estimators](/AutoML-Capabilities) we talked about previously can be be customized. 

#### Transmogrification

Automatic feature engineering can be customized completely by manual feature engineering and manual vector combination using the VectorsCombiner Transformer (short-hand ```.combine()```) if the user desires to have complete control over feature engineering.

```scala
val normedAge = age.fillMissingWithMean().zNormalize()
val ageGroup = age.map[PickList](_.value.map(v => if (v > 18) "adult" else "child").toPickList).pivot() 
val combinedFeature = Seq(normedAge, ageGroup).combine()
```

#### SanityChecker 

You can override defaults in the sanity checker params like so:
```scala
// Add sanity checker estimator

val checkedFeatures = new SanityChecker()
      .setMaxCorrelation(0.99)
      .setMinVariance(0.00001)
      .setCheckSample(1.0)
      .setRemoveBadFeatures(true)
      .setInput(label, features) // survived: response, passengerFeatures: transformed predictor features 
      .getOutput()
```
After the Sanity Checker has been fitted, one can access the metadata which summarizes the information used for feature selection.

```scala
val metadata = fittedWorkflow.getOriginStageOf(checkedFeatures).getMetadata()

```

The metadata contains a summary of the sanity checker run and can be cast into a [case class](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/stages/impl/preparators/SanityCheckerMetadata.scala#L74) containing this info for easy use:

```scala
val summaryData = SanityCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
```

The summary is composed as follows : 


* “featureStatistics” (SummaryStatistics) : Descriptive statistics of the inputs (label and features) :
    - “sampleFraction” (Double) : corresponds to the parameter `checkSample` of the sanity checker. It is the downsample fraction of the data.
    - “count” (Double) : size of the dataset
    - “variance” (Array[Double]) : variance of columns
    - “mean” (Array[Double]) : mean of each column
    - “min” (Array[Double]) : minimum of each column
    - “max” (Array[Double]) : maximum of each column
    - “numNull” (Array[Double]) : number of former missing elements for each column
* “names” (Array[String]) : names of label and features columns
* "dropped" (Array[String]) : names of the feature columns dropped
* "correlationsWLabel" (Correlations) : info about valid correlation (i.e. non Nan) of features with the labels column :
    - “values” (Array[Double]) : value of valid correlations features/label
    - “featuresIn” (Array[String]) : name of feature columns with non Nan correlations
* "correlationsWLabelIsNaN" (Array[String]) : names of the features that have a correlation of Nan with label column

In order to relate the statistics summary from sanity checker to the original parent features it is best to use the `workflowModel.modelInsights(feature)` [method](/Developer-Guide#extracting-modelinsights-from-a-fitted-workflow). This will output all the information gathered during workflow fitting formatted so that all feature statistics are grouped by the raw parent feature.


#### RawFeatureFilter

[RawFeatureFilter](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/filters/RawFeatureFilter.scala) is an optional stage that would ensure that the data distribution between the training and scoring set is similar. [Workflows](Developer-Guide#workflows) has `withRawFeatureFilter(Option(trainReader), Option(scoreReader),...)` method which enables this. When the scoring reader is specified both the `readerParams` and the `alternateReaderParams` in the [OpParams](https://github.com/salesforce/op/blob/master/features/src/main/scala/com/salesforce/op/OpParams.scala) passed into the Workflow need contain paths for the data. You will need to set the score data path in the `alternateReaderParams`.  

If only the training reader is specified the features will be checked for fill rates and correlation of filled values with the label. When both the training and scoring readers are specified the relationship between the two data sets is additionally checked for each raw feature and features which deviate beyond the specified acceptable range will be excluded from the workflow. The exclusion criteria have defaults, however you can set optional parameters for when to exclude features.

```scala
// Add raw feature filter estimator
val workflow = new OpWorkflow().setResultFeatures(survived, rawPrediction, prob, prediction)
      .withRawFeatureFilter(
        trainingReader = Option(trainReader),
        scoringReader = Option(scoringReader),
        // optional params below:
        bins = 100,
        minFillRate = 0.001,
        maxFillDifference = 0.90,
        maxFillRatioDiff = 20.0,
        maxJSDivergence = 0.90,
        maxCorrelation = 0.95,
        correlationType = CorrelationType.Pearson,
        protectedFeatures = Array.empty[OPFeature]
      )
```


#### ModelSelector

It is possible to set validation parameters such as the number of folds  or the evaluation metric (`AUROC` or `AUPR`) :

```scala
val modelSelector = BinaryClassificationModelSelector
 .withCrossValidation(numFolds = 10, validationMetric = Evaluators.BinaryClassification.auROC)
 .setModelsToTry(LogisticRegression, RandomForest)
 .setInput(survived, passengerFeatures)
```

Before evaluating each model,  it is possible for the BinaryClassificationModelSelector to balance the dataset by oversampling the minority class and undersampling the majority class. The user can decide on the balancing by for instance setting the targeted  proportion for the minority class the balanced dataset : 

```scala
val modelSelector = BinaryClassificationModelSelector
 .withCrossValidation(Option(DataBalancer(sampleFraction = 0.2)), numFolds = 10, validationMetric = Evaluators.BinaryClassification.auROC)
 .setModelsToTry(LogisticRegression, RandomForest)
 .setInput(survived, passengerFeatures)
```

Finally, each potential model can have its hyper-parameters tuned for the model selection. In case of setting multiple values for the same parameter,  ModelSelector will add a grid that will be used during Cross Validation via grid search: 

```scala
modelSelector.setLogisticRegressionRegParam(0.1)
             .setLogisticRegressionMaxIter(10, 100)
             .setRandomForestMaxDepth(2, 5, 10)
             .setRandomForestNumTrees(10)
```


The return type of the BinaryClassificationModelSelector (and MultiClassificationModelSelector) is a triplet of numeric, vector, and vector features representing the prediction, raw prediction and normalized raw prediction (or probability) columns outputted by the model selector once it is fit and applied to raw data.

```scala
val (pred, rawPred, prob): (Feature[RealNN], Feature[OpVector], Feature[OpVector]) =
    modelSelector.getOutput()
```

## Interoperability with SparkML

All TransmogrifAI Stages can be used as spark ML stages by passing a Dataset or DataFrame directly into the `.fit()` or `.transform()` method. The important thing to note when using stages in this manner is that the **names of the input features for the Stage must match the names of the column** you wish to act on.


## Workflows

Workflows are used to control the execution of the ML pipeline once the final features have been defined. Each Feature contains the history of how it is defined by tracking both the parent Features and the parent Stages. However, this is simply a *description* of how the raw data will be transformed in order to create the final data desired, until the Features are put into a Workflow there is no actual data associated with the pipeline. OpWorkflows create and transform the raw data needed to compute Features fed into them. In addition they optimize the application of Stages needed to create the final Features ensuring optimal computations within the full pipeline DAG. OpWorkflows can be fit to a given dataset using the `.train()` method. This produces and OpWorkflowModel which can then be saved to disk and applied to another dataset.

### Creating A Workflow

In order to create a Workflow that can be used to generate the desired features the result Features and the Reader (or data source) must be defined.

```scala
// Workflow definition with a reader (readers are used to manipulate data before the pipeline)
val trainDataReader = DataReaders.Simple.avro[Passenger](key = _.passengerId)

val workflow = new OpWorkflow()
   .setResultFeatures(prediction)
   .setReader(trainDataReader)
   
// Workflow definition by passing data in directly
val workflow = new OpWorkflow()
   .setResultFeatures(prediction)
   .setInputDataSet[Passenger](passengerDataSet) // passengerDataSet is a DataSet[Passenger] or RDD[Passenger]
```

DataReaders are used to load and process data before entry into the workflow, for example aggregation of data or joining of multiple data sources can easily be performed using DataReaders as described in the [DataReaders](/Developer-Guide#datareaders) section below. If you have a dataset already loaded and simply wish to pass it into the Workflow the `setInputDataSet` and `setInputRdd` methods will create a simple DataReader for you to allow this.

It is important to understand that up until this point nothing has happened. While all the Features, Stages (transformers + estimators), and data source have been defined, none of the actual data associated with the  features has been computed. Computation does not happen and Features are not materialized until the Workflow is fitted.

### Fitting a Workflow

When a workflow gets fitted 

```scala
val model: OpWorkflowModel = workflow.train()
```

a number of things happen: the data is read using the DataReader, raw Features are built, each Stage is executed in sequence and all Features are materialized and added to the underlying Dataframe. During Stage execution, each Estimator gets fitted and becomes a Transformer. A fitted Workflow (eg. a WorkflowModel) therefore contains sequence of Transformers (map operations) which can be applied to any input data of the appropriate type. 

### Fitted Workflows

A WorkflowModel, much like a Spark Pipeline, can be used to score data ingested from any DataReader that returns the appropriate data input type. If the score method is called immediately after train, it will score the data used to train the model. Updating the DataReader_ _allows scoring on a test set or on production data.

```scala
val trainDataReader = DataReaders.Aggregate.avro[Passenger](key = _.passengerId, path = Some("my/train/data/path")) // aggregates data across many records
val workflowModel = new OpWorkflow()
   .setResultFeatures(prediction)
   .setReader(trainDataReader)
   .train()
val scoredTrainData = workflowModel.score()

val testDataReader = DataReaders.Simple.avro[Passenger](key = _.passengerId, path = Some("my/test/data/path")) // simply reads and returns data
val scoredTestData = workflowModel.setReader(testDataReader).score()
```

The `score` method will execute the entire pipeline up to the final result Feature and return a DataFrame containing the result features.

 It is also possible to compute and return the DataFrame up to any intermediate feature (for debugging workflows). This call takes the Feature that you wish to compute up to as an input and returns a DataFrame containing all raw and intermediate Features computed up to the level of the DAG of that Feature (note that the result features and reader must be set on the workflow in order to define the DA.

```scala
// same method will work on workflows and workflowModels
val df = workflow.computeDataUpTo(normedAge)
val df = workflowModel.computeDataUpTo(normedAge)
```

Here it is important to realize that when models are shared across applications, or namespaces, all Features must be within the scope of the calling application. A user developing a model with the intention of sharing it must take care to properly define and expose all Features.

### Saving Workflows

Fitted workflows can be saved to a file which can be reloaded and used to score data at a later time. The typical use case is for one run to train a workflow, while another run will use this fitted workflow for scoring. Saving a workflow requires the use of dedicated save method.

```scala
val workflowModel: OpWorkflowModel = workflow.train()
workflowModel.save(path = "/my/model/location", overwrite = true)
```

The saved workflow model consists of several json files that contain all the information required to reconstruct the model for later scoring and evaluation runs. We store the computed metadata, information on features, transformers and fitted estimators with their Spark parameters.

For features we store their type, name, response or predictor, origin stage UID and parent feature names.

Persisting transfomer stages is trivial. We save transformer's class name, UID and Spark param values provided during training.

Estimators are a bit trickier. We save the class name of the estimator model, UID, constructor arguments and Spark param values provided during training. Using the class name and the constructor arguments we then able to reconctruct the instance of the estimator model and set the Spark param values.


### Loading saved Workflows

Just like saving a workflow, loading them requires the use of dedicated load method on the workflow. Note that you have to use the exact same workflow that was used during training, otherwise expect errors.

```scala
// this is the workflow instance we trained before
val workflow = ...
// load the model for the previosely trained workflow
val workflowModel: OpWorkflowModel = workflow.loadModel(path = "/my/model/location")
```

When loading a model we match the saved features and stages to the ones provided in the workflow instance using their UIDs. For each transformer we simply set its Spark params. The estimator models are contructred using reflection using their class names and costructor arguments, and finally their Spark params are set. The operation of assigning UIDs is based on a process global state, therefore loading models is not a thread safe operation. Meaning if you are going to load models in a multi-threaded program make sure to synchronize access to `loadModel` call accordingly.

Once loaded, the model can be modified and operated on normally. It is important to note that readers are not saved with the rest of the OpWorkflowModel, so in order to use a loaded model for scoring a DataReader must be assigned to the OpWorkflowModel.

```scala
val results = workflowModel
  .setReader(scoringReader) // set the reader 
  .setParameters(opParams)  // set the OpParams
  .score()                  // run scoring or evaluation
```

### Removing problematic features

One of the fundamental assumptions of machine learning is that the data you are using to train your model reflects the data that you wish to score. TransmogrifAI has an optional stage after data reading that allows to to check that your features do not violate this assumption and remove any features that do. This stage is called the [RawFeatureFilter](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/filters/RawFeatureFilter.scala), and to use it you simply call the method `withRawFeatureFilter(Some(trainReader), Some(scoreReader),...)` on your [Workflow](https://github.com/salesforce/op/blob/master/core/src/main/scala/com/salesforce/op/OpWorkflow.scala#L360). This method takes the training and scoring data readers as well as some optional settings for when to exclude features (If you specify the data path in the [OpParams](https://github.com/salesforce/op/blob/master/features/src/main/scala/com/salesforce/op/OpParams.scala) passed into the Workflow you will need to set the score data path in the `alternateReaderParams`).  It will load the training and scoring data and exclude individual features based on fill rate, retaliative fill rate between training ans scoring, or differences in the distribution of data between training and scoring. Features that are excluded based on these criteria will be blacklisted from the model and removed from training.

This stage can eliminate many issues, such as leakage of information that is only filled out after the label and changes in data collection practices, before they effect your model. In addition because this is applied immediately after feature extraction it can greatly improve performance if there are many poor features int the data. Addition issues with features can be detected by the [SanityChecker](https://github.com/salesforce/TransmogrifAI/wiki/AutoML-Stages#sanitychecker), however these checks occur on features that have undergone feature engineering steps and so often detect different kinds of data issues.

### Extracting ModelInsights from a Fitted Workflow

Once you have fit your Workflow you often wish to examine the results of the fit to evaluate whether your should use the workflow going forward. We provide two mechanisms for examining the results of the workflow.

The first is the `.summary()` (or `summaryJson()`) method that pulls all [metadata](/Developer-Guide#metadata) generated by the stages in the workflow into a string (or JSON) for consumption.

The second mechanism is the `modelInsights(feature)` method. This method take the feature you which to get a model history for (necessary since multiple models can be run in the same workflow) and extracts as much information as possible about the modeling process for that feature, returning a [ModelInsights](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/ModelInsights.scala) object. This method traces the history of the input feature (which should be the output of the model of interest) to find the last ModelSelector stage aplied and the last SanityChecker stage applied to the feature vector that went into creating the model output feature. It collects the metadata from all of the stages and the feature vector to create a summary of all of the information collected in these stages and groups the information so that all feature information can be traced to the raw input features used in modeling. It also contains the training parameters and stage parameters used in training for reference across runs.

### Extracting a Particular Stage from a Fitted Workflow

Sometimes it is necessary to obtain information about how a particular stage in a workflow was fit or reuse that stage for another workflow or manipulation. The syntax to support these operations is shown below.

```scala
val indexedLabel: Feature[RealNN] = new OpStringIndexerNoFilter().setInput(label).getOutput

// .... some manipulations that combine your features with the indexed label
// to give a finalScore ...

val fittedLeadWorkflow = new OpWorkflow()
    .setResultFeatures(finalScore)
    .setReader(myReader)
    .train()

// if you want to use just the fitted label indexer from the model or extract information
// from that fitted stage, you can use the getOriginStageOf method on the fitted WorkflowModel
val labelIndexer = fittedLeadWorkflow
.getOriginStageOf(indexedLabel).asInstanceOf[OpStringIndexerNoFilter] // 


// if you want to create a new feature that uses the "fitted" indexer you can use it as follows
val indexedLabel2: Feature[Numeric] = labelIndexer.setInput(newLabel).getOutput()
```

The Feature created by the Stage can be used as a handle to retrieve the specific stage from the fitted DAG. The returned Stage will be of type `OpPipelineStage` and so must be cast back into the original Stage type for full utility.


### Adding new features to a fitted workflow

If you wish use stages from model that was perviously fit (for example to recalibrate the final score on a separate dataset) it is possible to add the fitted stages to new Workflow with the following syntax:

```scala
val fittedModel = ... // a fitted workflow containing model stages that you wish to re-use without refitting to the new dataset
val workflow = new OpWorkflow().setResultFeatures(calibration).setReader(calibrationReader)
val newWorkflow = workflow.withModelStages(fittedModel)
```


This will add the stages from the model to the Workflow replacing any estimators that have corresponding fitted models in the workflow with the fitted version. The two workflows and all their stages and features must be created in the same workspace as only directly matching stages will be replaced. When train is called on this Workflow only Estimators that did NOT appear in the previous DAG will be fit in order to create the new WorkflowModel. All stages that appeared in the original WorkflowModel will use the fit values obtained in the first fit (corresponding to the first dataset).

### Metadata

Metadata is used to enhance the schema of a given feature (column in a dataset) with additional information not contained elsewhere.  For example, if a given feature is of type OpVector, we still need a way to specify/query the names and history of the features used to create the columns within the vector.  In addition, Metadata is also used to store information obtained during Estimator fitting which may useful later but is not part of the resulting Transformer. An example of this is the SanityChecker, which stores in the Metadata field diagnostic information related to data leakage analysis and feature statistics.

In cases where the Metadata will be frequently used, as described above, it is nice to create a case class to contain the Metadata information. Metadata itself is a spark class with an underlying structure of `Map[String, Any]` and remembering the exact structure for complex information sets is error prone. The case class can contain all of the neccessary information and contain a `.toMetadata()` function with a companion object containing a `fromMetadata(meta: Metadata)` function to convert back and forth. The Metadata itself is saved as a parameter on the stage `.setMetadata(meta: Metadata)` as well as being written into the schema of the output dataframe. This ensures that the information is retained and passed along with the created features.

In order to use the the OpVector Metadata one can then pull the Metadata off the stage or dataframe and convert it to the corresponding [case class](https://github.com/salesforce/TransmogrifAI/blob/master/utils/src/main/scala/com/salesforce/op/utils/spark/OpVectorMetadata.scala).

```scala
val metadata = workflowModel.getOriginStageOf(featureVector).getMetadata()
val vectorHistory = OpVectorMetadata(featureVector.name, metadata).getColumnHistory()
val dataset = workflowModel.score()
val vectorHistory2 = OpVectorMetadata(dataset.schema(featureVector.name)).getColumnHistory()
```


 The metadata from the SanityChecker is stored as summary Metadata under a special key to allow such information to be added to any features Metadata. It can also be cast to it's corresponding [case class](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/stages/impl/preparators/SanityCheckerMetadata.scala) for ease of use. In the example below `checkedFeatures` is the feature vector output of the SanityChecker.

```scala
val metadata = workflowModel.getOriginStageOf(checkedFeatures).getMetadata()
val summaryData = SanityCheckerSummary.fromMetadata(metadata.getSummaryMetadata())
```

If you wish to combine the metadata from stages commonly used in modeling (ModelSelectors, SanityCheckers, Vectorizers) into a single easy(er) to reference case class we have provided a method for this in the [WorkflowModel](/Developer-Guide#extracting-modelInsights-from-a-fitted-workflow) so that users don't need to stitch this information together for themselves.

In addition to accessing Metadata that is created by stages you may wish to add Metadata to stages of your own. For example if you created your own string indexer to map strings to integers (though we have a stage that does [this](https://github.com/salesforce/TransmogrifAI/blob/master/core/src/main/scala/com/salesforce/op/stages/impl/feature/OpStringIndexerNoFilter.scala)), you might wish to save the mapping from idex back to string in the Metadata of the new column of integers you are creating. You would do this within the `fitFn` of the Estimator you are creating by using the `setMetadata(meta: Metadata)` method. You need a MetadataBuilder object to work with Metadata, which is essentially a wrapper around a Map of Map.  For example, within an Estimator you would get a reference to a MetadataBuilder and use it as follows:

```scala
// get a reference to the current metadata
val preExistingMetadata = getMetadata()

// create a new metadataBuilder and seed it with the current metadata
val metaDataBuilder = new MetadataBuilder().withMetadata(preExistingMetadata)

// add a new key value pair to the metadata (key is a string,
// and value is a string array)
metaDataBuilder.putStringArray("Labels", labelMap.keys.toArray)
metaDataBuilder.putLongArray("Integers", labelMap.values.map(_.toLong).toArray) // Metadata supports longs not ints
 
// package the new metadata, which includes the preExistingMetadata 
// and the updates/additions
val updatedMetadata = metaDataBuilder.build()

// save the updatedMetadata to the outputMetadata parameter                                                            
setMetadata(updatedMetadata)
```

This metadata can be accessed later  in various ways, for example, as part of a fitted model, by calling the model's getMetadata method:

```scala
val model = labelIndexer.fit(label)
val metaData = model.getMetadata()
```

We provide utility functions to simplify working with Metadata in [com.salesforce.op.utils.spark.RichMetadata](https://github.com/salesforce/TransmogrifAI/blob/master/utils/src/main/scala/com/salesforce/op/utils/spark/RichMetadata.scala). For example we have functions to add and get summary Metadata which are used in the workflow to log any information that has been saved as summary metadata.


## DataReaders 

 
DataReaders define how data should be loaded into the workflow. They load and process raw data to produce the Dataframe used by the workflow. DataReaders are tied to a specific data source with the type of the raw loaded data (for example the AVRO schema or a case class describing the columns in a CSV).

There are three types of DataReaders. [Simple DataReaders](/Developer-Guide/#datareaders) just load the data and return a DataFrame with one row for each row of data read. [Aggregate DataReaders](/Developer-Guide#aggregate-data-readers) will group the data by the entity (the thing you are scoring) key and combine values (with or without time filters) based on the aggregation function associated with each feature definition. For example aggregate readers can be used to compute features like total spend from a list of transactions. [Conditional DataReaders](Developer-Guide/#conditional-data-readers) are like aggregate readers but they allow an daynamic time cuttoff for each row that depends on fullfilment of a user defined condition. For example conditional readers can be used to compute features like total spend before a user becomes a member. These readers can be combined to [join](/examples/Time-Series-Aggregates-and-Joins.html) multiple datasources.

A constructor object provides shortcuts for defining most commonly used data readers. Defiing a data reader requires specifying the type of the data being read and the key for the data (the entity being scored).


```scala
val trainDataReader = DataReaders.Simple.avro[Passenger](
    key = _.passengerId
)
```


The data reader is often a good place for specifying pre-processing, e.g., filtering, that you would like applied to the data before features are extracted and transformed via the workflow. If you have specific types pre-processing steps to add to a reader this can be added by creating your own reader and overriding the read method used to load the data. The results of this read method are passed into the function that extracts the features from the input data to create the required Datafame for the workflow.


```scala
val trainDataReader = new DataReader[PassengerCensus](
    key = _.passengerId) {

    overrride protected def read(params: OpParams)(implicit spark: SparkSession): Either[RDD[PassengerCensus], Dataset[PassengerCensus]]
           // method that reads Passenger data & Census data and joins
           // and filters them to produce PassengerCensus data
    }

```



### Aggregate Data Readers

Aggregate data readers should be used when raw features extracted for each key need to be aggregated with respect to a particular point in time. It is easier to think about these kinds of features in the context of event data. Consider a table of events such as website visits, and imagine that we would like to extract features such as the number of times a visitor visited the site in the past 7 days. Here is the combination of FeatureBuilders and DataReaders that you would use to extract this sort of feature:

```scala
val numVisitsLast7Days = FeatureBuilder.Numeric[Visit]
    .extract(_ => 1)
    .aggregate(Sum)
    .daysAgo(7)
    .asPredictor

val dataReader = new AggregateDataReader[Visit](
    key = _.userId,
    aggregateParams = AggregateParams (
        timeStampFn = _.timeStamp
        cutOffTime = CutOffTime.UnixEpoch(1471046600)
    )
)
```

The timeStampFn in the aggregateParams specifies how the timestamps of the events are to be extracted, and the cutOffTime specifies the timestamp with respect to which the features are to be aggregated. All predictor features will be aggregated up until the cutOffTime, and all response features will be aggregated from the time following the cutOffTime. This kind of reader is useful for training a model to predict that an event will occur in a certain time window, for instance.

### Conditional Data Readers

Sometimes, when estimating conditional probabilities over event data, features need to be aggregated with respect to the occurrence of a particular event, and the time of occurrence of this event may vary from key to key in the data set.  This is when you would use a Conditional Data Reader. Continuing with our example of website visits, suppose we are attempting to estimate the likelihood of a visitor making  a purchase on the website after he makes a search, and one of the features we would like to use in the prediction is the number of times he visited the website before making the search. Here are the corresponding FeatureBuilders and Data Readers that would be needed:

```scala
val numVisitsLast7Days = FeatureBuilder.Numeric[Visit]
    .extract(_ => 1)
    .aggregate(Sum)
    .daysAgo(7)
    .asPredictor

val willPurchase = FeatureBuilder.Binary[Visit]
    .extract(_ => _.madePurchase)
    .aggregate(OR)
    .asResponse

val dataReader = new ConditionalDataReader[Visit](
    key = _.userId
    conditionalParams = ConditionalParams(
        timeStampFn = Option(_.timeStamp), // function for extracting
                                           // timestamp of event
        targetCondition = _.madeSearch, // function to figure out if
                                        // target event has occurred
        responseWindow = Some(Duration.standardDays(30)), // how many days after 
                                                          // target event to include
                                                          // in response aggregation
        predictorWindow = Some(Duration.standardDays(30)), // how many days before
                                                           // target event to include
                                                           // in predictor aggregation
        timeStampToKeep = TimeStampToKeep.Min // if a particular key met the
                                              // condition multiple times, which
                                              // of the instances should be kept
                                              // in the training set
    )
)
```

Using this reader in a workflow will ensure that for every visitor, we extract features relative to the first time he did a search. The predictor features are aggregated from a 30 day window preceding the search, and the response features are aggregated from a 30 day window succeeding the search. Each individual feature can override this value and be aggregated based on the time span specified in the FeatureBuilder. 

### Joined Data Readers

Sometimes it is necessary to read data from multiple locations and combine it in order to create all the desired features. While you can always apply any data processing logic in the read method of your data reader, the preferred approach for joining data sources is to use a joined data reader:

```scala
val joinedDataReader = passengerDataReader.leftOuterJoin(shipInfoDataReader)
```

Joined data readers allow your raw FeatureBuilders to be defined with respect to the simpler base types rather than the complex joint types.

Inner, left outer and full outer joins are supported. Joins will by default use the keys specified in the reader to join the data sources. However, it is possible to specifiy an [alternative key](https://github.com/salesforce/TransmogrifAI/blob/master/readers/src/main/scala/com/salesforce/op/readers/JoinedDataReader.scala#L58) to join on for one of the tables, e.g. if you need to aggregate on a key other than the key you need to join on. Joins are done after feature extraction for each of the datasources.

Sometimes it is important to aggreagte feature information after the join has been performed, e.g. you aggreagte only after an event in the first table has occured. We call this secondary aggreagtion and the most common use cases are supported by joined reasers. If a second aggregation phase is required it can be added using the JoinedReader method: 

```scala
 def withSecondaryAggregation(timeFilter: TimeBasedFilter): JoinedAggregateDataReader[T, U]
```


 This will produce a reader that joins the data and then performs an aggregation after the join. The secondary aggregation will use the aggregators defined in the feature builders. The secondary aggreagtion will only occur on the right table unless the join keys are the primary key for both tables. 

 The results of a joined reader can be used for futher joins as desired:

```scala
  reader1.leftJoin(reader2).withSecondayAggreagtion(timeFilter).innerJoin(reader3)
```


## Evaluators

Like in Spark MLlib, TransmogrifAI comes with a bunch of metrics that can be used to evaluate predictions on data. TransmogrifAI Evaluators can return one or many metrics. User can also create their own evaluators.

### Evaluators Factory

Evaluators are stored in a Factory that contains 3 categories : 

* _Binary Classification Evaluators_ : contains metrics such as AUROC, AUPR, Precision, Recall, F1-score,  Error rate, True Positive, True Negative, False Positive and False Negative.
    `Evaluators.BinaryClassification`
* _Multi Classification Evaluators_ : contains metrics like (weighted) Precision, (weighted) Recall, (weighted)  F1-score and Error rate.  
    `Evaluators.MultiClassification`
* _Regression Evaluators_ : with metrics RMSE, MSE, MAE and R2.
    `Evaluators.Regression`

### Single Evaluation

Like in Spark MLib's evaluators, TransmogrifAI evaluators have a `evaluate`  method that returns one metric. As example, let's compute the AUROC of a transformed dataset `transformedData`  with features `label`, `pred`, `rawPred` and `prob`:

```scala
val evaluator = Evaluators.BinaryClassification.auROC().setLabelCol(label).setRawPredictionCol(rawPred)
val metric = evaluator.evaluate(transformedData)
```

### Multiple Evaluation

TransmogrifAI Evaluator can also evaluate multiple metrics using `evaluateAll`. For a Regression prediction:

```scala
val evaluator = Evaluators.Regression().setLabelCol(label).setPredictionCol(pred)
val metrics = evaluator.evaluateAll(transformedData)
```

`metrics` is a case class that contains  the following metrics : RMSE, MSE, R2, MAE. For instance, to have access to the RMSE : 

```scala
val rmse = metrics.RootMeanSquaredError
```

### Creating a custom evaluator

Users can define their own custom evaluators too. As an example, let's create an multi classifcation evaluator that returns the cross entropy.

```scala
val crossEntropy = Evaluators.MultiClassification.custom(
      metricName = "cross entropy",
      isLargerBetter = false,
      evaluateFn = ds => ds.map { case (lbl, _, prob, _) => -math.log(prob.toArray(lbl.toInt)) }.reduce(_ + _)
    )
val evaluator = crossEntropy.setLabelCol(lbl).setRawPredictionCol(rawPrediction).setPredictionCol(prediction).setProbabilityCol(prob)

```

The field `isLargerBetter` is the same as the one in Spark's evaluators. It indicates whether the metric returned  should be maximized.
The method `evaluateFn`  takes a Dataset with columns `(label, rawPrediction, prediction, probability)` then returns the custom metric.
Base classes `OpEvaluatorBase`, `OpBinaryClassificationEvaluatorBase`,
 `OpMultiClassificationEvaluatorBase` and `OpRegressionEvaluatorBase`  are also available to define custom evaluators.

## TransmogrifAI App and Runner

Workflows can be run as spark applications using the `OpAppWithRunner` (or `OpApp`). Extend the `OpAppWithRunner` base class and define your workflow object and the training and testing data readers and then your workflow can be run as a spark app. Command line arguments are then used to specify the type of run you wish to do on the workflow (train, score, generate features, or evaluate).

```scala
// a simpler version with just a runner
object MyApp extends OpAppWithRunner {
  // build your workflow here
  def runner(opParams: OpParams): OpWorkflowRunner = new OpWorkflowRunner(/* workflow */)
}

// or a more customizable version
object MyApp extends OpApp {
  /**
   * The main function to run your [[OpWorkflow]].
   * The simplest way is to create an [[OpWorkflowRunner]] and run it.
   *
   * @param runType  run type
   * @param opParams op params
   * @param spark    spark session
   */
  def run(runType: OpWorkflowRunType, opParams: OpParams)(implicit spark: SparkSession): Unit = {
    // build your workflow here and then run it
    val result = new OpWorkflowRunner(/* workflow */).run(runType, opParams)
    // process the results here
  }
}
```

All the other pieces, such as `kryoRegistrator`, `sparkConf`, `appName`, `sparkSession`, `parseArgs` etc, are also easily customizable by overriding.

### Parameter Injection Into Workflows and Workflow Runners

```scala
class OpParams
(
  val stageParams: Map[String, Map[String, Any]],
  val readerParams: Map[String, ReaderParams],
  val modelLocation: Option[String],
  val writeLocation: Option[String],
  val metricsLocation: Option[String],
  val metricsCompress: Option[Boolean],
  val metricsCodec: Option[String],
  val customTagName: Option[String],
  val customTagValue: Option[String],
  val logStageMetrics: Option[Boolean],
  val customParams: Map[String, Any]
)
```

In addition to defining the reader and the final feature to “materialize,” OpWorkflows also allow the user to update/change the parameters for any stage within the workflow. Recall that stage state variables are stored as Spark Param key-value objects.

```scala
// Map of parameters to inject into stages. 
// Format is Map(StageSimpleName -> Map(ParameterName -> Value)).
workflow.setParameters(new OpParams(stageParams = ("MyTopKStage" -> ("TopK" -> 10))) 
```

Here we are resetting the “TopK” parameter of a stage with class name “MyTopKStage” to a value of 10. Caution must be exercised here because valid settings are not checked until runtime. This method is primarily designed to be used with CLI and Scheduler services as a means of injecting parameters values.


***


.. toctree::
   :maxdepth: 2