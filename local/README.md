# TransmogrifAI Local

This module enables local scoring with TransmogrifAI models without the need for Spark Session during scoring.
Instead it implementes local inference by applying TransmogrifAI's transformers and [MLeap](https://github.com/combust/mleap) runtime on JVM. It delivers unprecedented portability and performance of TransmogrifAI models allowing the serving of scores from any JVM process.

## Usage

Add the `transmogrifai-local` dependency into your project.

For Gradle in `build.gradle` add:
```gradle
dependencies {
    compile 'com.salesforce.transmogrifai:transmogrifai-local_2.11:0.5.1'
}
```
For SBT in `build.sbt` add:
```sbt
libraryDependencies += "com.salesforce.transmogrifai" %% "transmogrifai-local" % "0.5.1"
```

Then in your code you may load and score models as follows:
```scala
import com.salesforce.op.local._

// Spark Session needed for model loading & score function creation
implicit val spark = SparkSession.builder().getOrCreate()

// Create your workflow & load the model
val workflow: OpWorkflow = ...
val model = workflow.loadModel("/path/to/model")

// Create score function once and use it indefinitely
val scoreFn = model.scoreFunction

// Spark Session can be stopped now since it's not required during local scoring
spark.stop()

// Compute scores with score function
val rawData = Seq(Map("name" -> "Peter", "age" -> 18), Map("name" -> "John", "age" -> 23))
val scores = rawData.map(scoreFn)
```

Or using the local runner:
```scala
val scoreFn = new OpWorkflowRunnerLocal(workflow).scoreFunction(opParams)
```
**Note**: *Spark Session is only required for loading the model & preparing the scoring function. Once scoring function is returned the Spark Session can be shutdown since it's not required during local scoring.*

## Performance Results

Below is an example of measured scoring performance on a sample with 10 fields and 12 transformations applied.
Executed on MacBook Pro i7 3.5Ghz in a single thread.
```
Scored 6,000,000 records in 202s
Average time per record: 0.0336ms
```
