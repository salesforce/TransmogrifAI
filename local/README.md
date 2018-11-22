# TransmogrifAI Local

This module enables local scoring with TransmogrifAI models without the need for a Spark session.
Instead it uses [Aardpfark](https://github.com/CODAIT/aardpfark) and [Hadrian](https://github.com/opendatagroup/hadrian)
runtime for JVM. It delivers unprecedented portability and performance of TransmogrifAI models
allowing the serving of scores from any JVM process.

## Usage

Add the `transmogrifai-local` dependency into your project together with `hadrian` for runtime.

For Gradle in `build.gradle` add:
```gradle
dependencies {
    compile 'com.salesforce.transmogrifai:transmogrifai-local_2.11:0.5.0'
    runtime 'com.opendatagroup:hadrian:0.8.5'
}
```
For SBT in `build.sbt` add:
```sbt
libraryDependencies += "com.salesforce.transmogrifai" %% "transmogrifai-local" % "0.5.0"

libraryDependencies += libraryDependencies += "com.opendatagroup" % "hadrian" % "0.8.5" % Runtime
```

Then in your code you may load and score models as follows:
```scala
import com.salesforce.op.local._

val model = workflow.loadModel("/path/to/model")
val scoreFn = model.scoreFunction
val rawData = Seq(Map("age" -> 18, "name" -> "Peter"), Map("name" -> "John"))
val scores= rawData.map(scoreFn)
```

Or using the local runner:
```scala
val scoreFn = new OpWorkflowRunnerLocal(workflow).score(opParams)
```


## Performance Results

Below is an example of measured scoring performance on 6m records with 10 fields and 12 transformations applied.
Executed on MacBook Pro i7 3.5Ghz in a single thread.
```
Scored 6,000,000 records in 239s
Average time per record: 0.0399215ms
```
