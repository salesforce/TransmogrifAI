# TransmogrifAI Local

This module enables local scoring of TransmogrifAI models without need in Spark session,
instead using [Aardpfark](https://github.com/CODAIT/aardpfark) and [Hadrian](https://github.com/opendatagroup/hadrian)
runtime for JVM. It delivers unprecedented portability and performance of TransmogrifAI models
allowing scores serving from any JVM process.

## Usage

Add the `transmogrifai-local` dependency into your project together with `hadrian` for runtime.

Gradle: 
```gradle
dependencies {
    compile 'com.salesforce.transmogrifai:transmogrifai-local_2.11:0.4.0'
    runtime 'com.opendatagroup:hadrian:0.8.5'
}
```
SBT:
```sbt
libraryDependencies ++= "com.salesforce.transmogrifai" %% "transmogrifai-local" % "0.4.0"

libraryDependencies ++= libraryDependencies += "com.opendatagroup" % "hadrian" % "0.8.5" % Runtime
```

Then in your code you might score as follows:
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
