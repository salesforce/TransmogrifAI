# TransmogrifAI Command Line Interface (OP CLI)

## Installation

1. add to your bash profile:
    ```bash
    export OP_HOME="/path/to/octopus-prime/"
    alias op="java -cp $OP_HOME/cli/build/libs/\* com.salesforce.op.cli.CLI"
    ```
2. run `./gradlew cli:shadowJar` in your octopus-prime project directory
3. now running `op` in terminal should run the most recently build shadow jar every time
4. if you change the CLI, rerun `./gradlew cli:shadowJar`

## Usage

Run `op --help` to list the all the CLI options.

For instance, in order to generate a binary classification model with Titanic data one can do it simply by running:
```bash
op gen --input Passengers.csv --id passengerId --response survived --schema Passenger.avsc TitanicProj
```

If you have answers to questions in a file (see e.g. passengers.answers), and want to overwrite the old project,
the command line looks like this:
```
op gen --input templates/simple/src/main/resources/PassengerData.csv --id passengerId --response survived --schema utils/src/main/avro/PassengerCSV.avsc --answers cli/passengers.answers TitanicProj10112017 --overwrite
```
## Templates

Right now there is only one template, `simple`, defined in `templates/simple`.
The template compiles with the rest of the project to make sure that there are no errors.
Each file in the template will be copied over whenever a user runs the `op gen` command, and they will be run through
a very tiny templating engine that allows substitutions.
To exclude a file from templating (the best reason for this is if the file is binary), add it to the
 [shouldCopy](https://github.com/salesforce/TransmogrifAI/blob/master/cli/src/main/scala/com/salesforce/op/cli/gen/templates/SimpleTemplate.scala#L26)
 method for your template.

This templating engine has directives in comments that look like:
```scala
val (pred, raw, prob) = BinaryClassificationModelSelector() /* << PROBLEM_KIND */
    .setInput(label, checkedFeatures)
    .getOutput()
```

The templating engine reads comments like these and pulls the value to substitute (in this case `PROBLEM_KIND`)
from the map of arguments created by the project template (for example, in 
[here](https://github.com/salesforce/TransmogrifAI/blob/master/cli/src/main/scala/com/salesforce/op/cli/gen/templates/SimpleTemplate.scala#L71-L82)).

What this directive does is replace the previous "scala expression" with the value to substitute. A "scala expression", in this case,
is a string of non-whitespace characters. Whitespace characters are included so long as there are unclosed brackets (`[`, `(`, or `{`).

To replace the next expression instead, do this:

```scala
object /* APP_NAME >> */ Simple extends OpApp with Features {
```

#### Examples

##### 1.
```scala
val (pred, raw, prob) = BinaryClassificationModelSelector()   /* << PROBLEM_KIND */
    .setInput(label, checkedFeatures)
    .getOutput()
```

with `Map("PROBLEM_KIND" -> "MY_VALUE")` becomes:

```scala
val (pred, raw, prob) = MY_VALUE
    .setInput(label, checkedFeatures)
    .getOutput()
```

##### 2.
```scala
object /* APP_NAME >> */ Simple extends OpApp with Features {
```

with `Map("APP_NAME" -> "MY_VALUE")` becomes:

```scala
object MY_VALUE extends OpApp with Features {
```

##### 3.

```scala
trait Features extends Serializable /* FEATURES >> */ {

  val survived = FeatureBuilder.RealNN[Passenger]
    .extract(_.getSurvived.toDouble.toRealNN).asResponse

  val embarked = FeatureBuilder.MultiPickList[Passenger]
    .extract(d => Option(d.getEmbarked).toSet[String].toMultiPickList).asPredictor

}
```

with `Map("FEATURES" -> "{\n\nhello world\n\n}")` becomes:

```scala
trait Features extends Serializable {

hello world

}
```

##### 4.

```scala
 val checkedFeatures = new SanityChecker()
    .setCheckSample(0.10)
    .setInput(label, featureVector) /* << NEW_INPUT */
    .getOutput()
```

with `Map("NEW_INPUT" -> ".myMethod(blah)")` becomes:

```scala
 val checkedFeatures = new SanityChecker()
    .setCheckSample(0.10)
    .myMethod(blah)
    .getOutput()
```

[See also our wiki](https://github.com/salesforce/TransmogrifAI/wiki/Bootstrap-Your-First-Project)
