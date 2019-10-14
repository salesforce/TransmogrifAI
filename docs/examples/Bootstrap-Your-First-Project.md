# Bootstrap Your First Project

We provide a convenient way to bootstrap you first project with TransmogrifAI using the TransmogrifAI CLI.
As an illustration, let's generate a binary classification model with the Titanic passenger data.

Clone the TransmogrifAI repo:
```bash
git clone https://github.com/salesforce/TransmogrifAI.git
```
Checkout the latest release branch (in this example `0.6.2`):
```bash
cd ./TransmogrifAI
git checkout 0.6.2
```
Build the TransmogrifAI CLI by running:
```bash
./gradlew cli:shadowJar
alias transmogrifai="java -cp `pwd`/cli/build/libs/\* com.salesforce.op.cli.CLI"
```
Finally generate your Titanic model project (follow the instructions on screen):
```bash
transmogrifai gen --input `pwd`/test-data/PassengerDataAll.csv \
  --id passengerId --response survived \
  --schema `pwd`/test-data/PassengerDataAll.avsc Titanic
```  

If you run this command more than once, two important command line arguments will be useful:
- `--overwrite` will allow to overwrite an existing project; if not specified, the generator will fail
- `--answers <answers_file>` will provide answers to the questions that the generator asks.

e.g.
```bash
transmogrifai gen --input `pwd`/test-data/PassengerDataAll.csv \
  --id passengerId --response survived \
  --schema `pwd`/test-data/PassengerDataAll.avsc \
  --answers cli/passengers.answers Titanic --overwrite
```
will do the generation without asking you anything.

Here we have specified the schema of the input data as an Avro schema. Avro is the schema format that the TransmogrifAI CLI understands. Note that when writing up your machine learning workflow by hand, you can always use case classes instead.

Your Titanic model project is ready to go. 

You will notice a default set of [FeatureBuilders](../developer-guide#featurebuilders) generated from the provided Avro schema. You are encouraged to edit this code to customize feature generation and take full advantage of the Feature types available (selecting the appropriate type will improve automatic feature engineering steps).
 
The generated code also uses the ```.transmogrify()``` shortcut to apply default feature transformations to the raw features and create a single feature vector. This is in essence the [automatic feature engineering Stage](../automl-capabilities#vectorizers-and-transmogrification) of TransmogrifAI. Once again, you can customize and expand on the feature manipulations applied by acting directly on individual features before applying ```.vectorize()```. You can also choose to completely discard ```.vectorize()``` in favor of hand-tuned feature engineering and manual vector creation using the VectorsCombiner Estimator (short-hand ```Vectorizers.combine()```) if you desire to have complete control over the feature engineering.

For convenience we have provided a simple `OpAppWithRunner` (and a more customizable `OpApp`) which takes in a workflow and allows you to run spark jobs from the command line rather than creating your own Spark App.

```scala
object Titanic extends OpAppWithRunner with TitanicWorkflow {
   def runner(opParams: OpParams): OpWorkflowRunner =
    new OpWorkflowRunner(
      workflow = titanicWorkflow,
      trainingReader = trainingReader,
      scoringReader = scoringReader,
      evaluator = evaluator,
      scoringEvaluator = None,
      featureToComputeUpTo = featureVector,
      kryoRegistrator = classOf[TitanicKryoRegistrator]
    )
}
```

This app is generated as part of the template and can be run like this:

```bash
cd titanic
./gradlew compileTestScala installDist
./gradlew sparkSubmit -Dmain=com.salesforce.app.Titanic -Dargs="--run-type=train --model-location=/tmp/titanic-model --read-location Passenger=`pwd`/../test-data/PassengerDataAll.csv"
```


To generate a project for any other dataset, simply modify the parameters to point to your specific data and its schema.

Happy modeling!
