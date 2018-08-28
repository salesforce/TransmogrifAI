# FAQ

## What is TransmogrifAI?

TransmogrifAI is an AutoML library written in Scala that runs on top of Spark. It was developed with a focus on enhancing machine learning developer productivity through machine learning automation, and an API that enforces compile-time type-safety, modularity and reuse.

Use TransmogrifAI if you need a machine learning library to:

* Rapidly train good quality machine learnt models with minimal hand tuning
* Build modular, reusable, strongly typed machine learning workflows

## Why is "op" in the package name and at the start of many class names?

OP is a reference to the internal codename that the project was developed under: Optimus Prime.

## I am used to working in Python why should I care about type safety?

The flexibility of Salesforce Objects allows customers to modify even standard objects schemas. This means that when writing models for a multi-tenant environment the only information about what is in a column that we can really count on is the Salesforce type (i.e. Phone, Email, Mulipicklist, Percent, etc.). Working in a strictly typed environment allows us to leverage this information to perform sensible automatic feature engineering. 

In addition type safety assures that you get fewer unexpected data issues in production.

## What does automatic feature engineering based on types look like?

In order to take advantage of automatic type based feature engineering in TransmogrifAI one simply defines the features that will be used in the model and relies on TransmogrifAI to do the feature engineering. The code for this would look like:

```scala
val featureVector = Seq(email, name, description, salary, phone).transmogrify()
```

The transmogrify shortcut will sort the features by type and apply appropriate transformations. For example, email which will have the type `Email` will be split into prefix and domain, checked for spam-i-ness and pivoted for top domains. Similarly, description which will have the type `TextArea` will be automatically be converted to a feature vector using the [hashing trick](https://en.wikipedia.org/wiki/Feature_hashing). 

Of course if you want to manually perform these or other transformations you can simply specify the steps for each feature and use the VectorsCombiner Transformer to manually combine your final features. However, this gives developers the option of using default type specific feature engineering.

## What other AutoML functionality does TransmogrifAI provide? 

Look at the [AutoML Capabilities](../AutoML-Capabilities) section for a complete list of the powerful AutoML estimators that TransmogrifAI provides. In a nutshell, they are Transmogrifier for automatic feature engineering, SanityChecker and RawFeatureFilter for data cleaning and automatic feature selection, and ModelSelectors for different classes of problems for automatic model selection.

## What imports do I need for TransmogrifAI to work?

```scala
// TransmogrifAI functionality: feature types, feature builders, feature dsl, readers, aggregators etc.
import com.salesforce.op._
import com.salesforce.op.aggregators._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.readers._

// Spark enrichments (optional)
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRDD._
import com.salesforce.op.utils.spark.RichRow._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.RichStructType._
```

## I don't need joins or aggregations in my data preparation why can't I just use Spark to load my data and pass it into a Workflow?
You can! Simply use the `.setInputRDD(myRDD)` or `.setInputDataSet(myDataSet)` methods on Workflow to pass in your data.

## How do I examine intermediate data when trying to debug my ML workflow?
You can generate data up to any particular point in the Workflow using the method `.computeDataUpTo(myFeature)`. Calling this method on your Workflow or WorkflowModel will compute a DataFrame which contains all of the rows for features created up to that point in your flow.

