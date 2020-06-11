# Running from Spark Shell

Start up your spark shell and add the [TransmogrifAI package](https://spark-packages.org/package/salesforce/TransmogrifAI):

```bash
$SPARK_HOME/bin/spark-shell --packages com.salesforce.transmogrifai:transmogrifai-core_2.11:0.7.0
```

Or if you'd like to use the latest version from master:
```bash
cd TransmogrifAI && ./gradlew core:shadowJar
$SPARK_HOME/bin/spark-shell --jars core/build/libs/transmogrifai-x.y.z.jar
```

Once the `spark-shell` starts up, create your spark session:

```scala
// Use the existing Spark session
implicit val spark = ss
// or set up a new one SparkSession if needed
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

val conf = new SparkConf().setAppName("TitanicPrediction")
implicit val spark = SparkSession.builder.config(conf).getOrCreate()
```

Import TransmogrifAI:
```scala
// All the TransmogrifAI functionality: feature types, feature builders, feature dsl, readers, aggregators etc.
import com.salesforce.op._
import com.salesforce.op.readers._
import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.aggregators._

// Optional - Spark type enrichments as follows
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRDD._
import com.salesforce.op.utils.spark.RichMetadata._
import com.salesforce.op.utils.spark.RichStructType._
```

Now follow along with the rest of the code from the Titanic example found [here](Titanic-Binary-Classification.html).


