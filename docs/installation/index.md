# Installation

* Install Java 1.8
* Get Spark 2.2.x: [Download](https://spark.apache.org/downloads.html), unzip it and then set an environment variable: `export SPARK_HOME=<SPARK_FOLDER>`
* Clone the TransmogrifAI repo: `git clone https://github.com/salesforce/TransmogrifAI.git`
* Build the project: `cd TransmogrifAI && ./gradlew compileTestScala installDist`
* Start hacking

# (Optional) Configuration

## Custom Output Committer's

Depending on the deployment approach, we can choose to implement/use customized OutputCommitter classes. Following properties can be configured to override default classes and use customized output committer classes.
* `spark.hadoop.mapred.output.committer.class`
* `spark.hadoop.spark.sql.sources.outputCommitterClass`

* [S3A Committer](https://hadoop.apache.org/docs/current3/hadoop-aws/tools/hadoop-aws/committers.html), [Cloud Integration](https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/latest/cloud-integration.html#configuring) guides provide more details on the topic.
