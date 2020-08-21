# Installation

* Download and install Java 1.8, then set an environment variable: `export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)`
* Get Spark 2.4.x: [Download](https://spark.apache.org/downloads.html), unzip it and then set an environment variable: `export SPARK_HOME=<SPARK_FOLDER>`
* Clone the TransmogrifAI repo: `git clone https://github.com/salesforce/TransmogrifAI.git`
* Build the project: `cd TransmogrifAI && ./gradlew compileTestScala installDist`
* Start hacking
