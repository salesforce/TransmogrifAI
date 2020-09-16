# Installation

* Download and install Java 1.8
    * To stay sane, some prefer to use JENV to manage different versions of Java: [Recommended Blog](https://medium.com/@danielnenkov/multiple-jdk-versions-on-mac-os-x-with-jenv-5ea5522ddc9b)
    * On a Mac you might install Java 1.8 with: `brew cask install homebrew/cask-versions/zulu8`
    * Add this new version to your JENV (on Mac): `jenv add /Library/Java/JavaVirtualMachines/zulu-8.jdk/Contents/Home/`
* Determine which Java version you are using (See JENV above):
  * `$ which java`
  * `/Users/<user>/.jenv/shims/java`
* If you are not pointing to the proper version: `jenv versions`
* Chose the proper version: `jenv local zulu64-1.8.x.x`
* Set the JAVA_HOME environment variable.  Examples:
  * `export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)`
  * `export JAVA_HOME=$(/Users/<user>/.jenv/shims/java -v 1.8)`
* Get Spark 2.4.5: [Download](https://spark.apache.org/downloads.html), unzip it and then set an environment variable: `export SPARK_HOME=<SPARK_FOLDER>`
* Clone the TransmogrifAI repo: `git clone https://github.com/salesforce/TransmogrifAI.git`
* Build the project: `cd TransmogrifAI && ./gradlew compileTestScala installDist`
* Start hacking
