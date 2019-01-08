/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op

import java.util.concurrent.TimeUnit

import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.StreamingContext
import org.slf4j.LoggerFactory
import scopt.Read

import scala.concurrent.duration.Duration


/**
 * A simple command line app for running an [[OpWorkflow]] with Spark.
 * A user needs to implement a [[run]] function.
 */
abstract class OpApp {

  @transient private val logr = LoggerFactory.getLogger(this.getClass)

  /**
   * The main function to run your [[OpWorkflow]].
   * The easiest way is to create an [[OpWorkflowRunner]] and run it.
   *
   * @param runType  run type
   * @param opParams  parameters injected at runtime
   * @param spark     spark session which runs the workflow
   * @param streaming spark streaming context which runs the workflow
   */
  def run(runType: OpWorkflowRunType, opParams: OpParams)
    (implicit spark: SparkSession, streaming: StreamingContext): Unit

  /**
   * Kryo registrar to use when creating a SparkConf.
   *
   * First create your own registrator by extending the [[OpKryoRegistrator]]
   * and then register your new classes by overriding [[OpKryoRegistrator.registerCustomClasses]].
   *
   * Then override this method to set your registrator with Spark.
   */
  def kryoRegistrator: Class[_ <: OpKryoRegistrator] = classOf[OpKryoRegistrator]

  /**
   * Default application name - to be used if 'spark.app.name' parameter is not set.
   */
  def defaultAppName: String = thisClassName

  private def thisClassName: String = this.getClass.getSimpleName.stripSuffix("$")

  /**
   * Application name (gets the value of 'spark.app.name' parameter).
   */
  def appName: String = sparkConf.get("spark.app.name")

  /**
   * Configuration for a Spark application.
   * Used to set various Spark parameters as key-value pairs.
   *
   * @return SparkConf
   */
  def sparkConf: SparkConf = {
    val conf = new SparkConf()
    conf
      .setAppName(conf.get("spark.app.name", defaultAppName))
      .set("spark.serializer", classOf[org.apache.spark.serializer.KryoSerializer].getName)
      .set("spark.kryo.registrator", kryoRegistrator.getName)
  }

  /**
   * Gets/creates a Spark Session.
   */
  def sparkSession: SparkSession = {
    val conf = sparkConf
    if (logr.isDebugEnabled) {
      logr.debug("*" * 80)
      logr.debug("SparkConf:\n{}", conf.toDebugString)
      logr.debug("*" * 80)
    }
    SparkSession.builder.config(conf).getOrCreate()
  }

  /**
   * Gets/creates a Spark Streaming Context.
   *
   * @param batchDuration the time interval at which streaming data will be divided into batches
   */
  def sparkStreamingContext(batchDuration: Duration): StreamingContext = {
    val bd = org.apache.spark.streaming.Milliseconds(batchDuration.toMillis)
    StreamingContext.getActiveOrCreate(() => new StreamingContext(sparkSession.sparkContext, bd))
  }

  /**
   * Parse command line arguments as [[OpParams]].
   *
   * @param args command line arguments
   * @return run type and [[OpParams]]
   */
  def parseArgs(args: Array[String]): (OpWorkflowRunType, OpParams) = {
    def optStr(s: String): Option[String] = if (s == null || s.isEmpty) None else Some(s)

    val parser = new scopt.OptionParser[OpWorkflowRunnerConfig](thisClassName) {
      implicit val runTypeRead: Read[OpWorkflowRunType] = scopt.Read.reads(OpWorkflowRunType.withNameInsensitive)
      override val errorOnUnknownArgument = false

      opt[OpWorkflowRunType]('t', "run-type").required().action { (x, c) =>
        c.copy(runType = x)
      }.text(s"the type of workflow run: ${OpWorkflowRunType.values.mkString(", ").toLowerCase}")

      opt[Map[String, String]]('r', "read-location").action { (x, c) =>
        c.copy(readLocations = x)
      }.text("optional location to read data from - will override reader default locations")

      opt[String]('m', "model-location").action { (x, c) =>
        c.copy(modelLocation = optStr(x))
      }.text("location to write/read a fitted model generated by workflow")

      opt[String]('w', "write-location").action { (x, c) =>
        c.copy(writeLocation = optStr(x))
      }.text("location in which to write out data generated by workflow")

      opt[String]('x', "metrics-location").action { (x, c) =>
        c.copy(metricsLocation = optStr(x))
      }.text("location in which to write out metrics generated by workflow")

      opt[String]('p', "param-location").action { (x, c) =>
        c.copy(paramLocation = optStr(x))
      }.text("optional location of parameters for workflow run")

      checkConfig(_.validate match { case Left(error: String) => Left(error) case _ => Right(()) })
      help("help").text("prints this usage text")
    }
    val config = parser.parse(args, OpWorkflowRunnerConfig())
    config match {
      case None => sys.exit(1)
      case Some(conf) =>
        logr.info("Parsed config:\n{}", conf)
        conf.runType -> conf.toOpParams.get
    }
  }

  /**
   * The main method - loads the params and runs the workflow according to parameter settings.
   *
   * @param args command line args to be parsed into [[OpWorkflowRunnerConfig]]
   */
  def main(args: Array[String]): Unit = {
    val (runType, opParams) = parseArgs(args)
    val batchDuration = Duration(opParams.batchDurationSecs.getOrElse(1), TimeUnit.SECONDS)
    val (spark, streaming) = sparkSession -> sparkStreamingContext(batchDuration)
    run(runType, opParams)(spark, streaming)
  }

}

/**
 * A simple command line app for running an [[OpWorkflow]] with Spark.
 * A user needs to implement a [[runner]] creation function.
 */
abstract class OpAppWithRunner extends OpApp {
  /**
   * Override this function to create an instance of [[OpWorkflowRunner]] to run your workflow
   *
   * @param opParams parameters injected at runtime
   * @return an instance of [[OpWorkflowRunner]]
   */
  def runner(opParams: OpParams): OpWorkflowRunner

  /**
   * The main function to run your [[OpWorkflow]].
   * The easiest way is to create an [[OpWorkflowRunner]] and run it.
   *
   * @param runType   run type
   * @param opParams  parameters injected at runtime
   * @param spark     spark session which runs the workflow
   * @param streaming spark streaming context which runs the workflow
   */
  override def run(runType: OpWorkflowRunType, opParams: OpParams)
    (implicit spark: SparkSession, streaming: StreamingContext): Unit = {
    runner(opParams).run(runType, opParams)
  }
}
