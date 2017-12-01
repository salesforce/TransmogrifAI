/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.utils.date.DateTimeUtils
import org.apache.spark.scheduler._
import org.slf4j.LoggerFactory

/**
 * Logs metrics upon completion of Spark application, jobs, stages
 *
 * @param appName         application name
 * @param appId           application id
 * @param runType         [[OpWorkflowRunType]]
 * @param customTagName   tag name printed on log lines
 * @param customTagValue  the value for the tag printed on log lines
 * @param logStageMetrics should log metrics for every stage
 *                        Note: can increase logging significantly if app has too many stages
 */
class OpSparkListener
(
  val appName: String,
  val appId: String,
  val runType: String,
  val customTagName: Option[String],
  val customTagValue: Option[String],
  val logStageMetrics: Boolean
) extends SparkListener {

  private lazy val log = LoggerFactory.getLogger(this.getClass)
  private var jobStartTime = DateTimeUtils.now().getMillis
  private var appStartTime = DateTimeUtils.now().getMillis

  val logPrefix: String = "%s:%s,RUN_TYPE:%s,APP:%s,APP_ID:%s".format(
    customTagName.getOrElse("APP_NAME"),
    customTagValue.getOrElse(appName),
    runType, appName, appId
  )

  log.info("Instantiated spark listener: {}. Log Prefix {}", this.getClass.getName, logPrefix: Any)

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    if (logStageMetrics) {
      log.info("{},STAGE:{},MEMORY_SPILLED_BYTES:{},GC_TIME_MS:{},STAGE_TIME_MS:{}",
        logPrefix, stageCompleted.stageInfo.name,
        stageCompleted.stageInfo.taskMetrics.memoryBytesSpilled.toString,
        stageCompleted.stageInfo.taskMetrics.jvmGCTime.toString,
        stageCompleted.stageInfo.taskMetrics.executorRunTime.toString
      )
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    jobStartTime = jobStart.time
  }

  override def onJobEnd(jobEnd: SparkListenerJobEnd): Unit = {
    val result = jobEnd.jobResult.getClass.getSimpleName.stripSuffix("$")
    log.info("{},JOB_ID:{},RESULT:{},JOB_TIME_MS:{}",
      logPrefix, jobEnd.jobId.toString, result,
      (jobEnd.time - jobStartTime).toString
    )
  }

  override def onApplicationStart(applicationStart: SparkListenerApplicationStart): Unit = {
    appStartTime = applicationStart.time
  }

  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
    log.info("{},APP_TIME_MS:{}", logPrefix.toString, applicationEnd.time - appStartTime: Any)
  }

}
