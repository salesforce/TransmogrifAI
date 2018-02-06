/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.version.VersionInfo
import org.apache.spark.scheduler._
import org.joda.time.Duration
import org.joda.time.format.PeriodFormatterBuilder
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

/**
 * Logs & collects metrics upon completion of Spark application, jobs, stages
 *
 * @param appName             application name
 * @param appId               application id
 * @param runType             [[OpWorkflowRunType]]
 * @param customTagName       tag name printed on log lines
 * @param customTagValue      the value for the tag printed on log lines
 * @param logStageMetrics     should log metrics for every stage
 *                            Note: can increase logging significantly if app has too many stages.
 * @param collectStageMetrics should collect metrics for every stage
 *                            Note: can increase memory usage on the driver if app has too many stages.
 */
class OpSparkListener
(
  val appName: String,
  val appId: String,
  val runType: String,
  val customTagName: Option[String],
  val customTagValue: Option[String],
  val logStageMetrics: Boolean,
  val collectStageMetrics: Boolean
) extends SparkListener {

  private lazy val log = LoggerFactory.getLogger(classOf[OpSparkListener])

  private var (jobStartTime, appStartTime, appEndTime) = {
    val now = DateTimeUtils.now().getMillis
    (now, now, now)
  }
  private val stageMetrics = ArrayBuffer.empty[StageMetrics]

  val logPrefix: String = "%s:%s,RUN_TYPE:%s,APP:%s,APP_ID:%s".format(
    customTagName.getOrElse("APP_NAME"),
    customTagValue.getOrElse(appName),
    runType, appName, appId
  )
  log.info("Instantiated spark listener: {}. Log Prefix {}", this.getClass.getName, logPrefix: Any)

  /**
   * All the metrics computed by the spark listener
   */
  def metrics: AppMetrics = AppMetrics(
    appName = appName,
    appId = appId,
    runType = runType,
    customTagName = customTagName,
    customTagValue = customTagValue,
    appStartTime = appStartTime,
    appEndTime = appEndTime,
    appDuration = appEndTime - appStartTime,
    stageMetrics = stageMetrics.toList,
    versionInfo = VersionInfo()
  )

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    val si = stageCompleted.stageInfo
    val tm = si.taskMetrics
    if (collectStageMetrics) stageMetrics += StageMetrics(si)
    if (logStageMetrics) {
      log.info("{},STAGE:{},MEMORY_SPILLED_BYTES:{},GC_TIME_MS:{},STAGE_TIME_MS:{}",
        logPrefix, si.name, tm.memoryBytesSpilled.toString, tm.jvmGCTime.toString, tm.executorRunTime.toString
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
    appEndTime = applicationEnd.time
    log.info("{},APP_TIME_MS:{}", logPrefix.toString, appEndTime - appStartTime: Any)
  }

}

/**
 * App metrics container.
 * Contains the app info, all the stage metrics computed by the spark listener and project version info.
 */
case class AppMetrics
(
  appName: String,
  appId: String,
  runType: String,
  customTagName: Option[String],
  customTagValue: Option[String],
  appStartTime: Long,
  appEndTime: Long,
  appDuration: Long,
  stageMetrics: Seq[StageMetrics],
  versionInfo: VersionInfo
) extends JsonLike {

  def appDurationPretty: String = {
    val duration = new Duration(appDuration)
    new PeriodFormatterBuilder()
      .appendHours().appendSuffix("h")
      .appendMinutes().appendSuffix("m")
      .appendSecondsWithOptionalMillis().appendSuffix("s")
      .toFormatter.print(duration.toPeriod())
  }
}

/**
 * Spark stage metrics container for a [[org.apache.spark.scheduler.StageInfo]]
 * Note: all the time values are in milliseconds.
 */
case class StageMetrics private
(
  stageId: Int,
  attemptId: Int,
  name: String,
  numTasks: Int,
  parentIds: Seq[Int],
  status: String,
  numAccumulables: Int,
  failureReason: Option[String],
  submissionTime: Option[Long],
  completionTime: Option[Long],
  duration: Option[Long],
  executorRunTime: Long,
  executorCpuTime: Long,
  executorDeserializeTime: Long,
  executorDeserializeCpuTime: Long,
  resultSerializationTime: Long,
  jvmGCTime: Long,
  resultSizeBytes: Long,
  numUpdatedBlockStatuses: Int,
  diskBytesSpilled: Long,
  memoryBytesSpilled: Long,
  peakExecutionMemory: Long,
  recordsRead: Long,
  bytesRead: Long,
  recordsWritten: Long,
  bytesWritten: Long,
  shuffleFetchWaitTime: Long,
  shuffleTotalBytesRead: Long,
  shuffleTotalBlocksFetched: Long,
  shuffleLocalBlocksFetched: Long,
  shuffleRemoteBlocksFetched: Long,
  shuffleWriteTime: Long,
  shuffleBytesWritten: Long,
  shuffleRecordsWritten: Long
) extends JsonLike

object StageMetrics {
  /**
   * Create an instance of [[StageMetrics]] container form a [[org.apache.spark.scheduler.StageInfo]] instance
   *
   * @param si spark stage info
   * @return [[StageMetrics]]
   */
  def apply(si: StageInfo): StageMetrics = {
    val tm = si.taskMetrics
    def toMillis(ns: Long): Long = ns / 1000000 // some time values are in nanoseconds so we convert those
    StageMetrics(
      stageId = si.stageId,
      attemptId = si.attemptId,
      name = si.name,
      numTasks = si.numTasks,
      parentIds = si.parentIds,
      status = {
        // matches the spark private `StageInfo.getStatusString` function
        if (si.completionTime.isDefined && si.failureReason.isDefined) "failed"
        else if (si.completionTime.isDefined) "succeeded"
        else "running"
      },
      // TODO: consider also collection all the accumilables - might be costly
      numAccumulables = si.accumulables.size,
      failureReason = si.failureReason,
      submissionTime = si.submissionTime,
      completionTime = si.completionTime,
      duration = for {s <- si.submissionTime; c <- si.completionTime} yield c - s,
      executorRunTime = tm.executorRunTime,
      executorCpuTime = toMillis(tm.executorCpuTime),
      executorDeserializeTime = tm.executorDeserializeTime,
      executorDeserializeCpuTime = toMillis(tm.executorDeserializeCpuTime),
      resultSerializationTime = tm.resultSerializationTime,
      jvmGCTime = tm.jvmGCTime,
      resultSizeBytes = tm.resultSize,
      numUpdatedBlockStatuses = tm.updatedBlockStatuses.length,
      diskBytesSpilled = tm.diskBytesSpilled,
      memoryBytesSpilled = tm.memoryBytesSpilled,
      peakExecutionMemory = tm.peakExecutionMemory,
      recordsRead = tm.inputMetrics.recordsRead,
      bytesRead = tm.inputMetrics.bytesRead,
      recordsWritten = tm.outputMetrics.recordsWritten,
      bytesWritten = tm.outputMetrics.bytesWritten,
      shuffleFetchWaitTime = tm.shuffleReadMetrics.fetchWaitTime,
      shuffleTotalBytesRead = tm.shuffleReadMetrics.totalBytesRead,
      shuffleTotalBlocksFetched = tm.shuffleReadMetrics.totalBlocksFetched,
      shuffleLocalBlocksFetched = tm.shuffleReadMetrics.localBlocksFetched,
      shuffleRemoteBlocksFetched = tm.shuffleReadMetrics.remoteBlocksFetched,
      shuffleWriteTime = toMillis(tm.shuffleWriteMetrics.writeTime),
      shuffleBytesWritten = tm.shuffleWriteMetrics.bytesWritten,
      shuffleRecordsWritten = tm.shuffleWriteMetrics.recordsWritten
    )
  }
}
