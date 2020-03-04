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

package com.salesforce.op.utils.spark

import com.fasterxml.jackson.core.JsonGenerator
import com.fasterxml.jackson.databind.SerializerProvider
import com.fasterxml.jackson.databind.ser.std.StdSerializer
import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.json.{JsonLike, JsonUtils, SerDes}
import com.salesforce.op.utils.version.VersionInfo
import com.twitter.algebird.Max
import com.twitter.algebird.Operators._
import com.twitter.algebird.macros.caseclass
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
  private var jobGroup = OpSparkListener.DEFAULT_GROUP_ID
  private val stageMetrics = ArrayBuffer.empty[StageMetrics]
  private var cumulativeStageMetrics: CumulativeStageMetrics = CumulativeStageMetrics.zero

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
    cumulativeStageMetrics = cumulativeStageMetrics,
    versionInfo = VersionInfo()
  )

  override def onStageCompleted(stageCompleted: SparkListenerStageCompleted): Unit = {
    val si = stageCompleted.stageInfo
    val tm = si.taskMetrics
    val sm = StageMetrics(si)
    if (collectStageMetrics) {
      stageMetrics += sm
    }

    cumulativeStageMetrics = CumulativeStageMetrics.plus(cumulativeStageMetrics, sm)

    if (logStageMetrics) {
      log.info("{},STAGE:{},MEMORY_SPILLED_BYTES:{},GC_TIME_MS:{},STAGE_TIME_MS:{},JOB_GROUP:{}",
        logPrefix, si.name, tm.memoryBytesSpilled.toString, tm.jvmGCTime.toString, tm.executorRunTime.toString, jobGroup
      )
    }
  }

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    jobGroup = jobStart.properties.getProperty(OpSparkListener.SPARK_JOB_GROUP_ID, OpSparkListener.DEFAULT_GROUP_ID)
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

object OpSparkListener {
  private val SPARK_JOB_GROUP_ID = "spark.jobGroup.id"
  val DEFAULT_GROUP_ID: String = "other"
}

trait MetricJsonLike extends JsonLike {
  override def toJson(pretty: Boolean): String = {
    JsonUtils.toJsonString(this, pretty = pretty, Seq(SerDes[Max[Long]](
      classOf[Max[Long]],
      new StdSerializer[Max[Long]](classOf[Max[Long]]) {
        override def serialize(
          value: Max[Long],
          gen: JsonGenerator, provider: SerializerProvider
        ): Unit = {
          gen.writeNumber(value.get)
        }
      },
      null // not necessary
    )))
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
  cumulativeStageMetrics: CumulativeStageMetrics,
  versionInfo: VersionInfo
) extends MetricJsonLike {

  def appDurationPretty: String = {
    val duration = new Duration(appDuration)
    new PeriodFormatterBuilder()
      .appendHours().appendSuffix("h")
      .appendMinutes().appendSuffix("m")
      .appendSecondsWithOptionalMillis().appendSuffix("s")
      .toFormatter.print(duration.toPeriod())
  }
}


trait BaseStageMetrics {
  def numTasks: Int
  def numAccumulables: Int
  def executorRunTime: Long
  def executorCpuTime: Long
  def executorDeserializeTime: Long
  def executorDeserializeCpuTime: Long
  def resultSerializationTime: Long
  def jvmGCTime: Long
  def resultSizeBytes: Long
  def numUpdatedBlockStatuses: Int
  def diskBytesSpilled: Long
  def memoryBytesSpilled: Long
  def recordsRead: Long
  def bytesRead: Long
  def recordsWritten: Long
  def bytesWritten: Long
  def shuffleFetchWaitTime: Long
  def shuffleTotalBytesRead: Long
  def shuffleTotalBlocksFetched: Long
  def shuffleLocalBlocksFetched: Long
  def shuffleRemoteBlocksFetched: Long
  def shuffleWriteTime: Long
  def shuffleBytesWritten: Long
  def shuffleRecordsWritten: Long
  def duration: Option[Long]
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
) extends BaseStageMetrics with MetricJsonLike {
  override val duration: Option[Long] =
    for {
      c <- completionTime
      s <- submissionTime
    } yield c - s
}

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
      attemptId = si.attemptNumber,
      name = si.name,
      numTasks = si.numTasks,
      parentIds = si.parentIds,
      status = {
        // matches the spark private `StageInfo.getStatusString` function
        if (si.completionTime.isDefined && si.failureReason.isDefined) "failed"
        else if (si.completionTime.isDefined) "succeeded"
        else "running"
      },
      // TODO: consider also collecting all the accumilables - might be costly
      numAccumulables = si.accumulables.size,
      failureReason = si.failureReason,
      submissionTime = si.submissionTime,
      completionTime = si.completionTime,
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

case class CumulativeStageMetrics
(
  numTasks: Int,
  numAccumulables: Int,
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
  peakExecutionMemory: Max[Long],
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
  shuffleRecordsWritten: Long,
  duration: Option[Long] = None
) extends BaseStageMetrics with MetricJsonLike

object CumulativeStageMetrics {
  implicit val stageSG = caseclass.semigroup[CumulativeStageMetrics]

  val zero: CumulativeStageMetrics = CumulativeStageMetrics(
    numTasks = 0,
    numAccumulables = 0,
    executorRunTime = 0L,
    executorCpuTime = 0L,
    executorDeserializeTime = 0L,
    executorDeserializeCpuTime = 0L,
    resultSerializationTime = 0L,
    jvmGCTime = 0L,
    resultSizeBytes = 0L,
    numUpdatedBlockStatuses = 0,
    diskBytesSpilled = 0L,
    memoryBytesSpilled = 0L,
    peakExecutionMemory = Max(0L),
    recordsRead = 0L,
    bytesRead = 0L,
    recordsWritten = 0L,
    bytesWritten = 0L,
    shuffleFetchWaitTime = 0L,
    shuffleTotalBytesRead = 0L,
    shuffleTotalBlocksFetched = 0L,
    shuffleLocalBlocksFetched = 0L,
    shuffleRemoteBlocksFetched = 0L,
    shuffleWriteTime = 0L,
    shuffleBytesWritten = 0L,
    shuffleRecordsWritten = 0L
  )

  def plus(csm: CumulativeStageMetrics, sm: StageMetrics): CumulativeStageMetrics = csm +
    CumulativeStageMetrics(
      numTasks = sm.numTasks,
      numAccumulables = sm.numAccumulables,
      executorRunTime = sm.executorRunTime,
      executorCpuTime = sm.executorCpuTime,
      executorDeserializeTime = sm.executorDeserializeTime,
      executorDeserializeCpuTime = sm.executorDeserializeCpuTime,
      resultSerializationTime = sm.resultSerializationTime,
      jvmGCTime = sm.jvmGCTime,
      resultSizeBytes = sm.resultSizeBytes,
      numUpdatedBlockStatuses = sm.numUpdatedBlockStatuses,
      diskBytesSpilled = sm.diskBytesSpilled,
      memoryBytesSpilled = sm.memoryBytesSpilled,
      peakExecutionMemory = Max(sm.peakExecutionMemory),
      recordsRead = sm.recordsRead,
      bytesRead = sm.bytesRead,
      recordsWritten = sm.recordsWritten,
      bytesWritten = sm.bytesWritten,
      shuffleFetchWaitTime = sm.shuffleFetchWaitTime,
      shuffleTotalBytesRead = sm.shuffleTotalBytesRead,
      shuffleTotalBlocksFetched = sm.shuffleTotalBlocksFetched,
      shuffleLocalBlocksFetched = sm.shuffleLocalBlocksFetched,
      shuffleRemoteBlocksFetched = sm.shuffleRemoteBlocksFetched,
      shuffleWriteTime = sm.shuffleWriteTime,
      shuffleBytesWritten = sm.shuffleBytesWritten,
      shuffleRecordsWritten = sm.shuffleRecordsWritten,
      duration = sm.duration
  )
}
