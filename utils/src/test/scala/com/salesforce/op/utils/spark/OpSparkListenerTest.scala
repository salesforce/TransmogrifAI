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

import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.date.DateTimeUtils
import com.twitter.algebird.Max
import com.twitter.algebird.Operators._
import com.twitter.algebird.macros.caseclass
import org.apache.log4j._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.TableDrivenPropertyChecks

import scala.collection.mutable.ArrayBuffer

@RunWith(classOf[JUnitRunner])
class OpSparkListenerTest extends FlatSpec with TableDrivenPropertyChecks with TestSparkContext {
  val sparkLogAppender: MemoryAppender = {
    val sparkAppender = new MemoryAppender()
    sparkAppender.setName("spark-appender")
    sparkAppender.setThreshold(Level.INFO)
    sparkAppender.setLayout(new org.apache.log4j.PatternLayout)
    LogManager.getLogger(classOf[OpSparkListener]).setLevel(Level.INFO)
    Logger.getRootLogger.addAppender(sparkAppender)
    sparkAppender
  }

  val start = DateTimeUtils.now().getMillis
  val listener = new OpSparkListener(sc.appName, sc.applicationId, "testRun", Some("tag"), Some("tagValue"), true, true)
  sc.addSparkListener(listener)
  val _ = spark.read.csv(s"$testDataDir/PassengerDataAll.csv").count()
  spark.close()

  Spec[OpSparkListener] should "capture app metrics" in {
    val appMetrics: AppMetrics = listener.metrics
    appMetrics.appName shouldBe sc.appName
    appMetrics.appId shouldBe sc.applicationId
    appMetrics.runType shouldBe "testRun"
    appMetrics.customTagName shouldBe Some("tag")
    appMetrics.customTagValue shouldBe Some("tagValue")
    appMetrics.appStartTime should be >= start
    appMetrics.appEndTime should be >= appMetrics.appStartTime
    appMetrics.appDuration shouldBe (appMetrics.appEndTime - appMetrics.appStartTime)
    appMetrics.appDurationPretty.isEmpty shouldBe false
  }

  it should "capture app stage metrics" in {
    val appMetrics = listener.metrics
    val stageMetrics = appMetrics.stageMetrics
    stageMetrics.size should be > 0
    val firstStage = stageMetrics.head
    firstStage.name should startWith("csv at OpSparkListenerTest.scala")
    firstStage.stageId shouldBe 0
    firstStage.numTasks shouldBe 1
    firstStage.status shouldBe "succeeded"
    val dur = firstStage.completionTime.getOrElse(0L) - firstStage.submissionTime.getOrElse(0L)
    firstStage.duration shouldBe Option(dur)
    firstStage.toJson(pretty = true) should include(s""""duration" : $dur""")
  }

  it should "log messages for listener initialization, stage completion, app completion" in {
    val firstStage = listener.metrics.stageMetrics.head
    val logPrefix = listener.logPrefix
    val logs = sparkLogAppender.logs.map(_.getMessage.toString)
    val messages = Table("Spark Log Messages",
      "Instantiated spark listener: %s. Log Prefix %s".format(classOf[OpSparkListener].getName, logPrefix),
      "%s,APP_TIME_MS:%s".format(logPrefix, listener.metrics.appEndTime - listener.metrics.appStartTime),
      "%s,STAGE:%s,MEMORY_SPILLED_BYTES:%s,GC_TIME_MS:%s,STAGE_TIME_MS:%s,JOB_GROUP:%s".format(
        logPrefix, firstStage.name, firstStage.memoryBytesSpilled, firstStage.jvmGCTime, firstStage.executorRunTime,
        OpStep.Other.toString
      )
    )
    forAll(messages) { m => logs.contains(m) shouldBe true }
  }

  it should "be able to aggregate Stage metrics" in {
    implicit val stageSG = caseclass.semigroup[StageMetrics]

    val sm0 = CumulativeStageMetrics(
      numTasks = 1,
      numAccumulables = 100,
      executorRunTime = 1000L,
      executorCpuTime = 700L,
      executorDeserializeTime = 750L,
      executorDeserializeCpuTime = 740L,
      resultSerializationTime = 450L,
      jvmGCTime = 2000L,
      resultSizeBytes = 1L,
      numUpdatedBlockStatuses = 1,
      diskBytesSpilled = 1L,
      memoryBytesSpilled = 1L,
      peakExecutionMemory = Max(1000L),
      recordsRead = 1,
      bytesRead = 1,
      recordsWritten = 1,
      bytesWritten = 1,
      shuffleFetchWaitTime = 1,
      shuffleTotalBytesRead = 1,
      shuffleTotalBlocksFetched = 1,
      shuffleLocalBlocksFetched = 1,
      shuffleRemoteBlocksFetched = 1,
      shuffleWriteTime = 1,
      shuffleBytesWritten = 1,
      shuffleRecordsWritten = 1,
      duration = Some(1)
    )

    val sm1 = CumulativeStageMetrics(
      numTasks = 10,
      numAccumulables = 100,
      executorRunTime = 1000,
      executorCpuTime = 700,
      executorDeserializeTime = 750,
      executorDeserializeCpuTime = 740,
      resultSerializationTime = 450,
      jvmGCTime = 2000,
      resultSizeBytes = 1,
      numUpdatedBlockStatuses = 1,
      diskBytesSpilled = 1,
      memoryBytesSpilled = 1,
      peakExecutionMemory = Max(1001),
      recordsRead = 1,
      bytesRead = 1,
      recordsWritten = 1,
      bytesWritten = 1,
      shuffleFetchWaitTime = 1,
      shuffleTotalBytesRead = 1,
      shuffleTotalBlocksFetched = 1,
      shuffleLocalBlocksFetched = 1,
      shuffleRemoteBlocksFetched = 1,
      shuffleWriteTime = 1,
      shuffleBytesWritten = 1,
      shuffleRecordsWritten = 1,
      duration = Some(2)
    )

    val total = Seq(sm0, sm1).foldLeft(CumulativeStageMetrics.zero)(_ + _)

    total.peakExecutionMemory shouldBe Max(1001)
    val jsonStr = total.toJson(pretty = true)
    jsonStr should include ("\"peakExecutionMemory\" : 1001")
    jsonStr should include ("\"duration\" : 3")
  }
}

/**
 * Class to enable in memory logging for tests
 */
class MemoryAppender extends AppenderSkeleton {
  private val logRecords = new ArrayBuffer[spi.LoggingEvent]

  override def requiresLayout: Boolean = true

  /**
   * Clear out the logRecords in log collection
   * @return Unit
   */
  override def close(): Unit = {
    logRecords.clear
  }

  /**
   * Add a log to the log collection
   * @param event The log event
   * @return Unit
   */
  override def append(event: spi.LoggingEvent): Unit = {
    logRecords.append(event)
  }

  /**
   * Log event collection
   * @return A collection of log events
   */
  def logs: ArrayBuffer[spi.LoggingEvent] = logRecords
}
