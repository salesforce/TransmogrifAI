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
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class OpSparkListenerTest extends FlatSpec with TestSparkContext {
  val listener = new OpSparkListener(sc.appName, sc.applicationId, "testRun", Some("tag"), Some("tagValue"), true, true)
  sc.addSparkListener(listener)
  spark.read.csv(s"$testDataDir/PassengerDataAll.csv")

  Spec[OpSparkListener] should "capture app metrics" in {
    val appMetrics: AppMetrics = listener.metrics
    appMetrics.appName shouldBe sc.appName
    appMetrics.appId shouldBe sc.applicationId
    appMetrics.runType shouldBe "testRun"
    appMetrics.customTagName shouldBe Some("tag")
    appMetrics.customTagValue shouldBe Some("tagValue")
    appMetrics.appDurationPretty shouldBe "0s"
  }

  it should "capture app stage metrics" in {
    val stageMetrics = listener.metrics.stageMetrics
    stageMetrics.size shouldBe 1
    stageMetrics.head.name startsWith "csv at OpSparkListenerTest.scala"
    stageMetrics.head.stageId shouldBe 0
    stageMetrics.head.numTasks shouldBe 1
    stageMetrics.head.status shouldBe "succeeded"
  }
}
