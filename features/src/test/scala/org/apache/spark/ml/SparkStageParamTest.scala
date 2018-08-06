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

package org.apache.spark.ml

import com.salesforce.op.stages.SparkStageParam
import com.salesforce.op.test.TestSparkContext
import org.apache.spark.ml.feature.StandardScaler
import org.joda.time.DateTime
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods.{parse, _}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}


@RunWith(classOf[JUnitRunner])
class SparkStageParamTest extends FlatSpec with TestSparkContext with BeforeAndAfterEach {
  import SparkStageParam._

  var savePath: String = _
  var param: SparkStageParam[StandardScaler] = _
  var stage: StandardScaler = _

  override def beforeEach(): Unit = {
    super.beforeEach()
    savePath = tempDir + "/op-stage-param-test-" + DateTime.now().getMillis
    param = new SparkStageParam[StandardScaler](parent = "test" , name = "test", doc = "none")
    // by setting both to be the same, we guarantee that at least one isn't the default value
    stage = new StandardScaler().setWithMean(true).setWithStd(false)
  }

  // easier if test both at the same time
  Spec[SparkStageParam[_]] should "encode and decode properly when is set" in {
    param.savePath = Option(savePath)
    val jsonOut = param.jsonEncode(Option(stage))
    val parsed = parse(jsonOut).asInstanceOf[JObject]
    val updated = parsed ~ ("path" -> savePath) // inject path for decoding

    updated shouldBe JObject(
      "className" -> JString(stage.getClass.getName),
      "uid" -> JString(stage.uid),
      "path" -> JString(savePath)
    )
    val updatedJson = compact(updated)

    param.jsonDecode(updatedJson) match {
      case None => fail("Failed to recover the stage")
      case Some(stageRecovered) =>
        stageRecovered shouldBe a[StandardScaler]
        stageRecovered.uid shouldBe stage.uid
        stageRecovered.getWithMean shouldBe stage.getWithMean
        stageRecovered.getWithStd shouldBe stage.getWithStd
    }
  }

  it should "except out when path is empty" in {
    intercept[RuntimeException](param.jsonEncode(Option(stage))).getMessage shouldBe
      s"Path must be set before Spark stage '${stage.uid}' can be saved"
  }

  it should "have empty path if stage is empty" in {
    param.savePath = Option(savePath)
    val jsonOut = param.jsonEncode(None)
    val parsed = parse(jsonOut)

    parsed shouldBe JObject("className" -> JString(NoClass), "uid" -> JString(NoUID))
    param.jsonDecode(jsonOut) shouldBe None
  }
}
