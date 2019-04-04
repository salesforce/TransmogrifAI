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

package com.salesforce.op.stages

import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.{Model, Transformer}
import org.apache.spark.sql.types.{DataType, Metadata, MetadataBuilder}
import org.json4s.JsonAST.JValue
import org.json4s.jackson.JsonMethods.{compact, parse, pretty, render}
import org.json4s.{JArray, JObject}
import org.scalatest.FlatSpec
import org.slf4j.LoggerFactory


// TODO: consider adding a read/write test for a spark wrapped stage as well
private[stages] abstract class OpPipelineStageReaderWriterTest
  extends FlatSpec with PassengerSparkFixtureTest {

  val meta = new MetadataBuilder().putString("foo", "bar").build()
  val expectedFeaturesLength = 1
  def stage: OpPipelineStageBase
  val expected: Array[Real]
  val hasOutputName = true

  private val log = LoggerFactory.getLogger(this.getClass)
  private lazy val savePath = tempDir + "/" + this.getClass.getSimpleName + "-" + System.currentTimeMillis()
  private lazy val writer = new OpPipelineStageWriter(stage)
  private lazy val stageJsonString: String = writer.writeToJsonString(savePath, writeLambdas = true)
  private lazy val stageJson: JValue = parse(stageJsonString)
  private lazy val isModel = stage.isInstanceOf[Model[_]]
  private val FN = FieldNames

  Spec(this.getClass) should "write stage uid" in {
    log.info(pretty(stageJson))
    (stageJson \ FN.Uid.entryName).extract[String] shouldBe stage.uid
  }
  it should "write isModel" in {
    (stageJson \ FN.IsModel.entryName).extract[Boolean] shouldBe isModel
  }
  it should "write class name" in {
    (stageJson \ FN.Class.entryName).extract[String] shouldBe stage.getClass.getName
  }
  it should "write paramMap" in {
    val params = (stageJson \ FN.ParamMap.entryName).extract[Map[String, Any]]
    if (hasOutputName) {
      params should have size 4
      params.keys shouldBe Set("inputFeatures", "outputMetadata", "inputSchema", "outputFeatureName")
    } else {
      params should have size 3
      params.keys shouldBe Set("inputFeatures", "outputMetadata", "inputSchema")
    }
  }
  it should "write outputMetadata" in {
    val metadataStr = compact(render((stageJson \ FN.ParamMap.entryName) \ "outputMetadata"))
    val metadata = Metadata.fromJson(metadataStr)
    metadata shouldBe stage.getMetadata()
  }
  it should "write inputSchema" in {
    val schemaStr = compact(render((stageJson \ FN.ParamMap.entryName) \ "inputSchema"))
    val schema = DataType.fromJson(schemaStr)
    schema shouldBe stage.getInputSchema()
  }
  it should "write input features" in {
    val jArray = ((stageJson \ FN.ParamMap.entryName) \ "inputFeatures").extract[JArray]
    jArray.values should have length expectedFeaturesLength
    val obj = jArray(0).extract[JObject]
    obj.values.keys shouldBe Set("name", "isResponse", "isRaw", "uid", "typeName", "stages", "originFeatures")
  }
  it should "write model ctor args" in {
    if (stage.isInstanceOf[Model[_]]) {
      val ctorArgs = (stageJson \ FN.CtorArgs.entryName).extract[JObject]
      val (_, args) = ReflectionUtils.bestCtorWithArgs(stage)
      ctorArgs.values.keys shouldBe args.map(_._1).toSet
    }
  }
  it should "load stage correctly" in {
    val reader = new OpPipelineStageReader(stage)

    println(stageJsonString)
    val stageLoaded = reader.loadFromJsonString(stageJsonString, path = savePath, loadLambdas = true)
    stageLoaded shouldBe a[OpPipelineStageBase]
    stageLoaded shouldBe a[Transformer]
    stageLoaded.getOutput() shouldBe a[FeatureLike[_]]
    val _ = stage.asInstanceOf[Transformer].transform(passengersDataSet)
    val transformed = stageLoaded.asInstanceOf[Transformer].transform(passengersDataSet)
    transformed.collect(stageLoaded.getOutput().asInstanceOf[FeatureLike[Real]]) shouldBe expected
    stageLoaded.uid shouldBe stage.uid
    stageLoaded.operationName shouldBe stage.operationName
    stageLoaded.getInputFeatures() shouldBe stage.getInputFeatures()
    stageLoaded.getInputSchema() shouldBe stage.getInputSchema()
  }

}
