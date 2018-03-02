/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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

  def stage: OpPipelineStageBase
  val expected: Array[Real]

  private val log = LoggerFactory.getLogger(this.getClass)
  private lazy val writer = new OpPipelineStageWriter(stage)
  private lazy val stageJsonString: String = writer.writeToJsonString
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
    params should have size 3
    params.keys shouldBe Set("outputMetadata", "inputFeatures", "inputSchema")
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
    jArray.values should have length 1
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
    val stageLoaded = reader.loadFromJsonString(stageJsonString)
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
