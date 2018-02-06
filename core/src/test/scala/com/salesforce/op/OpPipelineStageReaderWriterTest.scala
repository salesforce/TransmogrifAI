/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.features._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.{OpPipelineStageBase, OpPipelineStageReader, OpPipelineStageWriter}
import com.salesforce.op.stages.base.unary.{UnaryEstimator, UnaryLambdaTransformer, UnaryModel}
import com.salesforce.op.stages.impl.feature.PercentileCalibrator
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.reflection.ReflectionUtils
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.stages.OpPipelineStageReadWriteShared._
import org.apache.spark.ml.{Model, Transformer}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DataType, Metadata, MetadataBuilder}
import org.json4s.JsonAST.JValue
import org.json4s.jackson.JsonMethods.{compact, parse, pretty, render}
import org.json4s.{JArray, JObject}
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory


@RunWith(classOf[JUnitRunner])
class OpPipelineStageReaderWriterTest extends FlatSpec with PassengerSparkFixtureTest {

  val log = LoggerFactory.getLogger(this.getClass)

  val meta = new MetadataBuilder().putString("foo", "bar").build()

  val testTransformer = new UnaryLambdaTransformer[Real, Real](
    operationName = "test",
    transformFn = _.map(_ * 0.1234).toReal,
    uid = UID[UnaryLambdaTransformer[_, _]]
  ).setInput(weight).setMetadata(meta)

  val testEstimator = new MinMaxNormEstimator().setInput(weight).setMetadata(meta)

  val calibrator = new PercentileCalibrator().setInput(height)


  testReadWrite(
    stage = testTransformer,
    expected = Array(21.2248.toReal, Real.empty, 9.6252.toReal, 8.2678.toReal, 11.8464.toReal, 8.2678.toReal)
  )

  testReadWrite(
    stage = testEstimator.fit(passengersDataSet),
    expected =
      Array(1.0.toReal, Real.empty, 0.10476190476190476.toReal, 0.0.toReal, 0.2761904761904762.toReal, 0.0.toReal)
  )

  testReadWrite(
    stage = calibrator.fit(passengersDataSet),
    expected = Array(99.0.toReal, 25.0.toReal, 25.0.toReal, 25.0.toReal, 74.0.toReal, 50.0.toReal)
  )


  // TODO: consider testing a spark wrapped stages here as well

  private def testReadWrite(stage: OpPipelineStageBase, expected: Array[Real]): Unit = {
    val testName = stage.getClass.getSimpleName
    val writer = new OpPipelineStageWriter(stage)

    lazy val stageJsonString: String = writer.writeToJsonString
    lazy val stageJson: JValue = parse(stageJsonString)
    val isModel = stage.isInstanceOf[Model[_]]
    val FN = FieldNames

    s"Write($testName)" should "write stage uid" in {
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
    if (isModel) {
      it should "write model ctor args" in {
        val ctorArgs = (stageJson \ FN.CtorArgs.entryName).extract[JObject]
        val (_, args) = ReflectionUtils.bestCtorWithArgs(stage)
        ctorArgs.values.keys shouldBe args.map(_._1).toSet
      }
    }

    s"Read($testName)" should "load stage correctly" in {
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
}


class MinMaxNormEstimator(uid: String = UID[MinMaxNormEstimator])
  extends UnaryEstimator[Real, Real](operationName = "minMaxNorm", uid = uid) {

  def fitFn(dataset: Dataset[Real#Value]): UnaryModel[Real, Real] = {
    val grouped = dataset.groupBy()
    val maxVal = grouped.max().first().getDouble(0)
    val minVal = grouped.min().first().getDouble(0)
    new MinMaxNormEstimatorModel(
      min = minVal,
      max = maxVal,
      seq = Seq(minVal, maxVal),
      map = Map("a" -> Map("b" -> 1.0, "c" -> 2.0), "d" -> Map.empty),
      operationName = operationName,
      uid = uid
    )
  }
}

final class MinMaxNormEstimatorModel private[op]
(
  val min: Double,
  val max: Double,
  val seq: Seq[Double],
  val map: Map[String, Map[String, Double]],
  operationName: String, uid: String
)
  extends UnaryModel[Real, Real](operationName = operationName, uid = uid) {
  def transformFn: Real => Real = r => r.map(v => (v - min) / (max - min)).toReal
}
