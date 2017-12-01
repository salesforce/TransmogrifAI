/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.features.types._
import com.salesforce.op.features.{OPFeature, TransientFeature}
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test.PassengerSparkFixtureTest
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{OpPipelineStageReader, OpPipelineStageWriter, Pipeline}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfterEach, FlatSpec}

import scala.reflect.runtime.universe.TypeTag


@RunWith(classOf[JUnitRunner])
class OpPipelineStagesTest
  extends FlatSpec with PassengerSparkFixtureTest with BeforeAndAfterEach with Serializable {

  val tfParam = new TransientFeatureArrayParam(name = "foo", parent = "null", doc = "nothing")

  var stage: TestStage = _

  override def beforeEach(): Unit = {
    stage = new TestStage
  }

  "TransientFeatureArrayParam" should "decode json properly" in {

    val features: Array[TransientFeature] = tfParam.jsonDecode(
      """[{"name":"f1","isResponse":true,"isRaw":true,"uid":"foo","typeName":"com.salesforce.op.features.types.Real",
        |"history":"{\"originFeatures\":[\"f1\"],\"stages\":[]}"},{"name":"f2","isResponse":false,
        |"isRaw":false,"uid":"bar","typeName":"com.salesforce.op.features.types.Text",
        |"history":"{\"originFeatures\":[\"f1\"],\"stages\":[]}"}]
        |""".stripMargin
    )

    features should have length 2
    features(0).name shouldBe "f1"
    features(1).name shouldBe "f2"

    features(0).uid shouldBe "foo"
    features(1).uid shouldBe "bar"

    features(0).isResponse && features(0).isRaw shouldBe true
    features(1).isResponse || features(1).isRaw shouldBe false

    assertThrows[RuntimeException] { features(0).getFeature() }
    assertThrows[RuntimeException] { features(1).getFeature() }
  }

  it should "encode json properly" in {
    val tfs = Array[TransientFeature](TransientFeature(weight), TransientFeature(height))
    val tfsRecovered = tfParam.jsonDecode(tfParam.jsonEncode(tfs))

    tfsRecovered should have length 2
    compare(tfsRecovered(0), weight)
    compare(tfsRecovered(1), height)
    assertThrows[RuntimeException] { tfsRecovered(0).getFeature() }
    assertThrows[RuntimeException] { tfsRecovered(1).getFeature() }
  }

  "Stage" should "set Transient Features properly as an input feature" in {
    stage.setInput(height)
    val inputs = stage.getInputFeatures()
    inputs should have length 1
    inputs.head shouldBe height

    stage.get(stage.inputFeatures).get.head.asFeatureLike[Real] shouldBe height
  }

  it should "be robust when getting features by index" in {
    assertThrows[RuntimeException] { stage.getInputFeature(0) }
    stage.getTransientFeature(0) shouldBe None

    stage.setInput(height)
    stage.getInputFeature(0) shouldBe Some(height)
    stage.getTransientFeature(0).get.getFeature() shouldBe height
    stage.getInputFeature(1) shouldBe None
    stage.getTransientFeature(1) shouldBe None
  }

  it should "copy the input feature through the ParamMap" in {
    stage.setInput(height)
    val copy = stage.copy(new ParamMap())

    copy.getInputFeatures should have length 1
  }

  val testOp = new com.salesforce.op.stages.base.unary.UnaryLambdaTransformer[Real, Real](
    operationName = "test",
    transformFn = (i: Real) => i,
    uid = "myID"
  )

  Spec[OpPipelineStageReader] should "load output from StageWriter and have correct transient feature state" in {
    val writer = new OpPipelineStageWriter(testOp)
    testOp.setInput(weight)
    val reader = new OpPipelineStageReader(testOp)
    val stage = reader.loadFromJsonString(writer.writeToJsonString).asInstanceOf[UnaryLambdaTransformer[Real, Real]]

    val features = stage.get(stage.inputFeatures).get
    features should have length 1
    features(0).getFeature() shouldBe weight
    features(0).name shouldBe weight.name
    features(0).uid shouldBe weight.uid
  }

  private def compare(tf: TransientFeature, f: OPFeature): Unit = {
    tf.name shouldBe f.name
    tf.isRaw shouldBe f.isRaw
    tf.isResponse shouldBe f.isResponse
    tf.uid shouldBe f.uid
    tf.typeName shouldBe f.typeName
  }
}

class TestStage(implicit val tto: TypeTag[RealNN], val ttov: TypeTag[RealNN#Value])
  extends Pipeline with OpPipelineStage1[RealNN, RealNN] {
  override def operationName: String = "test"
}
