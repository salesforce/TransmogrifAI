/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.test._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import org.slf4j.LoggerFactory

import scala.util.Failure


@RunWith(classOf[JUnitRunner])
class OpParamsTest extends FlatSpec with TestCommon {

  val log = LoggerFactory.getLogger(this.getClass)

  val expectedParamsSimple = OpParams(
    stageParams = Map(
      "TestClass1" -> Map("param1" -> 11, "param2" -> "blarg", "param3" -> false),
      "TestClass2" -> Map("param1" -> List("a", "b", "c"), "param2" -> 0.25)
    ),
    readerParams = Map("Passenger" -> new ReaderParams()),
    customParams = Map("custom1" -> 1, "custom2" -> "2"),
    customTagName = Some("myTag")
  )

  Spec[OpParams] should "correctly load parameters from a json file" in {
    val workflowParams = OpParams.fromFile(resourceFile(name = "OpParams.json"))
    assertParams(workflowParams.get)
  }

  it should "correctly load parameters with a complex reader format" in {
    val params = OpParams.fromFile(resourceFile(name = "OpParamsComplex.json"))
    val readerParams = params.get.readerParams
    readerParams("Passenger").partitions shouldEqual Some(5)
    readerParams("Passenger").customParams.head shouldEqual("test" -> 1)
  }

  it should "correctly load parameters from a yaml file" in {
    val workflowParams = OpParams.fromFile(resourceFile(name = "OpParams.yaml"))
    assertParams(workflowParams.get)
  }

  it should "fail to load parameters from an invalid file" in {
    val workflowParams = OpParams.fromFile(resourceFile(name = "log4j.properties"))
    workflowParams shouldBe a[Failure[_]]
    workflowParams.failed.get shouldBe a[IllegalArgumentException]
  }

  it should "correctly load parameters from a json string" in {
    val workflowParams = OpParams.fromString(resourceString(name = "OpParams.json"))
    assertParams(workflowParams.get)
  }

  it should "correctly load parameters from a yaml string" in {
    val workflowParams = OpParams.fromString(resourceString(name = "OpParams.yaml", noSpaces = false))
    assertParams(workflowParams.get)
  }

  it should "fail to load parameters from an invalid string" in {
    val workflowParams = OpParams.fromString(resourceString(name = "log4j.properties"))
    workflowParams shouldBe a[Failure[_]]
    workflowParams.failed.get shouldBe a[IllegalArgumentException]
  }

  private def assertParams(loaded: OpParams, expected: OpParams = expectedParamsSimple): Unit = {
    log.info("loaded:\n" + loaded)
    log.info("expected:\n" + expected)
    expected.stageParams shouldBe loaded.stageParams
    expected.readerParams.keySet shouldBe loaded.readerParams.keySet
    expected.readerParams.values zip loaded.readerParams.values foreach { case (r1, r2) =>
      r1.partitions shouldBe r2.partitions
      r1.path shouldBe r2.path
      r1.customParams shouldBe r2.customParams
    }
    expected.modelLocation shouldBe loaded.modelLocation
    expected.writeLocation shouldBe loaded.writeLocation
    expected.metricsLocation shouldBe loaded.metricsLocation
    expected.metricsCompress shouldBe loaded.metricsCompress
    expected.metricsCodec shouldBe loaded.metricsCodec
    expected.customTagName shouldBe loaded.customTagName
    expected.customTagValue shouldBe loaded.customTagValue
    expected.logStageMetrics shouldBe loaded.logStageMetrics
    expected.collectStageMetrics shouldBe loaded.collectStageMetrics
    expected.customParams shouldBe loaded.customParams
  }

}
