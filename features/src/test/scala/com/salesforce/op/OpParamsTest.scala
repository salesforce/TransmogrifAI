/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import java.io.File

import com.salesforce.op.test._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

import scala.util.Failure


@RunWith(classOf[JUnitRunner])
class OpParamsTest extends FlatSpec with TestCommon {

  val expectedParamsSimple = new OpParams(
    stageParams = Map(
      "TestClass1" -> Map("param1" -> 11, "param2" -> "blarg", "param3" -> false),
      "TestClass2" -> Map("param1" -> List("a", "b", "c"), "param2" -> 0.25)
    ),
    readerParams = Map("Passenger" -> new ReaderParams()),
    modelLocation = None,
    writeLocation = None,
    metricsLocation = None,
    metricsCompress = None,
    metricsCodec = None,
    customParams = Map("custom1" -> 1, "custom2" -> "2"),
    customTagName = Some("myTag"),
    customTagValue = None,
    logStageMetrics = None
  )

  Spec[OpParams] should "correctly load parameters from a json file" in {
    val workflowParams = OpParams.fromFile(resourceFile(name = "OpParams.json"))
    assertParams(workflowParams.get)
  }

  it should "correctly load parameters with a complex reader format" in {
    val params = OpParams.fromFile(resourceFile(name = "OpParamsComplex.json"))
    val readerParams = params.get.readerParams
    println(readerParams)
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
    println("loaded:\n" + loaded)
    println("expected:\n" + expected)
    val readerParam = expected.readerParams.values.head
    val readerLoaded = loaded.readerParams.values.head
    expected.stageParams shouldBe loaded.stageParams
    expected.readerParams.keySet shouldBe loaded.readerParams.keySet
    readerParam.partitions shouldBe readerLoaded.partitions
    readerParam.path shouldBe readerLoaded.path
    readerParam.customParams shouldBe readerLoaded.customParams
    expected.writeLocation shouldBe loaded.writeLocation
    expected.modelLocation shouldBe loaded.modelLocation
    expected.metricsCodec shouldBe loaded.metricsCodec
    expected.metricsCompress shouldBe loaded.metricsCompress
    expected.customParams shouldBe loaded.customParams
    expected.customTagName shouldBe loaded.customTagName
    expected.customTagValue shouldBe loaded.customTagValue
    expected.logStageMetrics shouldBe loaded.logStageMetrics
  }

}
