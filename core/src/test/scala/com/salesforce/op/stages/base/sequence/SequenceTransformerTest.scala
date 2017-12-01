/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichDataset._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class SequenceTransformerTest extends FlatSpec with PassengerSparkFixtureTest {

  val toMP = new SequenceLambdaTransformer[Real, MultiPickList](operationName = "MP",
    transformFn = value => MultiPickList(value.map(_.v.getOrElse(0.0).toString).toSet)
  )

  Spec[SequenceLambdaTransformer[_, _]] should "work when returning a MultiPickList feature" in {
    toMP.setInput(age, weight)
    val transformedData = toMP.transform(passengersDataSet)
    val columns = transformedData.columns
    assert(columns.contains(toMP.outputName))
    val output = toMP.getOutput()
    val answer = passengersArray.map(r =>
      toMP.transformFn(Seq(r.getFeatureType[Real](age), r.getFeatureType[Real](weight)))
    )
    transformedData.collect(output) shouldBe answer
  }

  it should "copy successfully" in {
    val tr = new SequenceLambdaTransformer[Text, Text](
      operationName = "foo",
      transformFn = x => x.head
    )
    tr.copy(new ParamMap()).uid shouldBe tr.uid
  }

}
