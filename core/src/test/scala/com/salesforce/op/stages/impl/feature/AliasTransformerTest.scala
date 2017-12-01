/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types._
import com.salesforce.op.test.PassengerSparkFixtureTest
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.FlatSpec

import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class AliasTransformerTest extends FlatSpec with PassengerSparkFixtureTest {

  Spec[AliasTransformer[_]] should "allow aliasing features" in {
    val myWeight = weight.alias
    myWeight.name shouldBe "myWeight"

    myWeight.originStage shouldBe a[AliasTransformer[_]]
    val all = myWeight.originStage.asInstanceOf[AliasTransformer[_]]
    all.tti.tpe =:= typeOf[Real] shouldBe true
    all.tto.tpe =:= typeOf[Real] shouldBe true
    all.ttov.tpe =:= typeOf[Real#Value] shouldBe true

    val transformed = all.transform(passengersDataSet)
    transformed.columns.contains(myWeight.name) shouldBe true
    transformed.columns.contains(weight.name) shouldBe true
    transformed.collect(weight) shouldBe transformed.collect(myWeight)
  }

  it should "copy successfully" in {
    val myFeature = ((weight * 2) / height).alias
    val copy = myFeature.originStage.copy(new ParamMap())
    copy shouldBe a[AliasTransformer[_]]
    copy.uid shouldBe myFeature.originStage.uid
  }

}
