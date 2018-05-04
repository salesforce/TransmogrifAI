/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.stages.FeatureGeneratorStage
import com.salesforce.op.stages.base.binary.BinaryLambdaTransformer
import com.salesforce.op.test.PassengerSparkFixtureTest
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class AliasTransformerTest extends FlatSpec with PassengerSparkFixtureTest {

  Spec[AliasTransformer[_]] should "allow aliasing features" in {
    val myFeature = (weight / height).alias
    myFeature.name shouldBe "myFeature"
    val all = myFeature.originStage.asInstanceOf[BinaryLambdaTransformer[_, _, _]]

    val transformed = all.transform(passengersDataSet)
    transformed.columns.contains(myFeature.name) shouldBe true
  }

  it should "copy successfully" in {
    val myFeature = ((weight * 2) / height).alias
    val copy = myFeature.originStage.copy(new ParamMap())
    copy.uid shouldBe myFeature.originStage.uid
  }

}
