/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.ternary

import com.salesforce.op.features.types.Text
import com.salesforce.op.test._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class TernaryTransformerTest extends FlatSpec with TestCommon {

  Spec[TernaryLambdaTransformer[_, _, _, _]] should "copy successfully" in {
    val tr = new TernaryLambdaTransformer[Text, Text, Text, Text](operationName = "foo", transformFn = (x, y, z) => x)
    tr.copy(new ParamMap()).uid shouldBe tr.uid
  }

}
