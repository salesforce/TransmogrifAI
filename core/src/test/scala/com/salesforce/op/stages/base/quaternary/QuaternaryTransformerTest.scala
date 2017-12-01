/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.base.quaternary

import com.salesforce.op.features.types.Text
import com.salesforce.op.test._
import org.apache.spark.ml.param.ParamMap
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertions, FlatSpec, Matchers}

@RunWith(classOf[JUnitRunner])
class QuaternaryTransformerTest extends FlatSpec with TestCommon {

  Spec[QuaternaryLambdaTransformer[_, _, _, _ , _]] should "copy successfully" in {
    val tr = new QuaternaryLambdaTransformer[Text, Text, Text, Text, Text](
      operationName = "foo",
      transformFn = (x, y, z, u) => x
    )
    tr.copy(new ParamMap()).uid shouldBe tr.uid
  }

}
