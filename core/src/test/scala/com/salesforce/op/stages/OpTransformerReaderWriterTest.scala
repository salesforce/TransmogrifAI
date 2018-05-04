/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpTransformerReaderWriterTest extends OpPipelineStageReaderWriterTest {

  override val hasOutputName = false

  val stage: OpPipelineStageBase =
    new UnaryLambdaTransformer[Real, Real](
      operationName = "test",
      transformFn = _.v.map(_ * 0.1234).toReal,
      uid = UID[UnaryLambdaTransformer[_, _]]
    ).setInput(weight).setMetadata(meta)

  val expected = Array(21.2248.toReal, Real.empty, 9.6252.toReal, 8.2678.toReal, 11.8464.toReal, 8.2678.toReal)
}
