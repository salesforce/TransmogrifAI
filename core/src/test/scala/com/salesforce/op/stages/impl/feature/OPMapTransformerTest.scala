/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.features.types.{Email, EmailMap, Integral, IntegralMap, Real, _}
import com.salesforce.op.stages.base.unary.{UnaryLambdaTransformer, UnaryTransformer}
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.stages.impl.feature.OPMapTransformerTest.TransformerType
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder, TestSparkContext}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

/**
 * @author ksuchanek
 * @since 214
 */
@RunWith(classOf[JUnitRunner])
class OPMapTransformerTest extends OpTransformerSpec[IntegralMap, TransformerType] {

  import OPMapTransformerTest._

  lazy val (dataEmailMap, top) = TestFeatureBuilder("name",
    Seq(
      Map("p1" -> "a@abcd.com", "p2" -> "xy@abcd.com")
    ).map(EmailMap(_))
  )

  /**
   * [[OpTransformer]] instance to be tested
   */
  override val transformer: TransformerType = new TransformerType(
    transformer = new BaseTransformer(),
    operationName = "testUnaryMapWrap").setInput(top)

  /**
   * Input Dataset to transform
   */
  override val inputData: Dataset[_] = dataEmailMap
  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  override val expectedResult: Seq[IntegralMap] = Seq(
    IntegralMap(Map("p1" -> 10L, "p2" -> 11L))
  )
}

object OPMapTransformerTest {
  type TransformerType = OPMapTransformer[Email, Integral, EmailMap, IntegralMap]

  class BaseTransformer extends UnaryTransformer[Email, Integral](
    operationName = "testUnary",
    uid = "1234"
  ) {
    override def transformFn: (Email => Integral) = (input: Email) => input.value.map(_.length).toIntegral
  }

}
