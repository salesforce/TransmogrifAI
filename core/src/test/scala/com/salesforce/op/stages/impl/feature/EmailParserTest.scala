/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryLambdaTransformer
import com.salesforce.op.test.{TestFeatureBuilder, _}
import com.salesforce.op.utils.spark.RichDataset._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class EmailParserTest extends FlatSpec with TestCommon with TestSparkContext {

  val (df, email) = TestFeatureBuilder("email", Seq(
    Email("test@example.com"),
    Email("@example.com"),
    Email("test@"),
    Email("@"),
    Email(""),
    Email("notanemail"),
    Email.empty,
    Email("first.last@example.com")
  ))

  "Email Extraction" should "extract prefix from simple email addresses" in {
    val prefix = email.toEmailPrefix
    val result = prefix.originStage.asInstanceOf[UnaryLambdaTransformer[Email, Text]].transform(df)

    result.collect(prefix) should contain theSameElementsInOrderAs
      Seq(Text("test"), Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text("first.last"))
  }

  it should "extract domain from simple email addresses" in {
    val domain = email.toEmailDomain
    val result = domain.originStage.asInstanceOf[UnaryLambdaTransformer[Email, Text]].transform(df)

    result.collect(domain) should contain theSameElementsInOrderAs
      Seq(Text("example.com"), Text.empty, Text.empty, Text.empty, Text.empty, Text.empty, Text.empty,
        Text("example.com"))
  }
}
