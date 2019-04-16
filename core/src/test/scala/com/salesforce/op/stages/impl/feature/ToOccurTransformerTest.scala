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

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.ml.Transformer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class ToOccurTransformerTest extends OpTransformerSpec[RealNN, ToOccurTransformer[Real]] {

  val testData = Seq(2.0.toReal, 0.0.toReal, Real(None))
  val (inputData, f1) = TestFeatureBuilder(testData)
  val expectedResult = Seq(1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN)

  val transformer: ToOccurTransformer[Real] = new ToOccurTransformer[Real]().setInput(f1)

  val extendedTestData = Seq(
    ("001", 0, None, Option(true), None),
    ("002", 1, None, None, Option(2.0)),
    ("003", 2, Option("abc"), Option(false), Option(0.0)),
    ("004", 0, Option("def"), Option(false), Option(1.0))
  ).map { case (leadId, numEmails, opptyId, doNotContact, numFormSubmits) =>
    (
      Text(leadId),
      numEmails.toRealNN,
      Text(opptyId),
      Binary(doNotContact),
      Real(numFormSubmits)
    )
  }

  lazy val (ds, leadId, numEmails, opptyId, doNotContact, numFormSubmits) =
    TestFeatureBuilder("leadId", "numEmails", "opptyId", "doNotContact", "numFormSubmits", extendedTestData)

  Spec[ToOccurTransformer[_]] should "convert features to boolean using shortcuts" in {
    val occurEmailOutput = numEmails.occurs(_.value.exists(_ > 1))
    val toOccurEmail = occurEmailOutput.originStage.asInstanceOf[Transformer]
    val occurFormOutput = numFormSubmits.occurs()
    val toOccurForm = occurFormOutput.originStage.asInstanceOf[Transformer]
    val occurContactOutput = doNotContact.occurs()
    val toOccurContact = occurContactOutput.originStage.asInstanceOf[Transformer]

    val toOccurDF = toOccurContact.transform(toOccurForm.transform(toOccurEmail.transform(ds)))

    val expected = Array(
      (Text("001"), 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN),
      (Text("002"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN),
      (Text("003"), 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN),
      (Text("004"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN)
    )

    toOccurDF.orderBy("leadId").collect(leadId, occurEmailOutput, occurFormOutput, occurContactOutput) shouldBe expected
  }

  it should "convert features to doolean" in {
    val toOccurEmail = new ToOccurTransformer[RealNN]().setInput(numEmails)
    val occurEmailOutput = toOccurEmail.getOutput()
    val toOccurForm = new ToOccurTransformer[Real]().setInput(numFormSubmits)
    val occurFormOutput = toOccurForm.getOutput()
    val toOccurContact = new ToOccurTransformer[Binary]().setInput(doNotContact)
    val occurContactOutput = toOccurContact.getOutput()
    val toOccurOppty = new ToOccurTransformer[Text](matchFn = _.nonEmpty).setInput(opptyId)
    val occurOpptyOutput = toOccurOppty.getOutput()

    val toOccurDF = toOccurOppty.transform(toOccurContact.transform(toOccurForm.transform(toOccurEmail.transform(ds))))

    val expected = Array(
      (Text("001"), 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN),
      (Text("002"), 1.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN),
      (Text("003"), 1.0.toRealNN, 0.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN),
      (Text("004"), 0.0.toRealNN, 1.0.toRealNN, 0.0.toRealNN, 1.0.toRealNN)
    )

    toOccurDF.orderBy("leadId").collect(leadId,
      occurEmailOutput, occurFormOutput, occurContactOutput, occurOpptyOutput) shouldBe expected
  }
}
