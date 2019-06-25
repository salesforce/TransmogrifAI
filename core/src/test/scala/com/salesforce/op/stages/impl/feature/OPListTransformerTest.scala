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


import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OPListTransformerTest
  extends OpTransformerSpec[TextList, OPListTransformer[Text, Text, TextList, TextList]] {

  lazy val (inputData, top) = TestFeatureBuilder("name", Seq(
    Seq("A", "B").toTextList
  ))

  val transformer: OPListTransformer[Text, Text, TextList, TextList] =
    new LowerCaseListTransformer().setInput(top)

  val expectedResult: Seq[TextList] = Seq(
    Seq("a", "b").toTextList
  )
}

class LowerCaseTransformer extends UnaryTransformer[Text, Text](
  operationName = "lowerCaseUnary",
  uid = UID[LowerCaseTransformer]
) {
  def transformFn: (Text => Text) = (input: Text) => input.value.map(_.toLowerCase()).toText
}

class LowerCaseListTransformer
(
  uid: String = UID[LowerCaseListTransformer],
  operationName: String = "lowerCaseList"
) extends OPListTransformer[Text, Text, TextList, TextList](
  uid = uid,
  operationName = operationName,
  transformer = new LowerCaseTransformer
)
