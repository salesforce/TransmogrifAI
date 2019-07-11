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

package com.salesforce.op.stages.base.sequence

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class BinarySequenceTransformerTest
  extends OpTransformerSpec[MultiPickList, BinarySequenceTransformer[Real, Text, MultiPickList]] {

  val sample = Seq(
    (1.toReal, "one".toText, "two".toText),
    ((-1).toReal, "three".toText, "four".toText),
    (15.toReal, "five".toText, "six".toText),
    (1.111.toReal, "seven".toText, "".toText)
  )

  val (inputData, f1, f2, f3) = TestFeatureBuilder(sample)

  val transformer = new BinarySequenceLambdaTransformer[Real, Text, MultiPickList](
    operationName = "realToMultiPicklist", transformFn = new BinarySequenceTransformerTest.Fun
  ).setInput(f1, f2, f3)

  val expectedResult = Seq(
    Set("1.0", "one", "two"),
    Set("-1.0", "three", "four"),
    Set("15.0", "five", "six"),
    Set("1.111", "seven", "")
  ).map(_.toMultiPickList)
}

object BinarySequenceTransformerTest {

  class Fun extends Function2[Real, Seq[Text], MultiPickList] with Serializable {
    def apply(r: Real, texts: Seq[Text]): MultiPickList =
      MultiPickList(texts.map(_.value.get).toSet + r.value.get.toString)
  }

}
