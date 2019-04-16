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

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}

@RunWith(classOf[JUnitRunner])
class SubstringTransformerTest extends OpTransformerSpec[Binary, SubstringTransformer[Text, Text]] {

  val sample = Seq((Text("a"), Text("abc")), (Text("abc"), Text("a")), (Text("no"), Text("YesNO")),
    (Text.empty, Text("blah")), (Text.empty, Text("blah")), (Text.empty, Text.empty))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer: SubstringTransformer[Text, Text] = new SubstringTransformer[Text, Text]().setInput(f1, f2)
  val expectedResult: Seq[Binary] = Seq(Binary(true), Binary(false), Binary(true), Binary.empty,
    Binary.empty, Binary.empty)

  it should "have a working shortcut" in {
    val f3 = f1.isSubstring(f2)
    f3.originStage.isInstanceOf[SubstringTransformer[_, _]] shouldBe true
  }

}

