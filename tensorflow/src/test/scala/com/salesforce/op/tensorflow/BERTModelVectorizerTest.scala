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

package com.salesforce.op.tensorflow

import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpTransformerSpec, TestFeatureBuilder}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class BERTModelVectorizerTest extends OpTransformerSpec[OPVector, BERTModelVectorizer] {

  lazy val (inputData, f1) = TestFeatureBuilder(Seq(
    "I like apples".toText,
    "But what about beer?".toText,
    "Beer is fine... But cider is probably better. Let's go get some!".toText,
    Text.empty
  ))

  lazy val bertModelResource = "com/robrua/nlp/easy-bert/bert-uncased-L-12-H-768-A-12"

  lazy val bertLoader = new BERTModelResourceLoader(bertModelResource)

  lazy val transformer = new BERTModelVectorizer(bertLoader).setInput(f1)

  val expectedResult: Seq[OPVector] = Seq(OPVector.empty)

  Spec[BERTModelResourceLoader] should "create the BERT model" in {
    val bertModel = bertLoader.model
    bertModel.config shouldBe BERTModelConfig(
      doLowerCase = true,
      inputIds = "input_ids:0",
      inputMask = "input_mask:0",
      segmentIds = "segment_ids:0",
      pooledOutput = "module_apply_tokens/bert/pooler/dense/Tanh:0",
      sequenceOutput = "module_apply_tokens/bert/encoder/Reshape_13:0",
      maxSequenceLength = 128
    )
    bertModel.modelBundle should not be null
    bertModel.tokenizer should not be null
  }

}
