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
import com.salesforce.op.stages.base.sequence.SequenceModel
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.ml.linalg.Vectors
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class OpTextPivotVectorizerTest
  extends OpEstimatorSpec[OPVector, SequenceModel[Text, OPVector], OpTextPivotVectorizer[Text]] {

  lazy val (inputData, f1, f2) = TestFeatureBuilder("text1", "text2",
    Seq[(Text, Text)](
      ("hello world".toText, "Hello world!".toText),
      ("hello world".toText, "What's up".toText),
      ("good evening".toText, "How are you doing, my friend?".toText),
      ("hello world".toText, "Not bad, my friend.".toText),
      (Text.empty, Text.empty)
    )
  )

  /**
   * Estimator instance to be tested
   */
  override val estimator: OpTextPivotVectorizer[Text] = new OpTextPivotVectorizer()
    .setMinSupport(1)
    .setTopK(2)
    .setInput(f1, f2)

  /**
   * Expected result of the transformer applied on the Input Dataset
   */
  override val expectedResult: Seq[OPVector] = Seq(
    Vectors.sparse(8, Array(0, 4), Array(1.0, 1.0)),
    Vectors.sparse(8, Array(0, 6), Array(1.0, 1.0)),
    Vectors.sparse(8, Array(1, 5), Array(1.0, 1.0)),
    Vectors.sparse(8, Array(0, 6), Array(1.0, 1.0)),
    Vectors.sparse(8, Array(3, 7), Array(1.0, 1.0))
  ).map(_.toOPVector)
}
