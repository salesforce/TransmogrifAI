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

package com.salesforce.op.stages.base.quaternary

import com.salesforce.op.features.types._
import com.salesforce.op.test._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class QuaternaryTransformerTest
  extends OpTransformerSpec[Real, QuaternaryTransformer[Real, Integral, Text, Binary, Real]] {

  val sample = Seq(
    (Real(1.0), Integral(0), Text("abc"), Binary(false)),
    (Real(2.0), Integral(2), Text("a"), Binary(true)),
    (Real.empty, Integral(3), Text("abcdefg"), Binary(true))
  )

  val (inputData, f1, f2, f3, f4) = TestFeatureBuilder(sample)

  val transformer = new QuaternaryLambdaTransformer[Real, Integral, Text, Binary, Real](
    operationName = "quatro", transformFn = Lambda.fn
  ).setInput(f1, f2, f3, f4)

  val expectedResult = Seq(4.toReal, 6.toReal, 11.toReal)

}

object Lambda {
  def fn: (Real, Integral, Text, Binary) => Real = (r, i, t, b) =>
    (r.v.getOrElse(0.0) + i.toDouble.getOrElse(0.0) + b.toDouble.getOrElse(0.0) +
      t.value.map(_.length.toDouble).getOrElse(0.0)).toReal
}
