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

package com.salesforce.op.stages.base.ternary

import com.salesforce.op.features.types._
import com.salesforce.op.test._
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TernaryTransformerTest extends OpTransformerSpec[Real, TernaryTransformer[Real, Integral, Binary, Real]] {

  val sample = Seq(
    (Real(1.0), Integral(0), Binary(false)),
    (Real(2.0), Integral(2), Binary(true)),
    (Real.empty, Integral(3), Binary(true))
  )

  val (inputData, f1, f2, f3) = TestFeatureBuilder(sample)

  val transformer = new TernaryLambdaTransformer[Real, Integral, Binary, Real](
    operationName = "trio", transformFn = new TernaryTransformerTest.Fun
  ).setInput(f1, f2, f3)

  val expectedResult = Seq(1.toReal, 5.toReal, 4.toReal)

}

object TernaryTransformerTest {

  class Fun extends Function3[Real, Integral, Binary, Real] with Serializable {
    def apply(r: Real, i: Integral, b: Binary): Real =
      (r.v.getOrElse(0.0) + i.toDouble.getOrElse(0.0) + b.toDouble.getOrElse(0.0)).toReal
  }

}
