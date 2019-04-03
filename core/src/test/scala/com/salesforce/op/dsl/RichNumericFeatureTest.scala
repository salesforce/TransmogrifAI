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

package com.salesforce.op.dsl

import com.salesforce.op.features.types._
import com.salesforce.op.test.FeatureTestBase
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.utils.numeric.Number

@RunWith(classOf[JUnitRunner])
class RichNumericFeatureTest extends FlatSpec with FeatureTestBase with RichNumericFeature {

  // Value generators
  private final val doubleGen = Gen.choose(Double.MinValue, Double.MaxValue)
  private final val doubleTupleGen = for { x <- doubleGen; y <- doubleGen } yield x -> y

  Spec[RichNumericFeature[_]] should "divide numbers" in {
    val checkDivision = testOp[Real, Real, Real](x => y => x / y)

    checkDivision of(5.0.toReal, 2.0.toReal) expecting 2.5.toReal
    checkDivision of(Real.empty, Real.empty) expecting Real.empty
    checkDivision of(Real.empty, 1.1.toReal) expecting Real.empty
    checkDivision of(1.1.toReal, Real.empty) expecting Real.empty
    checkDivision of(1.0.toReal, 0.toReal) expecting Real.empty
  }

  it should "multiply numbers" in {
    val checkMultiplication = testOp[Real, Real, Real](x => y => x * y)

    checkMultiplication of(5.0.toReal, 2.0.toReal) expecting 10.toReal
    checkMultiplication of(Real.empty, Real.empty) expecting Real.empty
    checkMultiplication of(Real.empty, 1.1.toReal) expecting Real.empty
    checkMultiplication of(1.1.toReal, Real.empty) expecting Real.empty
    checkMultiplication of(1.0.toReal, 0.0.toReal) expecting 0.toReal
    checkMultiplication of(Double.MaxValue.toReal, 0.toReal) expecting 0.toReal
    checkMultiplication of(Double.MinValue.toReal, 0.toReal) expecting 0.toReal
    checkMultiplication of(Double.MaxValue.toReal, Double.MaxValue.toReal) expecting Real.empty
    checkMultiplication of(Double.MinValue.toReal, Double.MaxValue.toReal) expecting Real.empty
    checkMultiplication of(Double.MaxValue.toReal, Double.MinValue.toReal) expecting Real.empty
    checkMultiplication of(Double.MinValue.toReal, Double.MinValue.toReal) expecting Real.empty
    // scalastyle:off line.size.limit
    checkMultiplication of(Double.MaxValue.toReal, Double.MinPositiveValue.toReal) expecting 8.881784197001251E-16.toReal
    checkMultiplication of(Double.MinValue.toReal, Double.MinPositiveValue.toReal) expecting (-8.881784197001251E-16).toReal
    // scalastyle:on
  }

  it should "add numbers" in {
    val checkAddition = testOp[Real, Real, Real](x => y => x + y)

    checkAddition of(5.0.toReal, 2.0.toReal) expecting 7.0.toReal
    checkAddition of(Real.empty, Real.empty) expecting Real.empty
    checkAddition of(Real.empty, 1.1.toReal) expecting 1.1.toReal
    checkAddition of(1.1.toReal, Real.empty) expecting 1.1.toReal
    checkAddition of(1.0.toReal, 0.25.toReal) expecting 1.25.toReal
    checkAddition of(Double.MaxValue.toReal, Double.MaxValue.toReal) expecting Double.PositiveInfinity.toReal
    checkAddition of(Double.MinValue.toReal, Double.MinValue.toReal) expecting Double.NegativeInfinity.toReal
    checkAddition of(Double.MinValue.toReal, Double.MaxValue.toReal) expecting 0.toReal
  }

  it should "subtract numbers" in {
    val checkSubtraction = testOp[Real, Real, Real](x => y => x - y)

    checkSubtraction of(5.0.toReal, 2.0.toReal) expecting 3.0.toReal
    checkSubtraction of(Real.empty, Real.empty) expecting Real.empty
    checkSubtraction of(Real.empty, 1.1.toReal) expecting (-1.1).toReal
    checkSubtraction of(1.1.toReal, Real.empty) expecting 1.1.toReal
    checkSubtraction of(1.0.toReal, 0.25.toReal) expecting 0.75.toReal
    checkSubtraction of(Double.MaxValue.toReal, Double.MaxValue.toReal) expecting 0.toReal
    checkSubtraction of(Double.MinValue.toReal, Double.MinValue.toReal) expecting 0.toReal
    checkSubtraction of(Double.MinValue.toReal, Double.MaxValue.toReal) expecting Double.NegativeInfinity.toReal
    checkSubtraction of(Double.MaxValue.toReal, Double.MinValue.toReal) expecting Double.PositiveInfinity.toReal
  }

  it should "divide by scalar" in {
    forAll(doubleTupleGen) { case (a, b) =>
      val checkDivision = testOp[Real, Real](x => x / b)
      val exp = if (Number.isValid(a / b)) (a / b).toReal else Real.empty
      checkDivision of a.toReal expecting exp
    }
  }

  it should "multiply by scalar" in {
    forAll(doubleTupleGen) { case (a, b) =>
      val checkMultiplication = testOp[Real, Real](x => x * b)
      val exp = if (Number.isValid(a * b)) (a * b).toReal else Real.empty
      checkMultiplication of a.toReal expecting exp
    }
  }

  it should "plus scalar" in {
    forAll(doubleTupleGen) { case (a, b) =>
      testOp[Real, Real](x => x + b) of a.toReal expecting (a + b).toReal
    }
  }

  it should "minus scalar" in {
    forAll(doubleTupleGen) { case (a, b) =>
      testOp[Real, Real](x => x - b) of a.toReal expecting (a - b).toReal
    }
  }

  Spec[RichRealFeature[_]] should "test vectorize()" in {
    // TODO: add vectorize() test
  }

  Spec[RichRealNNFeature] should "perform math functions correctly" in {
    val checkAddition = testOp[RealNN, RealNN, Real](x => y => x + y)
    checkAddition of(5.0.toRealNN, 2.0.toRealNN) expecting 7.0.toReal

    val checkSubtraction = testOp[RealNN, RealNN, Real](x => y => x - y)
    checkSubtraction of(5.0.toRealNN, 2.0.toRealNN) expecting 3.0.toReal

    val checkMultiplication = testOp[RealNN, RealNN, Real](x => y => x * y)
    checkMultiplication of(5.0.toRealNN, 2.0.toRealNN) expecting 10.toReal
    checkMultiplication of(Double.NaN.toRealNN, 2.0.toRealNN) expecting Real.empty

    val checkDivision = testOp[RealNN, RealNN, Real](x => y => x / y)
    checkDivision of(5.0.toRealNN, 2.0.toRealNN) expecting 2.5.toReal
    checkDivision of(2.0.toRealNN, 0.0.toRealNN) expecting Real.empty
  }

  Spec[RichBinaryFeature] should "perform math functions" in {
    val checkAddition = testOp[Binary, Binary, Real](x => y => x + y)
    checkAddition of(true.toBinary, true.toBinary) expecting 2.0.toReal
    checkAddition of(true.toBinary, Binary.empty) expecting 1.0.toReal

    val checkSubtraction = testOp[Binary, Binary, Real](x => y => x - y)
    checkSubtraction of(true.toBinary, true.toBinary) expecting 0.0.toReal
    checkSubtraction of(true.toBinary, Binary.empty) expecting 1.0.toReal

    val checkMultiplication = testOp[Binary, Binary, Real](x => y => x * y)
    checkMultiplication of(true.toBinary, true.toBinary) expecting 1.0.toReal
    checkMultiplication of(true.toBinary, Binary.empty) expecting Real.empty

    val checkDivision = testOp[Binary, Binary, Real](x => y => x / y)
    checkDivision of(true.toBinary, true.toBinary) expecting 1.0.toReal
    checkDivision of(true.toBinary, false.toBinary) expecting Real.empty
  }

  Spec[RichIntegralFeature[_]] should "perform math functions" in {
    val checkAddition = testOp[Integral, Integral, Real](x => y => x + y)
    checkAddition of(5.toIntegral, 2.toIntegral) expecting 7.0.toReal
    checkAddition of(Integral.empty, 2.toIntegral) expecting 2.0.toReal

    val checkSubtraction = testOp[Integral, Integral, Real](x => y => x - y)
    checkSubtraction of(5.toIntegral, 2.toIntegral) expecting 3.0.toReal
    checkSubtraction of(Integral.empty, 2.toIntegral) expecting (-2.0).toReal

    val checkMultiplication = testOp[Integral, Integral, Real](x => y => x * y)
    checkMultiplication of(5.toIntegral, 2.toIntegral) expecting 10.toReal
    checkMultiplication of(Integral.empty, 2.toIntegral) expecting Real.empty

    val checkDivision = testOp[Integral, Integral, Real](x => y => x / y)
    checkDivision of(5.toIntegral, 2.toIntegral) expecting 2.5.toReal
    checkDivision of(2.toIntegral, 0.toIntegral) expecting Real.empty
  }
}
