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
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class AddTransformerTest extends OpTransformerSpec[Real, AddTransformer[Real, Real]] {
  val sample = Seq((Real(1.0), Real(2.0)), (Real(4.0), Real(4.0)), (Real.empty, Real(5.0)),
    (Real(5.0), Real.empty), (Real(2.0), Real(0.0)))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer: AddTransformer[Real, Real] = new AddTransformer[Real, Real]().setInput(f1, f2)
  override val expectedResult: Seq[Real] = Seq(Real(3.0), Real(8.0), Real(5.0), Real(5.0), Real(2.0))
}

@RunWith(classOf[JUnitRunner])
class ScalarAddTransformerTest extends OpTransformerSpec[Real, ScalarAddTransformer[Real, Double]] {
  val sample = Seq(Real(1.0), Real(4.0), Real.empty, Real(-1.0), Real(2.0))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: ScalarAddTransformer[Real, Double] = new ScalarAddTransformer[Real, Double](5.0)
    .setInput(f1)
  override val expectedResult: Seq[Real] = Seq(Real(6.0), Real(9.0), Real.empty, Real(4.0), Real(7.0))
}

@RunWith(classOf[JUnitRunner])
class SubtractTransformerTest extends OpTransformerSpec[Real, SubtractTransformer[Real, Real]] {
  val sample = Seq((Real(1.0), Real(2.0)), (Real(4.0), Real(4.0)), (Real.empty, Real(5.0)),
    (Real(5.0), Real.empty), (Real(2.0), Real(0.0)))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer: SubtractTransformer[Real, Real] = new SubtractTransformer[Real, Real]().setInput(f1, f2)
  override val expectedResult: Seq[Real] = Seq(Real(-1.0), Real(0.0), Real(-5.0), Real(5.0), Real(2.0))
}

@RunWith(classOf[JUnitRunner])
class ScalarSubtractTransformerTest extends OpTransformerSpec[Real, ScalarSubtractTransformer[Real, Double]] {
  val sample = Seq(Real(1.0), Real(4.0), Real.empty, Real(-1.0), Real(2.0))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: ScalarSubtractTransformer[Real, Double] = new ScalarSubtractTransformer[Real, Double](5.0)
    .setInput(f1)
  override val expectedResult: Seq[Real] = Seq(Real(-4.0), Real(-1.0), Real.empty, Real(-6.0), Real(-3.0))
}


@RunWith(classOf[JUnitRunner])
class MultiplyTransformerTest extends OpTransformerSpec[Real, MultiplyTransformer[Real, Real]] {
  val sample = Seq((Real(1.0), Real(2.0)), (Real(4.0), Real(4.0)), (Real.empty, Real(5.0)),
    (Real(5.0), Real.empty), (Real(2.0), Real(0.0)))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer: MultiplyTransformer[Real, Real] = new MultiplyTransformer[Real, Real]().setInput(f1, f2)
  override val expectedResult: Seq[Real] = Seq(Real(2.0), Real(16.0), Real.empty, Real.empty, Real(0.0))
}

@RunWith(classOf[JUnitRunner])
class ScalarMultiplyTransformerTest extends OpTransformerSpec[Real, ScalarMultiplyTransformer[Real, Double]] {
  val sample = Seq(Real(1.0), Real(4.0), Real.empty, Real(-1.0), Real(2.0))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: ScalarMultiplyTransformer[Real, Double] = new ScalarMultiplyTransformer[Real, Double](5.0)
    .setInput(f1)
  override val expectedResult: Seq[Real] = Seq(Real(5.0), Real(20.0), Real.empty, Real(-5.0), Real(10.0))
}

@RunWith(classOf[JUnitRunner])
class DivideTransformerTest extends OpTransformerSpec[Real, DivideTransformer[Real, Real]] {
  val sample = Seq((Real(1.0), Real(2.0)), (Real(4.0), Real(4.0)), (Real.empty, Real(5.0)),
    (Real(5.0), Real.empty), (Real(2.0), Real(0.0)))
  val (inputData, f1, f2) = TestFeatureBuilder(sample)
  val transformer: DivideTransformer[Real, Real] = new DivideTransformer[Real, Real]().setInput(f1, f2)
  override val expectedResult: Seq[Real] = Seq(Real(0.5), Real(1.0), Real.empty, Real.empty, Real.empty)
}

@RunWith(classOf[JUnitRunner])
class ScalarDivideTransformerTest extends OpTransformerSpec[Real, ScalarDivideTransformer[Real, Double]] {
  val sample = Seq(Real(1.0), Real(4.0), Real.empty, Real(-1.0), Real(2.0))
  val (inputData, f1) = TestFeatureBuilder(sample)
  val transformer: ScalarDivideTransformer[Real, Double] = new ScalarDivideTransformer[Real, Double](2.0)
    .setInput(f1)
  override val expectedResult: Seq[Real] = Seq(Real(0.5), Real(2.0), Real.empty, Real(-0.5), Real(1.0))
}





