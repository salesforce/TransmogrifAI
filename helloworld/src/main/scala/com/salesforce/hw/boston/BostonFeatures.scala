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

package com.salesforce.hw.boston

import com.salesforce.hw.boston.BostonFeatures._
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait BostonFeatures extends Serializable {
  val rowId = FeatureBuilder.Integral[BostonHouse].extract(new RowId).asPredictor
  val crim = FeatureBuilder.RealNN[BostonHouse].extract(new Crim).asPredictor
  val zn = FeatureBuilder.RealNN[BostonHouse].extract(new Zn).asPredictor
  val indus = FeatureBuilder.RealNN[BostonHouse].extract(new Indus).asPredictor
  val chas = FeatureBuilder.PickList[BostonHouse].extract(new Chas).asPredictor
  val nox = FeatureBuilder.RealNN[BostonHouse].extract(new Nox).asPredictor
  val rm = FeatureBuilder.RealNN[BostonHouse].extract(new RM).asPredictor
  val age = FeatureBuilder.RealNN[BostonHouse].extract(new Age).asPredictor
  val dis = FeatureBuilder.RealNN[BostonHouse].extract(new Dis).asPredictor
  val rad = FeatureBuilder.Integral[BostonHouse].extract(new Rad).asPredictor
  val tax = FeatureBuilder.RealNN[BostonHouse].extract(new Tax).asPredictor
  val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(new PTRatio).asPredictor
  val b = FeatureBuilder.RealNN[BostonHouse].extract(new B).asPredictor
  val lstat = FeatureBuilder.RealNN[BostonHouse].extract(new Lstat).asPredictor
  val medv = FeatureBuilder.RealNN[BostonHouse].extract(new Medv).asResponse
}

object BostonFeatures {

  abstract class BostonFeatureFunc[T] extends Function[BostonHouse, T] with Serializable

  class RealNNExtract(f: BostonHouse => Double) extends BostonFeatureFunc[RealNN] {
    override def apply(v1: BostonHouse): RealNN = f(v1).toRealNN
  }

  class IntegralExtract(f: BostonHouse => Int) extends BostonFeatureFunc[Integral] {
    override def apply(v1: BostonHouse): Integral = f(v1).toIntegral
  }

  class RowId extends IntegralExtract(_.rowId)

  class Rad extends IntegralExtract(_.rad)

  class Crim extends RealNNExtract(_.crim)

  class Zn extends RealNNExtract(_.zn)

  class Indus extends RealNNExtract(_.indus)

  class Nox extends RealNNExtract(_.nox)

  class RM extends RealNNExtract(_.rm)

  class Age extends RealNNExtract(_.age)

  class Dis extends RealNNExtract(_.dis)

  class Tax extends RealNNExtract(_.tax)

  class PTRatio extends RealNNExtract(_.ptratio)

  class B extends RealNNExtract(_.b)

  class Lstat extends RealNNExtract(_.lstat)

  class Medv extends RealNNExtract(_.medv)

  class Chas extends BostonFeatureFunc[PickList] {
    override def apply(v1: BostonHouse): PickList = Option(v1.chas).toPickList
  }
}
