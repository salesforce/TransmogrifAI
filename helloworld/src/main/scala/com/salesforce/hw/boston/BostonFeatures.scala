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

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

trait BostonFeatures extends Serializable {

  val rowId = FeatureBuilder.Integral[BostonHouse].extract(_.rowId.toIntegral).asPredictor

  val crim = FeatureBuilder.RealNN[BostonHouse].extract(_.crim.toRealNN).asPredictor

  val zn = FeatureBuilder.RealNN[BostonHouse].extract(_.zn.toRealNN).asPredictor

  val indus = FeatureBuilder.RealNN[BostonHouse].extract(_.indus.toRealNN).asPredictor

  val chas = FeatureBuilder.PickList[BostonHouse].extract(x => Option(x.chas).toPickList).asPredictor

  val nox = FeatureBuilder.RealNN[BostonHouse].extract(_.nox.toRealNN).asPredictor

  val rm = FeatureBuilder.RealNN[BostonHouse].extract(_.rm.toRealNN).asPredictor

  val age = FeatureBuilder.RealNN[BostonHouse].extract(_.age.toRealNN).asPredictor

  val dis = FeatureBuilder.RealNN[BostonHouse].extract(_.dis.toRealNN).asPredictor

  val rad = FeatureBuilder.Integral[BostonHouse].extract(_.rad.toIntegral).asPredictor

  val tax = FeatureBuilder.RealNN[BostonHouse].extract(_.tax.toRealNN).asPredictor

  val ptratio = FeatureBuilder.RealNN[BostonHouse].extract(_.ptratio.toRealNN).asPredictor

  val b = FeatureBuilder.RealNN[BostonHouse].extract(_.b.toRealNN).asPredictor

  val lstat = FeatureBuilder.RealNN[BostonHouse].extract(_.lstat.toRealNN).asPredictor

  val medv = FeatureBuilder.RealNN[BostonHouse].extract(_.medv.toRealNN).asResponse

}
