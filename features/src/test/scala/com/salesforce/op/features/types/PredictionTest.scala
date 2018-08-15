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

package com.salesforce.op.features.types

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class PredictionTest extends FlatSpec with TestCommon {
  import Prediction.Keys._

  Spec[Prediction] should "extend FeatureType" in {
    Prediction(1.0) shouldBe a[FeatureType]
    Prediction(1.0) shouldBe a[OPMap[_]]
    Prediction(1.0) shouldBe a[NumericMap]
    Prediction(1.0) shouldBe a[RealMap]
  }
  it should "error if prediction is missing" in {
    intercept[NonNullableEmptyException](new Prediction(null))
    intercept[NonNullableEmptyException](new Prediction(Map.empty))
    intercept[NonNullableEmptyException](Map.empty[String, Double].toPrediction)
    intercept[NonNullableEmptyException]((null: Map[String, Double]).toPrediction)
    assertPredictionError(new Prediction(Map("a" -> 1.0)))
    assertPredictionError(Map("a" -> 1.0, "b" -> 2.0).toPrediction)
    assertInvalidKeysError(new Prediction(Map(PredictionName -> 2.0, "a" -> 1.0)))
  }
  it should "compare values correctly" in {
    Prediction(1.0).equals(Prediction(1.0)) shouldBe true
    Prediction(1.0).equals(Prediction(0.0)) shouldBe false
    Prediction(1.0, Array(1.0), Array.empty[Double]).equals(Prediction(1.0)) shouldBe false
    Prediction(1.0, Array(1.0), Array(2.0, 3.0)).equals(Prediction(1.0, Array(1.0), Array(2.0, 3.0))) shouldBe true

    Map(PredictionName -> 5.0).toPrediction shouldBe a[Prediction]
  }
  it should "return prediction" in {
    Prediction(2.0).prediction shouldBe 2.0
  }
  it should "return raw prediction" in {
    Prediction(2.0).rawPrediction shouldBe Array()
    Prediction(1.0, Array(1.0, 2.0), Array.empty[Double]).rawPrediction shouldBe Array(1.0, 2.0)
  }
  it should "return probability" in {
    Prediction(3.0).probability shouldBe Array()
    Prediction(1.0, Array.empty[Double], Array(1.0, 2.0)).probability shouldBe Array(1.0, 2.0)
  }
  it should "return score" in {
    Prediction(4.0).score shouldBe Array(4.0)
    Prediction(1.0, Array(2.0, 3.0), Array.empty[Double]).score shouldBe Array(1.0)
    Prediction(1.0, Array.empty[Double], Array(2.0, 3.0)).score shouldBe Array(2.0, 3.0)
  }

  private def assertPredictionError(f: => Unit) =
    intercept[NonNullableEmptyException](f).getMessage shouldBe
      s"Prediction cannot be empty: value map must contain '$PredictionName' key"

  private def assertInvalidKeysError(f: => Unit) =
    intercept[IllegalArgumentException](f).getMessage shouldBe
      s"requirement failed: value map must only contain valid keys: '$PredictionName' or " +
        s"starting with '$RawPredictionName' or '$ProbabilityName'"

}
