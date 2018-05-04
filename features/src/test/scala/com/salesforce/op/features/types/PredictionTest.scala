/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
