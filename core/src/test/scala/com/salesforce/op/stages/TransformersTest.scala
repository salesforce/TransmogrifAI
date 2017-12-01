/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages

import com.salesforce.op._
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.test.PassengerFeaturesTest
import org.apache.spark.ml.{Estimator, Transformer}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}


@RunWith(classOf[JUnitRunner])
class TransformersTest extends FlatSpec with Matchers with PassengerFeaturesTest {

  "Transformers" should "allow division between numerics and nullable numerics variations" in {
    val ageHeight: FeatureLike[Real] = age / height
    val heightAge: FeatureLike[Real] = height / age
    val ageAge: FeatureLike[Real] = age / age
    val heightHeight: FeatureLike[Real] = height / height

    Seq(ageHeight, heightAge, ageAge, heightHeight)
      .foreach(_.history().stages.head contains "divide" shouldBe true)
    ageHeight.parents shouldBe Array(age, height)
    ageHeight.originStage shouldBe a[Transformer]
    heightAge.parents shouldBe Array(height, age)
    heightAge.originStage shouldBe a[Transformer]
    ageAge.parents shouldBe Array(age, age)
    ageAge.originStage shouldBe a[Transformer]
    heightHeight.parents shouldBe Array(height, height)
    heightHeight.originStage shouldBe a[Transformer]
  }
  it should "scaling numeric values" in {
    val scaledHeight: FeatureLike[RealNN] = height.map[RealNN](_.toRealNN()).zNormalize()
    val history = scaledHeight.history()
    history.stages.head.contains("map") shouldBe true
    history.stages.last.contains("stdScaled") shouldBe true
    scaledHeight.originStage shouldBe a[Estimator[_]]
  }
  it should "allow multiplication between numerics and nullable numerics variations" in {
    val ageHeight: FeatureLike[Real] = age * height
    val heightAge: FeatureLike[Real] = height * age
    val ageAge: FeatureLike[Real] = age * age
    val heightHeight: FeatureLike[Real] = height * height

    Seq(ageHeight, heightAge, ageAge, heightHeight)
      .foreach(_.history().stages.head contains "multiply" shouldBe true)
    ageHeight.parents shouldBe Array(age, height)
    ageHeight.originStage shouldBe a[Transformer]
    heightAge.parents shouldBe Array(height, age)
    heightAge.originStage.isInstanceOf[Transformer]
    ageAge.parents shouldBe Array(age, age)
    ageAge.originStage shouldBe a[Transformer]
    heightHeight.parents shouldBe Array(height, height)
    heightHeight.originStage shouldBe a[Transformer]
  }
  it should "allow addition between numerics and nullable numerics variations" in {
    val ageHeight: FeatureLike[Real] = age + height
    val heightAge: FeatureLike[Real] = height + age
    val ageAge: FeatureLike[Real] = age + age
    val heightHeight: FeatureLike[Real] = height + height

    Seq(ageHeight, heightAge, ageAge, heightHeight)
      .foreach(_.history().stages.head contains "plus" shouldBe true)
    ageHeight.parents shouldBe Array(age, height)
    ageHeight.originStage shouldBe a[Transformer]
    heightAge.parents shouldBe Array(height, age)
    heightAge.originStage shouldBe a[Transformer]
    ageAge.parents shouldBe Array(age, age)
    ageAge.originStage shouldBe a[Transformer]
    heightHeight.parents shouldBe Array(height, height)
    heightHeight.originStage shouldBe a[Transformer]
  }
  it should "allow subtraction between numerics and nullable numerics variations" in {
    val ageHeight: FeatureLike[Real] = age - height
    val heightAge: FeatureLike[Real] = height - age
    val ageAge: FeatureLike[Real] = age - age
    val heightHeight: FeatureLike[Real] = height - height

    Seq(ageHeight, heightAge, ageAge, heightHeight)
      .foreach(_.history().stages.head contains "minus" shouldBe true)
    ageHeight.parents shouldBe Array(age, height)
    ageHeight.originStage shouldBe a[Transformer]
    heightAge.parents shouldBe Array(height, age)
    heightAge.originStage shouldBe a[Transformer]
    ageAge.parents shouldBe Array(age, age)
    ageAge.originStage shouldBe a[Transformer]
    heightHeight.parents shouldBe Array(height, height)
    heightHeight.originStage shouldBe a[Transformer]
  }
  it should "allow interactions between numerics and constants" in {
    val ageHeight: FeatureLike[Real] = age + height + 1
    val heightAge: FeatureLike[Real] = (height / age) - 1
    val ageAge: FeatureLike[Real] = (age - age) * 2
    val heightHeight: FeatureLike[Real] = (height - height) / 2

    ageHeight.originStage shouldBe a[Transformer]
    heightAge.originStage shouldBe a[Transformer]
    ageAge.originStage shouldBe a[Transformer]
    heightHeight.originStage shouldBe a[Transformer]
  }
  it should "allow applying generic feature unary transformations" in {
    val ageMap: FeatureLike[Text] = age.map[Text](_.value.map(_.toString).toText)
    val heightFilter: FeatureLike[RealNN] =
      height.filter(_.value.contains(100.0), default = new RealNN(Double.MinValue))
    val heightFilterNot: FeatureLike[RealNN] =
      height.filterNot(_.value.contains(100.0), default = new RealNN(0.0))
    val heightCollect: FeatureLike[Real] =
      height.collect[Real](Real.empty){ case r if r.v.contains(100.0) => Real(123.0) }
    val ageExists: FeatureLike[Binary] = age.exists(_.value.contains(100.0))
    val heightReplaced: FeatureLike[RealNN] = height.replaceWith(new RealNN(1.0), new RealNN(2.0))
    val all = Seq(ageMap, heightFilter, heightFilterNot, heightCollect, ageExists, heightReplaced)

    all.flatMap(_.parents) shouldBe Array(age, height, height, height, age, height)
    all.forall(_.originStage.isInstanceOf[Transformer]) shouldBe true
  }
  it should "allow applying generic feature binary transformations" in {
    val heightSquare: FeatureLike[RealNN] = height.map[RealNN, RealNN](height, (l, r) =>
      (l.value.get / r.value.get).toRealNN
    )
    heightSquare.parents shouldBe Array(height, height)
    heightSquare.originStage shouldBe a[Transformer]
  }
  it should "allow applying generic feature ternary transformations" in {
    val heightRes: FeatureLike[RealNN] = height.map[RealNN, RealNN, RealNN](height, height, (l, r, z) =>
      (l.value.get * r.value.get + z.value.get).toRealNN
    )
    heightRes.parents shouldBe Array(height, height, height)
    heightRes.originStage shouldBe a[Transformer]
  }
  it should "allow applying generic feature quaternary transformations" in {
    val heightRes: FeatureLike[RealNN] = height.map[RealNN, RealNN, Real, RealNN](height, height, age,
      (h1, h2, h3, a) => (h1.value.get * h2.value.get + h3.value.get - a.value.getOrElse(0.0)).toRealNN
    )
    heightRes.parents shouldBe Array(height, height, height, age)
    heightRes.originStage shouldBe a[Transformer]
  }
}

