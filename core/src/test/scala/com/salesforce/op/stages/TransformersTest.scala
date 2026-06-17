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
    val scaledHeight: FeatureLike[RealNN] = height.zNormalize()
    val history = scaledHeight.history()
    history.stages.head.contains("stdScaled") shouldBe true
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
    val ageExists: FeatureLike[Binary] = age.exists(_.value.contains(100.0))
    val heightReplaced: FeatureLike[RealNN] = height.replaceWith(new RealNN(1.0), new RealNN(2.0))
    val all = Seq(ageMap, heightFilter, ageExists, heightReplaced)

    all.flatMap(_.parents) shouldBe Array(age, height, age, height)
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
    val heightRes: FeatureLike[RealNN] = height.map[RealNN, RealNN, Integer, RealNN](height, height, age,
      (h1, h2, h3, a) => (h1.value.get * h2.value.get + h3.value.get - a.value.getOrElse(0)).toRealNN
    )
    heightRes.parents shouldBe Array(height, height, height, age)
    heightRes.originStage shouldBe a[Transformer]
  }
}

