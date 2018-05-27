/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.features

import java.util

import com.salesforce.op.aggregators._
import com.salesforce.op.features.types._
import com.salesforce.op.stages.FeatureGeneratorStage
import com.salesforce.op.test.{Passenger, TestCommon}
import com.twitter.algebird.MonoidAggregator
import org.apache.spark.sql.Row
import org.joda.time.Duration
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.runtime.universe._


@RunWith(classOf[JUnitRunner])
class FeatureBuilderTest extends FlatSpec with TestCommon {
  private val name = "feature"
  private val passenger =
    Passenger.newBuilder()
      .setPassengerId(0).setGender("Male").setAge(1).setBoarded(2).setHeight(3)
      .setWeight(4).setDescription("").setSurvived(1).setRecordDate(4)
      .setStringMap(new util.HashMap[String, String]())
      .setNumericMap(new util.HashMap[String, java.lang.Double]())
      .setBooleanMap(new util.HashMap[String, java.lang.Boolean]())
      .build()

  Spec(FeatureBuilder.getClass) should "build a simple feature with a custom name" in {
    val f1 = FeatureBuilder.Real[Passenger]("a").extract(p => Option(p.getAge).map(_.toDouble).toReal).asPredictor
    assertFeature[Passenger, Real](f1)(in = passenger, out = 1.toReal, name = "a")

    val f2 = FeatureBuilder[Passenger, Real]("b").extract(p => Option(p.getAge).map(_.toDouble).toReal).asResponse
    assertFeature[Passenger, Real](f2)(in = passenger, isResponse = true, out = 1.toReal, name = "b")
  }

  it should "build a simple feature using macro" in {
    val feature = FeatureBuilder.Real[Passenger].extract(p => Option(p.getAge).map(_.toDouble).toReal).asResponse
    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, isResponse = true)
  }

  it should "build a simple feature from Row with a custom name" in {
    val feature = FeatureBuilder.fromRow[Text](name = "feat", Some(1)).asPredictor
    assertFeature[Row, Text](feature)(name = "feat", in = Row(1.0, "2"), out = "2".toText, isResponse = false)
  }

  it should "build a simple feature from Row using macro" in {
    val feature = FeatureBuilder.fromRow[Real](0).asResponse
    assertFeature[Row, Real](feature)(name = name, in = Row(1.0, "2"), out = 1.toReal, isResponse = true)
  }

  it should "return a default if extract throws an exception" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge / 0).map(_.toDouble).toReal, 123.toReal)
        .asResponse

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 123.toReal, isResponse = true)
  }

  it should "build an aggregated feature" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal).aggregate(MaxReal)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, aggregator = _ => MaxReal)
  }

  it should "build an aggregated feature with an aggregate window" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .window(new Duration(123))
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name,
      in = passenger, out = 1.toReal, aggregateWindow = Some(new Duration(123)))
  }

  it should "build an aggregated feature with a custom aggregator" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .aggregate(MaxReal)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal, aggregator = _ => MaxReal)
  }

  it should "build an aggregated feature with a custom aggregate function" in {
    val feature =
      FeatureBuilder.Real[Passenger]
        .extract(p => Option(p.getAge).map(_.toDouble).toReal)
        .aggregate((v1, _) => v1)
        .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal,
      aggregator = _ => feature.originStage.asInstanceOf[FeatureGeneratorStage[Passenger, Real]].aggregator
    )
  }

  it should "build an aggregated feature with a custom aggregate function with zero" in {
    val feature = FeatureBuilder.Real[Passenger]
      .extract(p => Option(p.getAge).map(_.toDouble).toReal)
      .aggregate(Real.empty.v, (v1, _) => v1)
      .asPredictor

    assertFeature[Passenger, Real](feature)(name = name, in = passenger, out = 1.toReal,
      aggregator = _ => feature.originStage.asInstanceOf[FeatureGeneratorStage[Passenger, Real]].aggregator
    )
  }

}

/**
 * Assert feature instance on a given input/output
 */
object assertFeature extends Matchers {

  /**
   * Assert feature instance on a given input/output
   *
   * @param f               feature to assert
   * @param in              input value
   * @param out             expected output value
   * @param name            expected name
   * @param isResponse      is expected to be a response
   * @param aggregator      expected aggregator
   * @param aggregateWindow expected aggregate window
   * @param tti             expected input typetag
   * @param wtt             expected output typetag
   * @tparam I input type
   * @tparam O output feature type
   */
  def apply[I, O <: FeatureType](f: FeatureLike[O])(
    in: I, out: O, name: String, isResponse: Boolean = false,
    aggregator: WeakTypeTag[O] => MonoidAggregator[Event[O], _, O] =
    (wtt: WeakTypeTag[O]) => MonoidAggregatorDefaults.aggregatorOf[O](wtt),
    aggregateWindow: Option[Duration] = None
  )(implicit tti: WeakTypeTag[I], wtt: WeakTypeTag[O]): Unit = {
    f.name shouldBe name
    f.isResponse shouldBe isResponse
    f.parents shouldBe Nil
    f.uid.startsWith(wtt.tpe.dealias.toString.split("\\.").last) shouldBe true
    f.wtt.tpe =:= wtt.tpe shouldBe true
    f.isRaw shouldBe true
    f.typeName shouldBe wtt.tpe.typeSymbol.fullName

    f.originStage shouldBe a[FeatureGeneratorStage[_, _ <: FeatureType]]
    val fg = f.originStage.asInstanceOf[FeatureGeneratorStage[I, O]]
    fg.tti shouldBe tti
    fg.aggregator shouldBe aggregator(wtt)
    fg.extractFn(in) shouldBe out
    fg.extractSource.nonEmpty shouldBe true // TODO we should eval the code here: eval(fg.extractSource)(in)
    fg.getOutputFeatureName shouldBe name
    fg.outputIsResponse shouldBe isResponse
    fg.aggregateWindow shouldBe aggregateWindow
    fg.uid.startsWith(classOf[FeatureGeneratorStage[I, O]].getSimpleName) shouldBe true
  }

}
